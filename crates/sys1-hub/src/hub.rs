use crate::{
    HfSource, HttpSource, HubConfig, HubError, LocalSource, ModelRef, ModelScopeSource,
    ModelSource, PullObserver, SourceContext, download, progress::FileProgress,
};
use indexmap::IndexMap;
use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};
use sys1_core::ModelManifest;

/// A file recorded in a cache entry.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CacheFile {
    pub path: String,
    pub size: u64,
}

/// A model directory under `<cache_dir>/models`.
#[derive(Clone, Debug)]
pub struct CacheEntry {
    pub key: String,
    pub name: String,
    pub directory: PathBuf,
    /// Actual bytes used by the directory, including `model.json`.
    pub size_bytes: u64,
    /// Manifest modification time; updated when a pull completes.
    pub last_used: SystemTime,
    pub files: Vec<CacheFile>,
    pub resident: bool,
    /// Whether the entry can be evicted right now. Resident models and active
    /// pulls are not evictable.
    pub evictable: bool,
}

/// Cache totals for `/api/cache` and `sys1 doctor`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CacheUsage {
    pub limit_bytes: Option<u64>,
    pub used_bytes: u64,
    pub free_bytes: Option<u64>,
    pub reclaimable_bytes: u64,
    pub model_count: usize,
    pub resident_count: usize,
}

/// One entry removed by quota enforcement.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EvictedCacheEntry {
    pub key: String,
    pub name: String,
    pub size_bytes: u64,
}

/// Result of [`Hub::enforce_quota`].
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct CacheEviction {
    pub entries: Vec<EvictedCacheEntry>,
    pub freed_bytes: u64,
}

/// Lets the hub avoid evicting models that an embedding application still has
/// loaded. `sys1-cli` and `InferenceEngine` install a registry-backed
/// implementation; a bare [`Hub`] has no resident models.
pub trait CacheResidency: Send + Sync {
    fn is_resident(&self, model_name: &str) -> bool;
}

#[derive(Default)]
struct CacheState {
    active_pulls: HashSet<String>,
}

struct ActivePull<'a> {
    hub: &'a Hub,
    key: String,
}

impl Drop for ActivePull<'_> {
    fn drop(&mut self) {
        self.hub
            .cache_state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .active_pulls
            .remove(&self.key);
    }
}

/// A resolved and materialized model.
#[derive(Clone, Debug)]
pub struct PulledModel {
    pub manifest: ModelManifest,
    pub directory: PathBuf,
    pub files: IndexMap<String, PathBuf>,
}

/// Resolves model references and materializes their files into the local cache.
pub struct Hub {
    config: HubConfig,
    client: reqwest::blocking::Client,
    sources: Vec<Arc<dyn ModelSource>>,
    locks: Mutex<HashMap<String, Arc<Mutex<()>>>>,
    residency: Mutex<Option<Arc<dyn CacheResidency>>>,
    cache_state: Mutex<CacheState>,
}

impl Hub {
    pub fn new(config: HubConfig) -> Result<Self, HubError> {
        Self::with_sources(
            config,
            vec![
                Arc::new(LocalSource),
                Arc::new(HttpSource),
                Arc::new(HfSource),
                Arc::new(ModelScopeSource),
            ],
        )
    }

    pub fn with_sources(
        config: HubConfig,
        sources: Vec<Arc<dyn ModelSource>>,
    ) -> Result<Self, HubError> {
        let client = reqwest::blocking::Client::builder()
            .timeout(config.timeout)
            .user_agent(concat!("rust-sys1/", env!("CARGO_PKG_VERSION")))
            .build()
            .map_err(|error| HubError::Http {
                url: "client".to_owned(),
                message: error.to_string(),
            })?;
        Ok(Self {
            config,
            client,
            sources,
            locks: Mutex::new(HashMap::new()),
            residency: Mutex::new(None),
            cache_state: Mutex::new(CacheState::default()),
        })
    }

    pub fn config(&self) -> &HubConfig {
        &self.config
    }

    /// Install the residency provider used by quota eviction.
    ///
    /// Without a provider every cache entry is treated as non-resident.
    pub fn set_residency(&self, residency: Arc<dyn CacheResidency>) {
        *self
            .residency
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner()) = Some(residency);
    }

    /// Cached model directories with sizes, timestamps, and eviction state.
    pub fn cache_entries(&self) -> Result<Vec<CacheEntry>, HubError> {
        let state = self
            .cache_state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        self.scan_entries(&state)
    }

    /// Current disk usage and quota headroom.
    pub fn cache_usage(&self) -> Result<CacheUsage, HubError> {
        let state = self
            .cache_state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let entries = self.scan_entries(&state)?;
        let used_bytes = entries.iter().map(|entry| entry.size_bytes).sum();
        let reclaimable_bytes = entries
            .iter()
            .filter(|entry| entry.evictable)
            .map(|entry| entry.size_bytes)
            .sum();
        let limit_bytes = self.active_limit();
        Ok(CacheUsage {
            limit_bytes,
            used_bytes,
            free_bytes: limit_bytes.map(|limit| limit.saturating_sub(used_bytes)),
            reclaimable_bytes,
            model_count: entries.len(),
            resident_count: entries.iter().filter(|entry| entry.resident).count(),
        })
    }

    /// Evict least-recently-used non-resident entries until
    /// `used + additional_bytes` fits the configured quota.
    ///
    /// Returns an empty report when no quota is configured. A model larger
    /// than the quota, or one blocked by resident/active entries, returns
    /// [`HubError::CacheQuotaExceeded`].
    pub fn enforce_quota(&self, additional_bytes: u64) -> Result<CacheEviction, HubError> {
        let state = self
            .cache_state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        self.enforce_quota_locked(additional_bytes, None, &state)
    }

    /// Resolve a reference to a manifest without downloading model files.
    pub fn resolve(&self, reference: &str) -> Result<ModelManifest, HubError> {
        let reference: ModelRef = reference.parse()?;
        let source = self.source_for(&reference)?;
        self.resolve_ref(&reference, &source)
    }

    /// Resolve and materialize a model.
    pub fn pull(&self, reference: &str) -> Result<PulledModel, HubError> {
        self.pull_with_observer(reference, None)
    }

    /// Resolve and materialize a model, reporting per-file progress to
    /// `observer`. Without an observer this is [`Hub::pull`].
    pub fn pull_with_observer(
        &self,
        reference: &str,
        observer: Option<&dyn PullObserver>,
    ) -> Result<PulledModel, HubError> {
        let reference: ModelRef = reference.parse()?;
        let source = self.source_for(&reference)?;
        let manifest = self.resolve_ref(&reference, &source)?;

        if let Some(directory) = source.local_directory(&reference) {
            let mut files = IndexMap::new();
            for file in &manifest.files {
                let path = source.local_file(&reference, file).ok_or_else(|| {
                    HubError::Unsupported(format!(
                        "local file mapping is missing for {}",
                        file.path
                    ))
                })?;
                download::verify_complete(&path, Some(file.size), file.sha256.as_deref())?;
                files.insert(file.path.clone(), path);
            }
            return Ok(PulledModel {
                manifest,
                directory,
                files,
            });
        }

        let cache_key = source.cache_key(&reference);
        let model_dir = self.config.cache_dir.join("models").join(&cache_key);

        let lock = self.lock_for(&cache_key);
        let _guard = lock.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
        let required = required_bytes(&manifest, &model_dir)?;
        let _active = self.begin_pull(&cache_key, required)?;

        fs::create_dir_all(&model_dir).map_err(|error| HubError::io(&model_dir, error))?;

        let total_bytes: u64 = manifest.files.iter().map(|file| file.size).sum();
        let file_count = manifest.files.len();
        let mut materialized = 0_u64;
        let mut files = IndexMap::new();
        for (index, file) in manifest.files.iter().enumerate() {
            let destination = model_dir.join(&file.path);
            let progress = observer.map(|observer| {
                FileProgress::new(
                    observer,
                    &file.path,
                    index + 1,
                    file_count,
                    file.size,
                    materialized,
                    total_bytes,
                )
            });
            if download::verify_file(&destination, Some(file.size), file.sha256.as_deref())? {
                if let Some(progress) = &progress {
                    progress.report(file.size);
                }
                materialized += file.size;
                files.insert(file.path.clone(), destination);
                continue;
            }
            if self.config.offline {
                return Err(HubError::Offline(format!(
                    "{} is not cached at {}",
                    file.path,
                    destination.display()
                )));
            }
            let context = SourceContext {
                client: &self.client,
                config: &self.config,
                progress: progress.as_ref(),
            };
            source.fetch(&reference, file, &destination, &context)?;
            download::verify_complete(&destination, Some(file.size), file.sha256.as_deref())?;
            if let Some(progress) = &progress {
                progress.report(file.size);
            }
            materialized += file.size;
            files.insert(file.path.clone(), destination);
        }

        write_manifest(&model_dir, &manifest)?;
        touch_manifest(&model_dir);
        Ok(PulledModel {
            manifest,
            directory: model_dir,
            files,
        })
    }

    /// Cache directory that would hold this reference.
    pub fn cache_location(&self, reference: &str) -> Result<PathBuf, HubError> {
        let reference: ModelRef = reference.parse()?;
        let source = self.source_for(&reference)?;
        Ok(self
            .config
            .cache_dir
            .join("models")
            .join(source.cache_key(&reference)))
    }

    /// Remove a cached model and its cached manifest.
    pub fn remove(&self, reference: &str) -> Result<bool, HubError> {
        let reference: ModelRef = reference.parse()?;
        let source = self.source_for(&reference)?;
        let cache_key = source.cache_key(&reference);
        if let Some(model) = self.resident_model(&cache_key) {
            return Err(HubError::CacheResident {
                key: cache_key,
                model,
            });
        }
        let state = self
            .cache_state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if state.active_pulls.contains(&cache_key) {
            return Err(HubError::CacheBusy { key: cache_key });
        }
        let directory = self.config.cache_dir.join("models").join(&cache_key);
        let removed = if directory.exists() {
            fs::remove_dir_all(&directory).map_err(|error| HubError::io(&directory, error))?;
            true
        } else {
            false
        };
        let manifest = self.cache_manifest_path(&cache_key);
        if manifest.exists() {
            fs::remove_file(&manifest).map_err(|error| HubError::io(&manifest, error))?;
        }
        Ok(removed)
    }

    /// Remove a cached model by its cache key (the directory name shown by
    /// [`Hub::list_cached`]).
    ///
    /// The key is validated against path traversal before any filesystem
    /// operation.
    pub fn remove_cached(&self, key: &str) -> Result<bool, HubError> {
        if !is_cache_key(key) {
            return Err(HubError::InvalidData {
                url: key.to_owned(),
                message: "invalid model cache key".to_owned(),
            });
        }
        if let Some(model) = self.resident_model(key) {
            return Err(HubError::CacheResident {
                key: key.to_owned(),
                model,
            });
        }
        let state = self
            .cache_state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if state.active_pulls.contains(key) {
            return Err(HubError::CacheBusy {
                key: key.to_owned(),
            });
        }
        let directory = self.config.cache_dir.join("models").join(key);
        let removed = if directory.is_dir() {
            fs::remove_dir_all(&directory).map_err(|error| HubError::io(&directory, error))?;
            true
        } else {
            false
        };
        let manifest = self.cache_manifest_path(key);
        if manifest.is_file() {
            fs::remove_file(&manifest).map_err(|error| HubError::io(&manifest, error))?;
        }
        Ok(removed)
    }

    /// List models present in the local cache.
    pub fn list_cached(&self) -> Result<Vec<PulledModel>, HubError> {
        let models_dir = self.config.cache_dir.join("models");
        if !models_dir.exists() {
            return Ok(Vec::new());
        }

        let mut models = Vec::new();
        let entries =
            fs::read_dir(&models_dir).map_err(|error| HubError::io(&models_dir, error))?;
        for entry in entries {
            let entry = entry.map_err(|error| HubError::io(&models_dir, error))?;
            let directory = entry.path();
            if !directory.is_dir() {
                continue;
            }
            let manifest_path = directory.join("model.json");
            if !manifest_path.exists() {
                continue;
            }
            let bytes =
                fs::read(&manifest_path).map_err(|error| HubError::io(&manifest_path, error))?;
            let manifest: ModelManifest =
                serde_json::from_slice(&bytes).map_err(|error| HubError::Json {
                    url: manifest_path.to_string_lossy().into_owned(),
                    message: error.to_string(),
                })?;
            manifest.validate()?;

            let mut files = IndexMap::new();
            for file in &manifest.files {
                let path = directory.join(&file.path);
                if path.exists() {
                    files.insert(file.path.clone(), path);
                }
            }
            models.push(PulledModel {
                manifest,
                directory,
                files,
            });
        }
        models.sort_by(|left, right| left.manifest.name.cmp(&right.manifest.name));
        Ok(models)
    }

    fn active_limit(&self) -> Option<u64> {
        self.config.max_cache_bytes.filter(|limit| *limit > 0)
    }

    fn begin_pull(&self, key: &str, additional: u64) -> Result<ActivePull<'_>, HubError> {
        let mut state = self
            .cache_state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        state.active_pulls.insert(key.to_owned());
        let eviction = match self.enforce_quota_locked(additional, Some(key), &state) {
            Ok(eviction) => eviction,
            Err(error) => {
                state.active_pulls.remove(key);
                return Err(error);
            }
        };
        if !eviction.entries.is_empty() {
            tracing::info!(
                entries = eviction.entries.len(),
                freed_bytes = eviction.freed_bytes,
                "evicted cache entries for quota"
            );
        }
        Ok(ActivePull {
            hub: self,
            key: key.to_owned(),
        })
    }

    fn enforce_quota_locked(
        &self,
        additional: u64,
        current_key: Option<&str>,
        state: &CacheState,
    ) -> Result<CacheEviction, HubError> {
        let Some(limit) = self.active_limit() else {
            return Ok(CacheEviction::default());
        };
        let entries = self.scan_entries(state)?;
        let used: u64 = entries.iter().map(|entry| entry.size_bytes).sum();
        let reclaimable: u64 = entries
            .iter()
            .filter(|entry| entry.evictable)
            .map(|entry| entry.size_bytes)
            .sum();
        if additional > limit {
            return Err(HubError::CacheQuotaExceeded {
                limit,
                used,
                additional,
                reclaimable,
            });
        }
        if used.saturating_add(additional) <= limit {
            return Ok(CacheEviction::default());
        }

        let mut result = CacheEviction::default();
        let mut evicted = HashSet::new();
        for entry in entries
            .iter()
            .filter(|entry| entry.evictable && current_key != Some(entry.key.as_str()))
        {
            if used
                .saturating_sub(result.freed_bytes)
                .saturating_add(additional)
                <= limit
            {
                break;
            }
            if self.resident_model(&entry.key).is_some() {
                continue;
            }
            self.remove_entry(entry)?;
            result.freed_bytes = result.freed_bytes.saturating_add(entry.size_bytes);
            evicted.insert(entry.key.clone());
            result.entries.push(EvictedCacheEntry {
                key: entry.key.clone(),
                name: entry.name.clone(),
                size_bytes: entry.size_bytes,
            });
        }

        let remaining_used = used.saturating_sub(result.freed_bytes);
        if remaining_used.saturating_add(additional) > limit {
            let reclaimable = entries
                .iter()
                .filter(|entry| {
                    entry.evictable
                        && !evicted.contains(&entry.key)
                        && self.resident_model(&entry.key).is_none()
                })
                .map(|entry| entry.size_bytes)
                .sum();
            return Err(HubError::CacheQuotaExceeded {
                limit,
                used: remaining_used,
                additional,
                reclaimable,
            });
        }
        Ok(result)
    }

    fn scan_entries(&self, state: &CacheState) -> Result<Vec<CacheEntry>, HubError> {
        let models_dir = self.config.cache_dir.join("models");
        if !models_dir.is_dir() {
            return Ok(Vec::new());
        }
        let residency = self
            .residency
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .clone();
        let mut entries = Vec::new();
        let read_dir =
            fs::read_dir(&models_dir).map_err(|error| HubError::io(&models_dir, error))?;
        for entry in read_dir {
            let entry = entry.map_err(|error| HubError::io(&models_dir, error))?;
            let directory = entry.path();
            if !directory.is_dir() {
                continue;
            }
            let Some(key) = directory
                .file_name()
                .and_then(|name| name.to_str())
                .filter(|key| is_cache_key(key))
                .map(str::to_owned)
            else {
                continue;
            };

            let manifest_path = directory.join("model.json");
            let (name, files) = match fs::read(&manifest_path)
                .ok()
                .and_then(|bytes| serde_json::from_slice::<ModelManifest>(&bytes).ok())
                .filter(|manifest| manifest.validate().is_ok())
            {
                Some(manifest) => {
                    let files = manifest
                        .files
                        .iter()
                        .map(|file| {
                            let size = fs::metadata(directory.join(&file.path))
                                .map(|metadata| metadata.len())
                                .unwrap_or(file.size);
                            CacheFile {
                                path: file.path.clone(),
                                size,
                            }
                        })
                        .collect();
                    (manifest.name, files)
                }
                None => (key.clone(), Vec::new()),
            };

            let size_bytes = directory_size(&directory)?;
            let last_used = fs::metadata(&manifest_path)
                .and_then(|metadata| metadata.modified())
                .or_else(|_| fs::metadata(&directory).and_then(|metadata| metadata.modified()))
                .unwrap_or(UNIX_EPOCH);
            let resident = residency
                .as_ref()
                .map(|residency| residency.is_resident(&name))
                .unwrap_or(false);
            entries.push(CacheEntry {
                key: key.clone(),
                name,
                directory,
                size_bytes,
                last_used,
                files,
                resident,
                evictable: !resident && !state.active_pulls.contains(&key),
            });
        }
        entries.sort_by(|left, right| {
            left.last_used
                .cmp(&right.last_used)
                .then_with(|| left.key.cmp(&right.key))
        });
        Ok(entries)
    }

    fn resident_model(&self, key: &str) -> Option<String> {
        let residency = self
            .residency
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .clone()?;
        let manifest_path = self
            .config
            .cache_dir
            .join("models")
            .join(key)
            .join("model.json");
        let name = fs::read(&manifest_path)
            .ok()
            .and_then(|bytes| serde_json::from_slice::<ModelManifest>(&bytes).ok())
            .filter(|manifest| manifest.validate().is_ok())
            .map(|manifest| manifest.name)
            .unwrap_or_else(|| key.to_owned());
        residency.is_resident(&name).then_some(name)
    }

    fn remove_entry(&self, entry: &CacheEntry) -> Result<(), HubError> {
        if entry.directory.is_dir() {
            fs::remove_dir_all(&entry.directory)
                .map_err(|error| HubError::io(&entry.directory, error))?;
        }
        let manifest = self.cache_manifest_path(&entry.key);
        if manifest.is_file() {
            fs::remove_file(&manifest).map_err(|error| HubError::io(&manifest, error))?;
        }
        tracing::info!(
            key = %entry.key,
            model = %entry.name,
            size_bytes = entry.size_bytes,
            "evicted model from cache"
        );
        Ok(())
    }

    fn resolve_ref(
        &self,
        reference: &ModelRef,
        source: &Arc<dyn ModelSource>,
    ) -> Result<ModelManifest, HubError> {
        let cache_key = source.cache_key(reference);
        let cache_path = self.cache_manifest_path(&cache_key);

        if self.config.offline && source.local_directory(reference).is_none() {
            let bytes = fs::read(&cache_path).map_err(|error| {
                if error.kind() == std::io::ErrorKind::NotFound {
                    HubError::Offline(format!(
                        "manifest for {} is not cached at {}",
                        reference.raw_key(),
                        cache_path.display()
                    ))
                } else {
                    HubError::io(&cache_path, error)
                }
            })?;
            let manifest: ModelManifest =
                serde_json::from_slice(&bytes).map_err(|error| HubError::Json {
                    url: cache_path.to_string_lossy().into_owned(),
                    message: error.to_string(),
                })?;
            manifest.validate()?;
            return Ok(manifest);
        }

        let context = SourceContext {
            client: &self.client,
            config: &self.config,
            progress: None,
        };
        let manifest = source.resolve(reference, &context)?;
        manifest.validate()?;
        write_manifest_cache(&cache_path, &manifest)?;
        Ok(manifest)
    }

    fn source_for(&self, reference: &ModelRef) -> Result<Arc<dyn ModelSource>, HubError> {
        self.sources
            .iter()
            .find(|source| source.matches(reference))
            .cloned()
            .ok_or_else(|| HubError::SourceNotFound(reference.scheme.clone()))
    }

    fn cache_manifest_path(&self, cache_key: &str) -> PathBuf {
        self.config
            .cache_dir
            .join("manifests")
            .join(format!("{cache_key}.json"))
    }

    fn lock_for(&self, cache_key: &str) -> Arc<Mutex<()>> {
        let mut locks = self
            .locks
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        locks
            .entry(cache_key.to_owned())
            .or_insert_with(|| Arc::new(Mutex::new(())))
            .clone()
    }
}

fn required_bytes(manifest: &ModelManifest, directory: &Path) -> Result<u64, HubError> {
    let mut required = 0_u64;
    for file in &manifest.files {
        let destination = directory.join(&file.path);
        let existing = fs::metadata(&destination)
            .map(|metadata| metadata.len())
            .unwrap_or(0);
        let partial = fs::metadata(download::partial_path(&destination))
            .map(|metadata| metadata.len())
            .unwrap_or(0);
        let reusable = existing.min(file.size).max(partial.min(file.size));
        required = required.saturating_add(file.size.saturating_sub(reusable));
    }
    Ok(required)
}

fn directory_size(directory: &Path) -> Result<u64, HubError> {
    let mut total = 0_u64;
    for entry in fs::read_dir(directory).map_err(|error| HubError::io(directory, error))? {
        let entry = entry.map_err(|error| HubError::io(directory, error))?;
        let path = entry.path();
        let metadata = fs::symlink_metadata(&path).map_err(|error| HubError::io(&path, error))?;
        if metadata.is_dir() {
            total = total.saturating_add(directory_size(&path)?);
        } else {
            total = total.saturating_add(metadata.len());
        }
    }
    Ok(total)
}

fn touch_manifest(directory: &Path) {
    let path = directory.join("model.json");
    let Ok(file) = fs::File::options().read(true).write(true).open(&path) else {
        return;
    };
    let _ = file.set_modified(SystemTime::now());
}

fn is_cache_key(key: &str) -> bool {
    !key.is_empty()
        && key != "."
        && key != ".."
        && key
            .chars()
            .all(|ch| ch.is_ascii_alphanumeric() || matches!(ch, '-' | '_' | '.'))
}

fn write_manifest(directory: &Path, manifest: &ModelManifest) -> Result<(), HubError> {
    let path = directory.join("model.json");
    write_json_atomic(&path, manifest)
}

fn write_manifest_cache(path: &Path, manifest: &ModelManifest) -> Result<(), HubError> {
    write_json_atomic(path, manifest)
}

fn write_json_atomic(path: &Path, manifest: &ModelManifest) -> Result<(), HubError> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(|error| HubError::io(parent, error))?;
    }
    let bytes = serde_json::to_vec_pretty(manifest).map_err(|error| HubError::InvalidData {
        url: path.to_string_lossy().into_owned(),
        message: error.to_string(),
    })?;
    let temporary = path.with_extension("json.tmp");
    fs::write(&temporary, bytes).map_err(|error| HubError::io(&temporary, error))?;
    fs::rename(&temporary, path).map_err(|error| HubError::io(path, error))
}
