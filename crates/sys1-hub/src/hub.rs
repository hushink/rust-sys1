use crate::{
    HfSource, HttpSource, HubConfig, HubError, LocalSource, ModelRef, ModelScopeSource,
    ModelSource, SourceContext, download,
};
use indexmap::IndexMap;
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use sys1_core::ModelManifest;

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
        })
    }

    pub fn config(&self) -> &HubConfig {
        &self.config
    }

    /// Resolve a reference to a manifest without downloading model files.
    pub fn resolve(&self, reference: &str) -> Result<ModelManifest, HubError> {
        let reference: ModelRef = reference.parse()?;
        let source = self.source_for(&reference)?;
        self.resolve_ref(&reference, &source)
    }

    /// Resolve and materialize a model.
    pub fn pull(&self, reference: &str) -> Result<PulledModel, HubError> {
        let reference: ModelRef = reference.parse()?;
        let source = self.source_for(&reference)?;
        let manifest = self.resolve_ref(&reference, &source)?;
        let context = SourceContext {
            client: &self.client,
            config: &self.config,
        };

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
        fs::create_dir_all(&model_dir).map_err(|error| HubError::io(&model_dir, error))?;

        let lock = self.lock_for(&cache_key);
        let _guard = lock.lock().unwrap_or_else(|poisoned| poisoned.into_inner());

        let mut files = IndexMap::new();
        for file in &manifest.files {
            let destination = model_dir.join(&file.path);
            if download::verify_file(&destination, Some(file.size), file.sha256.as_deref())? {
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
            source.fetch(&reference, file, &destination, &context)?;
            download::verify_complete(&destination, Some(file.size), file.sha256.as_deref())?;
            files.insert(file.path.clone(), destination);
        }

        write_manifest(&model_dir, &manifest)?;
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
