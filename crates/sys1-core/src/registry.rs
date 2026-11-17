use crate::{Error, ManifestError, ModelFactory, ModelManifest, Questions};
use indexmap::IndexMap;
use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock, RwLock};
use std::time::{Duration, Instant};
use sys1_protocol::{ModelMetadata, State, SystemOneResponse};

/// Registry limits.
#[derive(Clone, Copy, Debug)]
pub struct RegistryConfig {
    /// Maximum number of resident models. In-use models may temporarily exceed
    /// this limit.
    pub max_loaded: usize,
    /// Idle time after which an unused model is evicted.
    pub keep_alive: Duration,
}

impl Default for RegistryConfig {
    fn default() -> Self {
        Self {
            max_loaded: 3,
            keep_alive: Duration::from_secs(300),
        }
    }
}

impl RegistryConfig {
    pub fn new(max_loaded: usize, keep_alive: Duration) -> Self {
        Self {
            max_loaded: max_loaded.max(1),
            keep_alive,
        }
    }
}

/// A runtime catalog of manifests plus a lazily-loaded model cache.
///
/// The catalog starts from the manifests passed to [`ModelRegistry::new`] and
/// can be extended or trimmed at runtime with [`ModelRegistry::register`] and
/// [`ModelRegistry::unregister`]. The configured manifests are authoritative:
/// they return on the next start.
pub struct ModelRegistry {
    catalog: RwLock<Catalog>,
    factory: Arc<dyn ModelFactory>,
    config: RegistryConfig,
    state: Mutex<RegistryState>,
}

struct Catalog {
    manifests: IndexMap<String, ModelManifest>,
    aliases: HashMap<String, String>,
}

impl Catalog {
    fn resolve(&self, name: &str) -> Option<(String, ModelManifest)> {
        if let Some(manifest) = self.manifests.get(name) {
            return Some((manifest.name.clone(), manifest.clone()));
        }
        let canonical = self.aliases.get(name)?;
        let manifest = self.manifests.get(canonical)?;
        Some((manifest.name.clone(), manifest.clone()))
    }

    fn insert(&mut self, manifest: ModelManifest) -> Result<(), ManifestError> {
        manifest.validate()?;
        if self.manifests.contains_key(&manifest.name) {
            return Err(ManifestError::DuplicateName(manifest.name));
        }
        if self.aliases.contains_key(&manifest.name) {
            return Err(ManifestError::AliasConflict(manifest.name));
        }
        for alias in &manifest.aliases {
            if self.manifests.contains_key(alias) {
                return Err(ManifestError::AliasConflict(alias.clone()));
            }
            if self.aliases.contains_key(alias) {
                return Err(ManifestError::DuplicateAlias(alias.clone()));
            }
        }
        for alias in &manifest.aliases {
            self.aliases.insert(alias.clone(), manifest.name.clone());
        }
        self.manifests.insert(manifest.name.clone(), manifest);
        Ok(())
    }

    fn remove(&mut self, name: &str) -> Option<String> {
        let (canonical, _) = self.resolve(name)?;
        self.manifests.shift_remove(&canonical);
        self.aliases.retain(|_, target| target != &canonical);
        Some(canonical)
    }
}

#[derive(Default)]
struct RegistryState {
    entries: IndexMap<String, Arc<Entry>>,
}

struct Entry {
    cell: OnceLock<Arc<dyn crate::DecisionModel>>,
    last_used: Mutex<Instant>,
}

impl Entry {
    fn new() -> Self {
        Self {
            cell: OnceLock::new(),
            last_used: Mutex::new(Instant::now()),
        }
    }

    fn touch(&self) {
        *self
            .last_used
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner()) = Instant::now();
    }

    fn last_used(&self) -> Instant {
        *self
            .last_used
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }
}

impl ModelRegistry {
    pub fn new(
        manifests: Vec<ModelManifest>,
        factory: Arc<dyn ModelFactory>,
        config: RegistryConfig,
    ) -> Result<Self, ManifestError> {
        let mut catalog = Catalog {
            manifests: IndexMap::new(),
            aliases: HashMap::new(),
        };
        for manifest in manifests {
            catalog.insert(manifest)?;
        }

        Ok(Self {
            catalog: RwLock::new(catalog),
            factory,
            config,
            state: Mutex::new(RegistryState::default()),
        })
    }

    pub fn config(&self) -> RegistryConfig {
        self.config
    }

    /// Canonical manifests in catalog order.
    pub fn manifests(&self) -> Vec<ModelManifest> {
        self.lock_catalog().manifests.values().cloned().collect()
    }

    /// Add a model to the runtime catalog. The registry factory must be able
    /// to load it; see [`crate::ModelResolver`].
    pub fn register(&self, manifest: ModelManifest) -> Result<(), ManifestError> {
        self.lock_catalog_mut().insert(manifest)
    }

    /// Remove a model (or alias) from the runtime catalog and drop its resident
    /// instance. Returns whether the catalog contained the model.
    ///
    /// Configured manifests come back on the next start; dynamically added
    /// manifests are gone unless they are added again.
    pub fn unregister(&self, name: &str) -> bool {
        let removed = self.lock_catalog_mut().remove(name);
        let Some(canonical) = removed else {
            return false;
        };
        let mut state = self.lock_state();
        state.entries.shift_remove(&canonical);
        true
    }

    /// Names and aliases exposed by `GET /v1/models`.
    pub fn list_models(&self) -> Vec<ModelMetadata> {
        let catalog = self.lock_catalog();
        let mut models = Vec::new();
        for manifest in catalog.manifests.values() {
            models.push(manifest.metadata());
            for alias in &manifest.aliases {
                models.push(ModelMetadata::new(
                    alias.clone(),
                    manifest.description.clone(),
                    manifest.release_date.clone(),
                ));
            }
        }
        models
    }

    /// Resolve a model name or alias to its canonical manifest.
    pub fn resolve(&self, name: &str) -> Option<(String, ModelManifest)> {
        self.lock_catalog().resolve(name)
    }

    /// Return a loaded model without triggering a load.
    pub fn try_acquire(&self, name: &str) -> Option<(String, Arc<dyn crate::DecisionModel>)> {
        let (canonical, _) = self.resolve(name)?;
        let entry = {
            let state = self.lock_state();
            state.entries.get(&canonical)?.clone()
        };
        let model = entry.cell.get()?.clone();
        entry.touch();
        Some((canonical, model))
    }

    /// Return whether a model is currently resident, without touching its LRU
    /// timestamp.
    pub fn is_loaded(&self, name: &str) -> bool {
        let Some((canonical, _)) = self.resolve(name) else {
            return false;
        };
        let state = self.lock_state();
        state
            .entries
            .get(&canonical)
            .is_some_and(|entry| entry.cell.get().is_some())
    }

    /// Load a model if needed and return a shared handle.
    pub fn acquire(&self, name: &str) -> Result<Arc<dyn crate::DecisionModel>, Error> {
        let (canonical, manifest) = self.resolve(name).ok_or_else(|| Error::ModelNotFound {
            model: name.to_owned(),
        })?;

        let entry = {
            let mut state = self.lock_state();
            let entry = state
                .entries
                .entry(canonical.clone())
                .or_insert_with(|| Arc::new(Entry::new()))
                .clone();
            entry.touch();
            entry
        };

        if let Some(model) = entry.cell.get() {
            return Ok(model.clone());
        }

        let model = match self.factory.load(&manifest) {
            Ok(model) => model,
            Err(source) => {
                self.remove_empty(&canonical);
                return Err(Error::Model {
                    model: canonical,
                    source,
                });
            }
        };

        let _ = entry.cell.set(model);
        let model = entry
            .cell
            .get()
            .expect("model was set immediately above")
            .clone();
        self.evict_if_needed(&canonical);
        Ok(model)
    }

    /// Validate, resolve, load, and run a prediction.
    pub fn predict(
        &self,
        name: &str,
        state: &State,
        questions: &Questions,
    ) -> Result<SystemOneResponse, Error> {
        questions.validate()?;
        self.maintenance();

        let (canonical, _) = self.resolve(name).ok_or_else(|| Error::ModelNotFound {
            model: name.to_owned(),
        })?;
        let model = self.acquire(&canonical)?;
        let output = model
            .predict(state, questions)
            .map_err(|source| Error::Model {
                model: canonical.clone(),
                source,
            })?;

        Ok(SystemOneResponse {
            model: canonical,
            answers: output.answers.into_map(),
            usage: output.usage,
        })
    }

    /// Unload a model, returning whether it was resident.
    pub fn unload(&self, name: &str) -> bool {
        let Some((canonical, _)) = self.resolve(name) else {
            return false;
        };
        let mut state = self.lock_state();
        state.entries.shift_remove(&canonical).is_some()
    }

    /// Canonical names of currently resident models.
    pub fn loaded_models(&self) -> Vec<String> {
        let state = self.lock_state();
        state
            .entries
            .iter()
            .filter(|(_, entry)| entry.cell.get().is_some())
            .map(|(name, _)| name.clone())
            .collect()
    }

    /// Evict models that have been idle beyond `keep_alive` and are not in use.
    pub fn maintenance(&self) {
        let now = Instant::now();
        let mut state = self.lock_state();
        state.entries.retain(|_, entry| {
            let idle = now.saturating_duration_since(entry.last_used()) > self.config.keep_alive;
            let unloaded_idle = entry.cell.get().is_none() && idle;
            let unused_idle = idle
                && entry
                    .cell
                    .get()
                    .is_some_and(|model| Arc::strong_count(model) == 1);
            !(unloaded_idle || unused_idle)
        });
    }

    fn remove_empty(&self, name: &str) {
        let mut state = self.lock_state();
        if state
            .entries
            .get(name)
            .is_some_and(|entry| entry.cell.get().is_none())
        {
            state.entries.shift_remove(name);
        }
    }

    fn evict_if_needed(&self, protected: &str) {
        let mut state = self.lock_state();
        while state.entries.len() > self.config.max_loaded {
            let candidate = state
                .entries
                .iter()
                .filter(|(name, _)| name.as_str() != protected)
                .filter(|(_, entry)| {
                    entry
                        .cell
                        .get()
                        .is_some_and(|model| Arc::strong_count(model) == 1)
                })
                .min_by_key(|(_, entry)| entry.last_used())
                .map(|(name, _)| name.clone());

            match candidate {
                Some(name) => {
                    state.entries.shift_remove(&name);
                }
                None => break,
            }
        }
    }

    fn lock_catalog(&self) -> std::sync::RwLockReadGuard<'_, Catalog> {
        self.catalog
            .read()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    fn lock_catalog_mut(&self) -> std::sync::RwLockWriteGuard<'_, Catalog> {
        self.catalog
            .write()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    fn lock_state(&self) -> std::sync::MutexGuard<'_, RegistryState> {
        self.state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }
}
