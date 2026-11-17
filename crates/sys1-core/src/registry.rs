use crate::{Error, ManifestError, ModelFactory, ModelManifest, Questions};
use indexmap::IndexMap;
use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};
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
pub struct ModelRegistry {
    manifests: IndexMap<String, ModelManifest>,
    aliases: HashMap<String, String>,
    factory: Arc<dyn ModelFactory>,
    config: RegistryConfig,
    state: Mutex<RegistryState>,
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
        let mut by_name = IndexMap::new();
        let mut aliases = HashMap::new();

        for manifest in manifests {
            manifest.validate()?;
            if by_name.contains_key(&manifest.name) {
                return Err(ManifestError::DuplicateName(manifest.name));
            }
            for alias in &manifest.aliases {
                if aliases.contains_key(alias) {
                    return Err(ManifestError::DuplicateAlias(alias.clone()));
                }
                aliases.insert(alias.clone(), manifest.name.clone());
            }
            by_name.insert(manifest.name.clone(), manifest);
        }

        for alias in aliases.keys() {
            if by_name.contains_key(alias) {
                return Err(ManifestError::AliasConflict(alias.clone()));
            }
        }

        Ok(Self {
            manifests: by_name,
            aliases,
            factory,
            config,
            state: Mutex::new(RegistryState::default()),
        })
    }

    pub fn config(&self) -> RegistryConfig {
        self.config
    }

    pub fn manifests(&self) -> &IndexMap<String, ModelManifest> {
        &self.manifests
    }

    /// Names and aliases exposed by `GET /v1/models`.
    pub fn list_models(&self) -> Vec<ModelMetadata> {
        let mut models = Vec::new();
        for manifest in self.manifests.values() {
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
    pub fn resolve(&self, name: &str) -> Option<(&str, &ModelManifest)> {
        if let Some(manifest) = self.manifests.get(name) {
            return Some((manifest.name.as_str(), manifest));
        }
        let canonical = self.aliases.get(name)?;
        let manifest = self.manifests.get(canonical)?;
        Some((manifest.name.as_str(), manifest))
    }

    /// Return a loaded model without triggering a load.
    pub fn try_acquire(&self, name: &str) -> Option<(String, Arc<dyn crate::DecisionModel>)> {
        let (canonical, _) = self.resolve(name)?;
        let entry = {
            let state = self.lock_state();
            state.entries.get(canonical)?.clone()
        };
        let model = entry.cell.get()?.clone();
        entry.touch();
        Some((canonical.to_owned(), model))
    }

    /// Load a model if needed and return a shared handle.
    pub fn acquire(&self, name: &str) -> Result<Arc<dyn crate::DecisionModel>, Error> {
        let (canonical, manifest) = self.resolve(name).ok_or_else(|| Error::ModelNotFound {
            model: name.to_owned(),
        })?;
        let canonical = canonical.to_owned();
        let manifest = manifest.clone();

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
        let canonical = canonical.to_owned();
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
        let canonical = canonical.to_owned();
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

    fn lock_state(&self) -> std::sync::MutexGuard<'_, RegistryState> {
        self.state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }
}
