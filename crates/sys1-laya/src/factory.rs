use crate::{LayaModel, LayaOptions, LayaRouter};
use std::collections::{BTreeSet, HashMap};
use std::path::PathBuf;
use std::sync::{Arc, RwLock};
use sys1_core::{DecisionModel, ModelError, ModelFactory, ModelManifest};

struct Registered {
    directory: PathBuf,
    options: LayaOptions,
}

struct RouterRegistered {
    root: PathBuf,
    options: LayaOptions,
}

/// Loads Laya checkpoints from local directories registered by model name.
///
/// Registrations can be added while the server is running; the registry only
/// requires that a model is registered before it is loaded.
#[derive(Default)]
pub struct LayaFactory {
    models: RwLock<HashMap<String, Registered>>,
    routers: RwLock<HashMap<String, RouterRegistered>>,
}

impl LayaFactory {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn register(
        &self,
        name: impl Into<String>,
        directory: impl Into<PathBuf>,
        options: LayaOptions,
    ) {
        self.models
            .write()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .entry(name.into())
            .or_insert_with(|| Registered {
                directory: directory.into(),
                options,
            });
    }

    pub fn register_router(
        &self,
        name: impl Into<String>,
        root: impl Into<PathBuf>,
        options: LayaOptions,
    ) {
        self.routers
            .write()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .entry(name.into())
            .or_insert_with(|| RouterRegistered {
                root: root.into(),
                options,
            });
    }

    /// Forget a checkpoint or router registration.
    pub fn unregister(&self, name: &str) -> bool {
        let checkpoint = self
            .models
            .write()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .remove(name)
            .is_some();
        let router = self
            .routers
            .write()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .remove(name)
            .is_some();
        checkpoint || router
    }

    pub fn registered(&self) -> Vec<String> {
        let mut names = BTreeSet::new();
        names.extend(
            self.models
                .read()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .keys()
                .cloned(),
        );
        names.extend(
            self.routers
                .read()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .keys()
                .cloned(),
        );
        names.into_iter().collect()
    }
}

impl ModelFactory for LayaFactory {
    fn load(&self, manifest: &ModelManifest) -> Result<Arc<dyn DecisionModel>, ModelError> {
        if manifest.backend != "candle" || manifest.format != "safetensors" {
            return Err(ModelError::unsupported(format!(
                "Laya requires backend=candle and format=safetensors, got backend={} format={}",
                manifest.backend, manifest.format
            )));
        }

        let router = {
            let routers = self
                .routers
                .read()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            routers
                .get(&manifest.name)
                .map(|registered| (registered.root.clone(), registered.options.clone()))
        };
        if let Some((root, options)) = router {
            return Ok(Arc::new(LayaRouter::new(&root, options)));
        }

        let registered = {
            let models = self
                .models
                .read()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            models
                .get(&manifest.name)
                .map(|registered| (registered.directory.clone(), registered.options.clone()))
        };
        let Some((directory, options)) = registered else {
            return Err(ModelError::load(format!(
                "no local Laya directory registered for model `{}`",
                manifest.name
            )));
        };
        let model = LayaModel::load_dir(&directory, options)?;
        Ok(Arc::new(model))
    }
}
