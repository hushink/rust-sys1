use crate::{LayaModel, LayaOptions, LayaRouter};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
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
#[derive(Default)]
pub struct LayaFactory {
    models: HashMap<String, Registered>,
    routers: HashMap<String, RouterRegistered>,
}

impl LayaFactory {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn register(
        &mut self,
        name: impl Into<String>,
        directory: impl Into<PathBuf>,
        options: LayaOptions,
    ) {
        self.models.insert(
            name.into(),
            Registered {
                directory: directory.into(),
                options,
            },
        );
    }

    pub fn register_router(
        &mut self,
        name: impl Into<String>,
        root: impl Into<PathBuf>,
        options: LayaOptions,
    ) {
        self.routers.insert(
            name.into(),
            RouterRegistered {
                root: root.into(),
                options,
            },
        );
    }

    pub fn registered(&self) -> impl Iterator<Item = &str> {
        self.models
            .keys()
            .chain(self.routers.keys())
            .map(String::as_str)
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

        if let Some(router) = self.routers.get(&manifest.name) {
            return Ok(Arc::new(LayaRouter::new(
                &router.root,
                router.options.clone(),
            )));
        }

        let registered = self.models.get(&manifest.name).ok_or_else(|| {
            ModelError::load(format!(
                "no local Laya directory registered for model `{}`",
                manifest.name
            ))
        })?;
        let model = LayaModel::load_dir(&registered.directory, registered.options.clone())?;
        Ok(Arc::new(model))
    }
}
