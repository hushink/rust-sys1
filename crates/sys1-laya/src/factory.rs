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

impl LayaFactory {
    fn registered_directory(&self, name: &str) -> Option<(PathBuf, LayaOptions)> {
        let models = self
            .models
            .read()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        models
            .get(name)
            .map(|registered| (registered.directory.clone(), registered.options.clone()))
    }

    fn registered_router(&self, name: &str) -> Option<(PathBuf, LayaOptions)> {
        let routers = self
            .routers
            .read()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        routers
            .get(name)
            .map(|registered| (registered.root.clone(), registered.options.clone()))
    }
}

impl ModelFactory for LayaFactory {
    fn load(&self, manifest: &ModelManifest) -> Result<Arc<dyn DecisionModel>, ModelError> {
        match (manifest.backend.as_str(), manifest.format.as_str()) {
            ("candle", "safetensors") => {
                if let Some((root, options)) = self.registered_router(&manifest.name) {
                    return Ok(Arc::new(LayaRouter::new(&root, options)));
                }
                let Some((directory, options)) = self.registered_directory(&manifest.name) else {
                    return Err(ModelError::load(format!(
                        "no local Laya directory registered for model `{}`",
                        manifest.name
                    )));
                };
                let model = LayaModel::load_dir(&directory, options)?;
                Ok(Arc::new(model))
            }
            ("onnx", "onnx") => {
                #[cfg(feature = "onnx")]
                {
                    let Some((directory, options)) = self.registered_directory(&manifest.name)
                    else {
                        return Err(ModelError::load(format!(
                            "no local Laya directory registered for model `{}`",
                            manifest.name
                        )));
                    };
                    let model = crate::onnx::LayaOnnxModel::load_dir(&directory, options)?;
                    Ok(Arc::new(model))
                }
                #[cfg(not(feature = "onnx"))]
                {
                    Err(ModelError::unsupported(format!(
                        "model `{}` is ONNX, but this build of sys1-laya has the `onnx` feature disabled",
                        manifest.name
                    )))
                }
            }
            (backend, format) => Err(ModelError::unsupported(format!(
                "Laya supports backend=candle/format=safetensors or backend=onnx/format=onnx, got backend={backend} format={format}"
            ))),
        }
    }
}
