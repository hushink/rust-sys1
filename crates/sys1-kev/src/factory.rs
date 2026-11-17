use crate::{KevModel, KevOptions};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, RwLock};
use sys1_core::{DecisionModel, ModelError, ModelFactory, ModelManifest};

struct Registered {
    directory: PathBuf,
    options: KevOptions,
}

/// Registers Kev checkpoints by catalog name and loads them lazily.
#[derive(Default)]
pub struct KevFactory {
    models: RwLock<HashMap<String, Registered>>,
}

impl KevFactory {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn register(
        &self,
        name: impl Into<String>,
        directory: impl Into<PathBuf>,
        options: KevOptions,
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

    pub fn unregister(&self, name: &str) -> bool {
        self.models
            .write()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .remove(name)
            .is_some()
    }

    pub fn registered(&self) -> Vec<String> {
        let mut names: Vec<_> = self
            .models
            .read()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .keys()
            .cloned()
            .collect();
        names.sort();
        names
    }
}

impl ModelFactory for KevFactory {
    fn load(&self, manifest: &ModelManifest) -> Result<Arc<dyn DecisionModel>, ModelError> {
        if manifest.backend != "candle" && manifest.format != "safetensors" {
            return Err(ModelError::unsupported(format!(
                "Kev requires backend=\"candle\" and format=\"safetensors\", got \
                 backend={:?} format={:?}",
                manifest.backend, manifest.format
            )));
        }
        let (directory, options) = self
            .models
            .read()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .get(&manifest.name)
            .map(|registered| (registered.directory.clone(), registered.options.clone()))
            .ok_or_else(|| {
                ModelError::load(format!("Kev model {:?} is not registered", manifest.name))
            })?;
        Ok(Arc::new(KevModel::load_dir(&directory, options)?))
    }
}

/// Whether `directory` holds the Kev adapter, head, and tokenizer. The Qwen3.5
/// base may live beside them, under `base/`, or be pulled from the repository
/// declared in the adapter metadata.
pub fn is_kev_adapter(directory: &Path) -> bool {
    directory.join("adapter_model.safetensors").is_file()
        && directory.join("adapter_config.json").is_file()
        && directory.join("head.pt").is_file()
        && directory.join("tokenizer.json").is_file()
}

/// Whether a Qwen3.5 base is present beside the adapter or under `base/`.
pub fn has_kev_base(directory: &Path) -> bool {
    crate::local_base(directory).is_some()
}

/// Whether `directory` is a Kev checkpoint directory.
pub fn is_kev_directory(directory: &Path) -> bool {
    is_kev_adapter(directory)
}
