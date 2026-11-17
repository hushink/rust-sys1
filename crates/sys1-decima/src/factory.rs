use crate::{DecimaModel, DecimaOptions};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, RwLock};
use sys1_core::{DecisionModel, ModelError, ModelFactory, ModelManifest};

struct Registered {
    directory: PathBuf,
    options: DecimaOptions,
}

/// Registers Decima checkpoints by catalog name and loads them lazily.
#[derive(Default)]
pub struct DecimaFactory {
    models: RwLock<HashMap<String, Registered>>,
}

impl DecimaFactory {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn register(
        &self,
        name: impl Into<String>,
        directory: impl Into<PathBuf>,
        options: DecimaOptions,
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

impl ModelFactory for DecimaFactory {
    fn load(&self, manifest: &ModelManifest) -> Result<Arc<dyn DecisionModel>, ModelError> {
        if manifest.backend != "candle" && manifest.format != "safetensors" {
            return Err(ModelError::unsupported(format!(
                "Decima requires backend=\"candle\" and format=\"safetensors\", got \
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
                ModelError::load(format!(
                    "Decima model {:?} is not registered",
                    manifest.name
                ))
            })?;
        Ok(Arc::new(DecimaModel::load_dir(&directory, options)?))
    }
}

/// Whether `directory` is a Decima checkpoint, either the repository root
/// (with `pytorch/`) or the `pytorch/` directory itself.
pub fn is_decima_directory(directory: &Path) -> bool {
    let checkpoint = if directory.join("pytorch/decima.json").is_file() {
        directory.join("pytorch")
    } else {
        directory.to_path_buf()
    };
    checkpoint.join("decima.json").is_file()
        && checkpoint.join("head.safetensors").is_file()
        && checkpoint
            .join("encoder")
            .join("model.safetensors")
            .is_file()
        && (checkpoint.join("encoder/tokenizer.json").is_file()
            || checkpoint.join("tokenizer.json").is_file())
}
