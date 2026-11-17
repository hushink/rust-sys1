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

/// Whether `directory` holds a Kev adapter, head, tokenizer, and a Qwen3.5 base
/// either beside them or under `base/`.
pub fn is_kev_directory(directory: &Path) -> bool {
    let adapter = directory.join("adapter_model.safetensors").is_file()
        && directory.join("adapter_config.json").is_file()
        && directory.join("head.pt").is_file()
        && directory.join("tokenizer.json").is_file();
    if !adapter {
        return false;
    }
    [directory.to_path_buf(), directory.join("base")]
        .into_iter()
        .any(|base| {
            base.join("config.json").is_file()
                && std::fs::read_dir(&base).ok().is_some_and(|entries| {
                    entries.filter_map(|entry| entry.ok()).any(|entry| {
                        entry
                            .file_name()
                            .to_string_lossy()
                            .ends_with(".safetensors")
                    })
                })
        })
}
