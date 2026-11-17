use crate::{DecisionModel, DecisionOptions};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, RwLock};
use sys1_core::{DecisionModel as DecisionModelTrait, ModelError, ModelFactory, ModelManifest};

struct Registered {
    directory: PathBuf,
    options: DecisionOptions,
}

/// Registers Decision 1.0 checkpoints by catalog name and loads them lazily.
#[derive(Default)]
pub struct DecisionFactory {
    models: RwLock<HashMap<String, Registered>>,
}

impl DecisionFactory {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn register(
        &self,
        name: impl Into<String>,
        directory: impl Into<PathBuf>,
        options: DecisionOptions,
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

impl ModelFactory for DecisionFactory {
    fn load(&self, manifest: &ModelManifest) -> Result<Arc<dyn DecisionModelTrait>, ModelError> {
        if manifest.backend != "candle" && manifest.format != "safetensors" {
            return Err(ModelError::unsupported(format!(
                "Decision requires backend=\"candle\" and format=\"safetensors\", got \
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
                    "Decision model {:?} is not registered",
                    manifest.name
                ))
            })?;
        Ok(Arc::new(DecisionModel::load_dir(&directory, options)?))
    }
}

/// Whether `directory` is a Decision 1.0 checkpoint.
pub fn is_decision_directory(directory: &Path) -> bool {
    directory.join("decision_config.json").is_file()
        && directory.join("decision_head.safetensors").is_file()
        && directory.join("backbone/config.json").is_file()
        && std::fs::read_dir(directory.join("backbone"))
            .ok()
            .is_some_and(|entries| {
                entries.filter_map(|entry| entry.ok()).any(|entry| {
                    entry
                        .file_name()
                        .to_string_lossy()
                        .ends_with(".safetensors")
                })
            })
}
