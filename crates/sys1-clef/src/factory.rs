use crate::{ClefModel, ClefOptions};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, RwLock};
use sys1_core::{DecisionModel, ModelError, ModelFactory, ModelManifest};

struct Registered {
    directory: PathBuf,
    options: ClefOptions,
}

/// Registers Clef checkpoints by catalog name and loads them lazily.
#[derive(Default)]
pub struct ClefFactory {
    models: RwLock<HashMap<String, Registered>>,
}

impl ClefFactory {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn register(
        &self,
        name: impl Into<String>,
        directory: impl Into<PathBuf>,
        options: ClefOptions,
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

impl ModelFactory for ClefFactory {
    fn load(&self, manifest: &ModelManifest) -> Result<Arc<dyn DecisionModel>, ModelError> {
        if manifest.backend != "candle" && manifest.format != "safetensors" {
            return Err(ModelError::unsupported(format!(
                "Clef requires backend=\"candle\" and format=\"safetensors\", got \
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
                ModelError::load(format!("Clef model {:?} is not registered", manifest.name))
            })?;
        Ok(Arc::new(ClefModel::load_dir(&directory, options)?))
    }
}

/// Whether `directory` is a Clef checkpoint with a Qwen3.5 backbone and the
/// joint schema head.
pub fn is_clef_directory(directory: &Path) -> bool {
    let head = directory.join("joint_head.safetensors").is_file()
        && directory.join("joint_head_config.json").is_file();
    let tokenizer = directory.join("tokenizer.json").is_file();
    let model_type = std::fs::read(directory.join("config.json"))
        .ok()
        .and_then(|bytes| serde_json::from_slice::<serde_json::Value>(&bytes).ok())
        .and_then(|value| {
            value
                .get("model_type")
                .and_then(|kind| kind.as_str())
                .map(str::to_owned)
        });
    head && tokenizer
        && model_type.as_deref() == Some("qwen3_5")
        && std::fs::read_dir(directory).ok().is_some_and(|entries| {
            entries.filter_map(|entry| entry.ok()).any(|entry| {
                let name = entry.file_name().to_string_lossy().into_owned();
                name.starts_with("model") && name.ends_with(".safetensors")
            })
        })
}
