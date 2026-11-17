use crate::{Qwen3GuardModel, Qwen3GuardOptions};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, RwLock};
use sys1_core::{DecisionModel, ModelError, ModelFactory, ModelManifest};

struct Registered {
    directory: PathBuf,
    options: Qwen3GuardOptions,
}

/// Registers Qwen3Guard checkpoints by catalog name and loads them lazily.
#[derive(Default)]
pub struct Qwen3GuardFactory {
    models: RwLock<HashMap<String, Registered>>,
}

impl Qwen3GuardFactory {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn register(
        &self,
        name: impl Into<String>,
        directory: impl Into<PathBuf>,
        options: Qwen3GuardOptions,
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

impl ModelFactory for Qwen3GuardFactory {
    fn load(&self, manifest: &ModelManifest) -> Result<Arc<dyn DecisionModel>, ModelError> {
        if manifest.backend != "candle" && manifest.format != "safetensors" {
            return Err(ModelError::unsupported(format!(
                "Qwen3Guard requires backend=\"candle\" and format=\"safetensors\", got \
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
                    "Qwen3Guard model {:?} is not registered",
                    manifest.name
                ))
            })?;
        Ok(Arc::new(Qwen3GuardModel::load_dir(&directory, options)?))
    }
}

/// Whether `directory` is a Qwen3Guard Gen checkpoint: a Qwen3 config and a
/// tokenizer whose chat template carries the guard policy.
pub fn is_qwen3guard_directory(directory: &Path) -> bool {
    let config_path = directory.join("config.json");
    let tokenizer_path = [
        directory.join("tokenizer_config.json"),
        directory.join("tokenizer").join("tokenizer_config.json"),
    ]
    .into_iter()
    .find(|path| path.is_file());
    let weights = directory.join("model.safetensors").is_file()
        || std::fs::read_dir(directory).ok().is_some_and(|entries| {
            entries.filter_map(|entry| entry.ok()).any(|entry| {
                entry
                    .file_name()
                    .to_string_lossy()
                    .ends_with(".safetensors")
            })
        });
    let tokenizer = directory.join("tokenizer.json").is_file()
        || directory.join("tokenizer").join("tokenizer.json").is_file();
    if !config_path.is_file() || !weights || !tokenizer {
        return false;
    }
    let model_type = std::fs::read(&config_path)
        .ok()
        .and_then(|bytes| serde_json::from_slice::<serde_json::Value>(&bytes).ok())
        .and_then(|value| {
            value
                .get("model_type")
                .and_then(serde_json::Value::as_str)
                .map(str::to_owned)
        });
    if model_type.as_deref() != Some("qwen3") {
        return false;
    }
    tokenizer_path
        .and_then(|path| std::fs::read_to_string(path).ok())
        .is_some_and(|text| text.contains("Safety: Safe") && text.contains("Categories:"))
}
