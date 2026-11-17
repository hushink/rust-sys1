use crate::{GliclassModel, GliclassOptions};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, RwLock};
use sys1_core::{DecisionModel, ModelError, ModelFactory, ModelManifest};

struct Registered {
    directory: PathBuf,
    options: GliclassOptions,
}

/// Registers GLiClass checkpoints by catalog name and loads them lazily.
#[derive(Default)]
pub struct GliclassFactory {
    models: RwLock<HashMap<String, Registered>>,
}

impl GliclassFactory {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn register(
        &self,
        name: impl Into<String>,
        directory: impl Into<PathBuf>,
        options: GliclassOptions,
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

impl ModelFactory for GliclassFactory {
    fn load(&self, manifest: &ModelManifest) -> Result<Arc<dyn DecisionModel>, ModelError> {
        if manifest.backend != "candle" && manifest.format != "safetensors" {
            return Err(ModelError::unsupported(format!(
                "GLiClass requires backend=\"candle\" and format=\"safetensors\", got backend={:?} format={:?}",
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
                    "GLiClass model {:?} is not registered",
                    manifest.name
                ))
            })?;
        Ok(Arc::new(GliclassModel::load_dir(&directory, options)?))
    }
}

/// Whether `directory` is a ModernBERT-backed GLiClass uni-encoder checkpoint.
pub fn is_gliclass_directory(directory: &Path) -> bool {
    let config_path = directory.join("config.json");
    let weights = directory.join("model.safetensors");
    let tokenizer = directory.join("tokenizer.json").is_file()
        || directory.join("tokenizer").join("tokenizer.json").is_file();
    if !config_path.is_file() || !weights.is_file() || !tokenizer {
        return false;
    }
    let Ok(bytes) = std::fs::read(&config_path) else {
        return false;
    };
    let Ok(value) = serde_json::from_slice::<serde_json::Value>(&bytes) else {
        return false;
    };
    let is_gliclass = value.get("model_type").and_then(serde_json::Value::as_str)
        == Some("GLiClass")
        && value
            .get("architecture_type")
            .and_then(serde_json::Value::as_str)
            == Some("uni-encoder");
    let modernbert = value
        .get("encoder_config")
        .and_then(|config| config.get("model_type"))
        .and_then(serde_json::Value::as_str)
        == Some("modernbert");
    is_gliclass && modernbert
}
