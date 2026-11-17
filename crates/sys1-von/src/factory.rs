use crate::{VonModel, VonOptions};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, RwLock};
use sys1_core::{DecisionModel, ModelError, ModelFactory, ModelManifest};

struct Registered {
    directory: PathBuf,
    options: VonOptions,
}

/// Registers Von checkpoints by catalog name and loads them lazily.
#[derive(Default)]
pub struct VonFactory {
    models: RwLock<HashMap<String, Registered>>,
}

impl VonFactory {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn register(
        &self,
        name: impl Into<String>,
        directory: impl Into<PathBuf>,
        options: VonOptions,
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

impl ModelFactory for VonFactory {
    fn load(&self, manifest: &ModelManifest) -> Result<Arc<dyn DecisionModel>, ModelError> {
        if manifest.backend != "candle" && manifest.format != "safetensors" {
            return Err(ModelError::unsupported(format!(
                "Von requires backend=\"candle\" and format=\"safetensors\", got \
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
                ModelError::load(format!("Von model {:?} is not registered", manifest.name))
            })?;
        Ok(Arc::new(VonModel::load_dir(&directory, options)?))
    }
}

/// Whether `directory` is a Von checkpoint: ModernBERT config plus the
/// `option_marker.pt` weights.
pub fn is_von_directory(directory: &Path) -> bool {
    let weights = directory.join("option_marker.pt").is_file();
    let tokenizer = directory.join("tokenizer.json").is_file()
        || directory.join("tokenizer").join("tokenizer.json").is_file();
    if !weights || !tokenizer {
        return false;
    }
    std::fs::read(directory.join("config.json"))
        .ok()
        .and_then(|bytes| serde_json::from_slice::<serde_json::Value>(&bytes).ok())
        .and_then(|value| {
            value
                .get("model_type")
                .and_then(|kind| kind.as_str())
                .map(str::to_owned)
        })
        .as_deref()
        == Some("modernbert")
}
