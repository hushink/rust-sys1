use serde::Deserialize;
use std::fs;
use std::path::{Component, Path, PathBuf};
use sys1_core::ModelError;

/// Optional sidecar that customizes the ONNX scoring contract.
pub(crate) const ONNX_SIDECAR: &str = "onnx.json";

fn default_model() -> String {
    "model.onnx".to_owned()
}

fn default_max_batch() -> usize {
    8
}

fn default_input_ids() -> String {
    "input_ids".to_owned()
}

fn default_attention_mask() -> String {
    "attention_mask".to_owned()
}

fn default_type_ids() -> String {
    "type_ids".to_owned()
}

fn default_markers() -> String {
    "markers".to_owned()
}

fn default_output() -> String {
    "logits".to_owned()
}

/// Parsed `onnx.json`.
///
/// Without the `onnx` feature only the model path is read (for manifest
/// detection); the runtime fields are consumed by `LayaOnnxModel`.
#[derive(Clone, Debug, Deserialize)]
#[cfg_attr(not(feature = "onnx"), allow(dead_code))]
pub(crate) struct OnnxFileConfig {
    #[serde(default = "default_model")]
    pub model: String,
    #[serde(default = "default_max_batch")]
    pub max_batch: usize,
    #[serde(default)]
    pub intra_threads: Option<usize>,
    #[serde(default)]
    pub inter_threads: Option<usize>,
    #[serde(default = "default_input_ids")]
    pub input_ids: String,
    #[serde(default = "default_attention_mask")]
    pub attention_mask: String,
    #[serde(default = "default_type_ids")]
    pub type_ids: String,
    #[serde(default = "default_markers")]
    pub markers: String,
    #[serde(default = "default_output")]
    pub output: String,
}

impl Default for OnnxFileConfig {
    fn default() -> Self {
        Self {
            model: default_model(),
            max_batch: default_max_batch(),
            intra_threads: None,
            inter_threads: None,
            input_ids: default_input_ids(),
            attention_mask: default_attention_mask(),
            type_ids: default_type_ids(),
            markers: default_markers(),
            output: default_output(),
        }
    }
}

impl OnnxFileConfig {
    /// Read `onnx.json`, or return defaults when it is absent.
    pub(crate) fn load(directory: &Path) -> Result<Self, ModelError> {
        let path = directory.join(ONNX_SIDECAR);
        if !path.is_file() {
            return Ok(Self::default());
        }
        let bytes = fs::read(&path)
            .map_err(|error| ModelError::load(format!("read {}: {error}", path.display())))?;
        serde_json::from_slice(&bytes)
            .map_err(|error| ModelError::load(format!("parse {}: {error}", path.display())))
    }

    /// Resolve `model` inside `directory`, rejecting paths outside it.
    pub(crate) fn model_path(&self, directory: &Path) -> Result<PathBuf, ModelError> {
        let path = Path::new(&self.model);
        if path.as_os_str().is_empty() {
            return Err(ModelError::load("onnx model path must not be empty"));
        }
        if path.is_absolute()
            || path.components().any(|component| {
                matches!(
                    component,
                    Component::ParentDir | Component::RootDir | Component::Prefix(_)
                )
            })
        {
            return Err(ModelError::load(format!(
                "onnx model path must stay inside the model directory: {}",
                self.model
            )));
        }
        Ok(directory.join(path))
    }
}
