use serde::Deserialize;
use std::fs;
use std::path::{Component, Path, PathBuf};
use sys1_core::ModelError;

/// Optional sidecar that customizes the ONNX scoring contract.
pub(crate) const ONNX_SIDECAR: &str = "onnx.json";
/// Conventional act-head weights file written by the exporter.
#[cfg_attr(not(feature = "onnx"), allow(dead_code))]
pub(crate) const DEFAULT_ACT_HEAD: &str = "act_head.safetensors";

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
    /// Optional `[batch, hidden]` pooled output used by an act head.
    #[serde(default)]
    pub pooled_output: Option<String>,
    /// Optional act-head weights sidecar. Defaults to
    /// `act_head.safetensors` when that file exists.
    #[serde(default)]
    pub act_head: Option<String>,
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
            pooled_output: None,
            act_head: None,
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
        resolve_inside(directory, &self.model, "onnx model path")
    }

    /// Resolve the optional act-head sidecar.
    ///
    /// An explicit `act_head` must exist. Without one, a conventional
    /// `act_head.safetensors` beside the model enables the head automatically.
    #[cfg_attr(not(feature = "onnx"), allow(dead_code))]
    pub(crate) fn act_head_path(&self, directory: &Path) -> Result<Option<PathBuf>, ModelError> {
        let path = match &self.act_head {
            Some(value) => Some(resolve_inside(directory, value, "onnx act_head path")?),
            None => {
                let candidate = directory.join(DEFAULT_ACT_HEAD);
                candidate.is_file().then_some(candidate)
            }
        };
        if let Some(path) = &path
            && !path.is_file()
        {
            return Err(ModelError::load(format!(
                "missing act-head weights at {}",
                path.display()
            )));
        }
        Ok(path)
    }

    /// The pooled output name to request when an act head is enabled.
    #[cfg_attr(not(feature = "onnx"), allow(dead_code))]
    pub(crate) fn pooled_output(&self, has_act_head: bool) -> Option<String> {
        self.pooled_output
            .clone()
            .or_else(|| has_act_head.then(|| "pooled".to_owned()))
    }
}

fn resolve_inside(directory: &Path, value: &str, label: &str) -> Result<PathBuf, ModelError> {
    let path = Path::new(value);
    if path.as_os_str().is_empty() {
        return Err(ModelError::load(format!("{label} must not be empty")));
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
            "{label} must stay inside the model directory: {value}"
        )));
    }
    Ok(directory.join(path))
}
