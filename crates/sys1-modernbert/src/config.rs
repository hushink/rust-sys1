use serde::Deserialize;
use std::collections::HashMap;
use std::fs;
use std::path::Path;

fn default_layer_norm_eps() -> f64 {
    1e-5
}

fn default_global_every() -> usize {
    3
}

fn default_local_attention() -> usize {
    128
}

#[derive(Clone, Debug, Deserialize)]
pub struct RopeConfig {
    pub rope_theta: f64,
}

/// The parts of a Hugging Face ModernBERT `config.json` needed by the
/// encoder. Both the nested Laya `rope_parameters` form and the flat
/// `global_rope_theta` / `local_rope_theta` form are accepted.
#[derive(Clone, Debug, Deserialize)]
pub struct ModernBertConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_attention_heads: usize,
    pub num_hidden_layers: usize,
    pub intermediate_size: usize,
    pub max_position_embeddings: usize,
    #[serde(default = "default_layer_norm_eps")]
    pub layer_norm_eps: f64,
    /// Some checkpoints spell the same value `norm_eps`.
    #[serde(default)]
    pub norm_eps: Option<f64>,
    #[serde(default = "default_global_every")]
    pub global_attn_every_n_layers: usize,
    #[serde(default = "default_local_attention")]
    pub local_attention: usize,
    #[serde(default)]
    pub(crate) rope_parameters: HashMap<String, RopeConfig>,
    #[serde(default)]
    pub global_rope_theta: Option<f64>,
    #[serde(default)]
    pub local_rope_theta: Option<f64>,
    #[serde(default)]
    pub bos_token_id: Option<u32>,
    #[serde(default)]
    pub eos_token_id: Option<u32>,
    #[serde(default)]
    pub cls_token_id: Option<u32>,
    #[serde(default)]
    pub sep_token_id: Option<u32>,
    #[serde(default)]
    pub pad_token_id: Option<u32>,
}

impl ModernBertConfig {
    pub fn load(path: &Path) -> Result<Self, String> {
        let bytes = fs::read(path)
            .map_err(|error| format!("failed to read {}: {error}", path.display()))?;
        serde_json::from_slice(&bytes)
            .map_err(|error| format!("failed to parse {}: {error}", path.display()))
    }

    pub fn from_slice(bytes: &[u8]) -> Result<Self, String> {
        serde_json::from_slice(bytes).map_err(|error| error.to_string())
    }

    pub fn hidden_size(&self) -> usize {
        self.hidden_size
    }

    pub fn max_position_embeddings(&self) -> usize {
        self.max_position_embeddings
    }

    pub fn layer_norm_eps(&self) -> f64 {
        self.norm_eps.unwrap_or(self.layer_norm_eps)
    }

    pub fn global_rope_theta(&self) -> f64 {
        self.rope_parameters
            .get("full_attention")
            .map(|rope| rope.rope_theta)
            .or(self.global_rope_theta)
            .unwrap_or(10_000.0)
    }

    pub fn local_rope_theta(&self) -> f64 {
        self.rope_parameters
            .get("sliding_attention")
            .map(|rope| rope.rope_theta)
            .or(self.local_rope_theta)
            .unwrap_or_else(|| self.global_rope_theta())
    }
}
