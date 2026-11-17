use crate::calibration::bucket;
use serde::Deserialize;
use std::collections::HashMap;
use std::fs;
use std::path::Path;

#[derive(Clone, Debug, Deserialize)]
pub(crate) struct LayaConfig {
    pub max_len: usize,
    pub head_max_len: usize,
    pub temperature: Vec<f32>,
    #[serde(default)]
    pub temperature_by_options: HashMap<String, f32>,
    #[serde(default = "default_head_layers")]
    pub head_layers: usize,
}

fn default_head_layers() -> usize {
    2
}

impl LayaConfig {
    pub fn load(path: &Path) -> Result<Self, String> {
        let bytes = fs::read(path)
            .map_err(|error| format!("failed to read {}: {error}", path.display()))?;
        let config: Self = serde_json::from_slice(&bytes)
            .map_err(|error| format!("failed to parse {}: {error}", path.display()))?;
        if config.temperature.len() != 3 {
            return Err(format!(
                "{} must contain exactly three temperature values",
                path.display()
            ));
        }
        if config.head_layers == 0 {
            return Err("head_layers must be greater than zero".to_owned());
        }
        Ok(config)
    }

    pub fn temperature_for(&self, kind: usize, count: usize) -> f32 {
        let bucket = bucket(kind, count);
        let temperature = self
            .temperature_by_options
            .get(&bucket)
            .copied()
            .unwrap_or_else(|| self.temperature.get(kind).copied().unwrap_or(1.0));
        if temperature.is_finite() {
            temperature.clamp(0.5, 5.0)
        } else {
            1.0
        }
    }
}

#[derive(Clone, Debug, Deserialize)]
pub(crate) struct ModernBertConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub intermediate_size: usize,
    pub max_position_embeddings: usize,
    pub layer_norm_eps: f64,
    pub global_attn_every_n_layers: usize,
    pub local_attention: usize,
    pub rope_parameters: HashMap<String, RopeConfig>,
}

#[derive(Clone, Debug, Deserialize)]
pub(crate) struct RopeConfig {
    pub rope_theta: f64,
}

impl ModernBertConfig {
    pub fn load(path: &Path) -> Result<Self, String> {
        let bytes = fs::read(path)
            .map_err(|error| format!("failed to read {}: {error}", path.display()))?;
        serde_json::from_slice(&bytes)
            .map_err(|error| format!("failed to parse {}: {error}", path.display()))
    }

    pub fn hidden_size(&self) -> usize {
        self.hidden_size
    }

    pub fn max_position_embeddings(&self) -> usize {
        self.max_position_embeddings
    }

    pub fn global_rope_theta(&self) -> f64 {
        self.rope_parameters
            .get("full_attention")
            .map(|rope| rope.rope_theta)
            .unwrap_or(10_000.0)
    }

    pub fn local_rope_theta(&self) -> f64 {
        self.rope_parameters
            .get("sliding_attention")
            .map(|rope| rope.rope_theta)
            .unwrap_or_else(|| self.global_rope_theta())
    }
}
