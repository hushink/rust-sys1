use serde::Deserialize;
use std::fs;
use std::path::Path;

fn default_prefix() -> String {
    "model.".to_owned()
}

fn default_qwen35_prefix() -> String {
    "model.language_model.".to_owned()
}

/// Hugging Face `Qwen3Config`, as Qwen3Guard and other dense Qwen3 checkpoints
/// store it.
#[derive(Clone, Debug, Deserialize)]
pub struct Qwen3Config {
    #[serde(skip, default = "default_prefix")]
    pub prefix: String,
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    #[serde(default = "default_rms_eps")]
    pub rms_norm_eps: f64,
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f64,
    #[serde(default)]
    pub attention_bias: bool,
    #[serde(default = "default_hidden_act")]
    pub hidden_act: String,
    #[serde(default)]
    pub tie_word_embeddings: bool,
    #[serde(default)]
    pub use_sliding_window: bool,
    #[serde(default)]
    pub max_position_embeddings: usize,
}

/// Hugging Face `Qwen3_5TextConfig`, either flat or nested under
/// `text_config` in a `Qwen3_5ForConditionalGeneration` config.
#[derive(Clone, Debug, Deserialize)]
pub struct Qwen35Config {
    #[serde(skip, default = "default_qwen35_prefix")]
    pub prefix: String,
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub layer_types: Vec<String>,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    #[serde(default = "default_partial_rotary_factor")]
    pub partial_rotary_factor: f64,
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f64,
    #[serde(default)]
    pub attention_bias: bool,
    #[serde(default = "default_true")]
    pub attn_output_gate: bool,
    pub linear_num_key_heads: usize,
    pub linear_num_value_heads: usize,
    pub linear_key_head_dim: usize,
    pub linear_value_head_dim: usize,
    pub linear_conv_kernel_dim: usize,
    #[serde(default = "default_rms_eps")]
    pub rms_norm_eps: f64,
    #[serde(default = "default_hidden_act")]
    pub hidden_act: String,
    #[serde(default)]
    pub tie_word_embeddings: bool,
    #[serde(default)]
    pub mlp_only_layers: Vec<usize>,
    #[serde(default)]
    pub rope_parameters: Option<RopeParameters>,
}

#[derive(Clone, Debug, Deserialize)]
pub struct RopeParameters {
    pub rope_theta: f64,
    #[serde(default)]
    pub partial_rotary_factor: Option<f64>,
}

fn default_rms_eps() -> f64 {
    1e-6
}

fn default_rope_theta() -> f64 {
    10_000.0
}

fn default_hidden_act() -> String {
    "silu".to_owned()
}

fn default_true() -> bool {
    true
}

fn default_partial_rotary_factor() -> f64 {
    1.0
}

impl Qwen3Config {
    pub fn load(path: &Path) -> Result<Self, String> {
        let bytes = fs::read(path).map_err(|error| format!("read {}: {error}", path.display()))?;
        serde_json::from_slice(&bytes).map_err(|error| format!("parse {}: {error}", path.display()))
    }

    pub fn validate(&self) -> Result<(), String> {
        if self.hidden_act != "silu" {
            return Err(format!("unsupported hidden_act {:?}", self.hidden_act));
        }
        if self.num_attention_heads == 0
            || self.num_key_value_heads == 0
            || !self
                .num_attention_heads
                .is_multiple_of(self.num_key_value_heads)
            || self.head_dim == 0
            || !self.head_dim.is_multiple_of(2)
        {
            return Err("inconsistent Qwen3 attention dimensions".to_owned());
        }
        if self.num_key_value_heads * self.head_dim > self.hidden_size {
            return Err("Qwen3 key/value width exceeds hidden_size".to_owned());
        }
        if self.use_sliding_window {
            return Err("Qwen3 sliding windows are not supported".to_owned());
        }
        Ok(())
    }

    pub fn rotary_dim(&self) -> usize {
        self.head_dim
    }
}

impl Qwen35Config {
    pub fn load(path: &Path) -> Result<Self, String> {
        let bytes = fs::read(path).map_err(|error| format!("read {}: {error}", path.display()))?;
        let root: serde_json::Value = serde_json::from_slice(&bytes)
            .map_err(|error| format!("parse {}: {error}", path.display()))?;
        let text = root.get("text_config").unwrap_or(&root);
        let mut config: Self = serde_json::from_value(text.clone())
            .map_err(|error| format!("parse text_config: {error}"))?;
        if let Some(rope) = &config.rope_parameters {
            config.rope_theta = rope.rope_theta;
            if let Some(partial) = rope.partial_rotary_factor {
                config.partial_rotary_factor = partial;
            }
        }
        Ok(config)
    }

    pub fn validate(&self) -> Result<(), String> {
        if self.hidden_act != "silu" {
            return Err(format!("unsupported hidden_act {:?}", self.hidden_act));
        }
        if !self.attn_output_gate {
            return Err("Qwen3.5 attention output gates are required".to_owned());
        }
        if !self.mlp_only_layers.is_empty() {
            return Err("Qwen3.5 mlp_only_layers are not supported".to_owned());
        }
        if self.layer_types.len() != self.num_hidden_layers {
            return Err(format!(
                "layer_types has {} entries for {} layers",
                self.layer_types.len(),
                self.num_hidden_layers
            ));
        }
        if self.num_attention_heads == 0
            || self.num_key_value_heads == 0
            || !self
                .num_attention_heads
                .is_multiple_of(self.num_key_value_heads)
            || self.head_dim == 0
        {
            return Err("inconsistent Qwen3.5 attention dimensions".to_owned());
        }
        if self.linear_num_key_heads == 0
            || self.linear_num_value_heads == 0
            || !self
                .linear_num_value_heads
                .is_multiple_of(self.linear_num_key_heads)
            || self.linear_key_head_dim == 0
            || self.linear_value_head_dim == 0
            || self.linear_conv_kernel_dim == 0
        {
            return Err("inconsistent Qwen3.5 Gated DeltaNet dimensions".to_owned());
        }
        let rotary = self.rotary_dim();
        if rotary == 0 || !rotary.is_multiple_of(2) || rotary > self.head_dim {
            return Err(format!(
                "unsupported partial_rotary_factor {}",
                self.partial_rotary_factor
            ));
        }
        if self
            .layer_types
            .iter()
            .any(|kind| kind != "linear_attention" && kind != "full_attention")
        {
            return Err("layer_types must be linear_attention or full_attention".to_owned());
        }
        Ok(())
    }

    pub fn rotary_dim(&self) -> usize {
        (self.head_dim as f64 * self.partial_rotary_factor) as usize
    }

    pub fn has_lm_head(&self) -> bool {
        !self.tie_word_embeddings
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn reads_nested_qwen35_config() {
        let value = json!({
            "model_type": "qwen3_5",
            "text_config": {
                "vocab_size": 10, "hidden_size": 8, "intermediate_size": 12,
                "num_hidden_layers": 2, "layer_types": ["linear_attention", "full_attention"],
                "num_attention_heads": 2, "num_key_value_heads": 1, "head_dim": 4,
                "partial_rotary_factor": 0.5, "rope_theta": 1e7, "attention_bias": false,
                "attn_output_gate": true, "linear_num_key_heads": 2,
                "linear_num_value_heads": 2, "linear_key_head_dim": 4,
                "linear_value_head_dim": 4, "linear_conv_kernel_dim": 4,
                "rms_norm_eps": 1e-6, "hidden_act": "silu", "tie_word_embeddings": true
            }
        });
        let config: Qwen35Config = serde_json::from_value(value["text_config"].clone()).unwrap();
        config.validate().unwrap();
        assert_eq!(config.rotary_dim(), 2);
    }
}
