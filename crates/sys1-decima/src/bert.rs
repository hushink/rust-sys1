//! A post-layer-norm BERT encoder for Decima-small.
//!
//! `intfloat/multilingual-e5-small` is a plain `BertModel`: learned absolute
//! positions, token-type embeddings, `gelu` feed-forward blocks, and
//! layer normalization after each residual branch. The late-interaction scorer
//! only needs `last_hidden_state`, so no pooler is loaded.

use candle_core::{D, DType, Tensor};
use candle_nn::{Activation, Embedding, LayerNorm, Linear, Module, embedding, layer_norm, linear};
use serde::Deserialize;
use sys1_core::ModelError;

const ATTENTION_SCALE: f64 = 1e9;

/// The parts of a Hugging Face BERT `config.json` needed by the encoder.
#[derive(Clone, Debug, Deserialize)]
pub(crate) struct BertConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_attention_heads: usize,
    pub num_hidden_layers: usize,
    pub intermediate_size: usize,
    pub max_position_embeddings: usize,
    #[serde(default = "default_layer_norm_eps")]
    pub layer_norm_eps: f64,
    #[serde(default = "default_type_vocab_size")]
    pub type_vocab_size: usize,
    #[serde(default)]
    pub hidden_act: Option<String>,
    #[serde(default)]
    pub cls_token_id: Option<u32>,
    #[serde(default)]
    pub sep_token_id: Option<u32>,
}

fn default_layer_norm_eps() -> f64 {
    1e-12
}

fn default_type_vocab_size() -> usize {
    2
}

impl BertConfig {
    pub fn from_slice(bytes: &[u8]) -> Result<Self, String> {
        serde_json::from_slice(bytes).map_err(|error| error.to_string())
    }

    fn validate(&self) -> Result<(), ModelError> {
        if self.hidden_size == 0
            || self.num_attention_heads == 0
            || !self.hidden_size.is_multiple_of(self.num_attention_heads)
        {
            return Err(ModelError::load(format!(
                "BERT config has invalid hidden size {} and {} heads",
                self.hidden_size, self.num_attention_heads
            )));
        }
        if self.vocab_size == 0 || self.num_hidden_layers == 0 || self.max_position_embeddings == 0
        {
            return Err(ModelError::load("BERT config has empty dimensions"));
        }
        match self.hidden_act.as_deref().unwrap_or("gelu") {
            "gelu" | "gelu_new" | "gelu_pytorch_tanh" => Ok(()),
            other => Err(ModelError::unsupported(format!(
                "BERT activation {other:?}; only gelu variants are supported"
            ))),
        }
    }

    fn activation(&self) -> Activation {
        match self.hidden_act.as_deref().unwrap_or("gelu") {
            "gelu_new" | "gelu_pytorch_tanh" => Activation::GeluPytorchTanh,
            _ => Activation::Gelu,
        }
    }
}

struct BertAttention {
    query: Linear,
    key: Linear,
    value: Linear,
    output: Linear,
    heads: usize,
    head_dim: usize,
}

impl BertAttention {
    fn load(vb: candle_nn::VarBuilder, hidden: usize, heads: usize) -> Result<Self, ModelError> {
        let self_vb = vb.pp("attention.self");
        let load = |name: &str| -> Result<Linear, ModelError> {
            linear(hidden, hidden, self_vb.pp(name))
                .map_err(|error| ModelError::load(error.to_string()))
        };
        Ok(Self {
            query: load("query")?,
            key: load("key")?,
            value: load("value")?,
            output: linear(hidden, hidden, vb.pp("attention.output.dense"))
                .map_err(|error| ModelError::load(error.to_string()))?,
            heads,
            head_dim: hidden / heads,
        })
    }

    fn forward(&self, xs: &Tensor, bias: &Tensor) -> Result<Tensor, ModelError> {
        let (batch, length, hidden) = xs.dims3().map_err(inference_error)?;
        let q = xs
            .apply(&self.query)
            .map_err(inference_error)?
            .reshape((batch, length, self.heads, self.head_dim))
            .map_err(inference_error)?
            .transpose(1, 2)
            .map_err(inference_error)?;
        let k = xs
            .apply(&self.key)
            .map_err(inference_error)?
            .reshape((batch, length, self.heads, self.head_dim))
            .map_err(inference_error)?
            .transpose(1, 2)
            .map_err(inference_error)?;
        let v = xs
            .apply(&self.value)
            .map_err(inference_error)?
            .reshape((batch, length, self.heads, self.head_dim))
            .map_err(inference_error)?
            .transpose(1, 2)
            .map_err(inference_error)?;

        let scores = q
            .contiguous()
            .map_err(inference_error)?
            .matmul(
                &k.transpose(2, 3)
                    .map_err(inference_error)?
                    .contiguous()
                    .map_err(inference_error)?,
            )
            .map_err(inference_error)?
            .affine((self.head_dim as f64).powf(-0.5), 0.0)
            .map_err(inference_error)?
            .broadcast_add(bias)
            .map_err(inference_error)?;
        let dtype = scores.dtype();
        let probabilities = if matches!(dtype, DType::F16 | DType::BF16) {
            candle_nn::ops::softmax(
                &scores.to_dtype(DType::F32).map_err(inference_error)?,
                D::Minus1,
            )
            .map_err(inference_error)?
            .to_dtype(dtype)
            .map_err(inference_error)?
        } else {
            candle_nn::ops::softmax(&scores, D::Minus1).map_err(inference_error)?
        };
        probabilities
            .matmul(&v.contiguous().map_err(inference_error)?)
            .map_err(inference_error)?
            .transpose(1, 2)
            .map_err(inference_error)?
            .reshape((batch, length, hidden))
            .map_err(inference_error)?
            .apply(&self.output)
            .map_err(inference_error)
    }
}

struct BertLayer {
    attention: BertAttention,
    norm1: LayerNorm,
    intermediate: Linear,
    output: Linear,
    norm2: LayerNorm,
    activation: Activation,
}

impl BertLayer {
    fn load(
        vb: candle_nn::VarBuilder,
        hidden: usize,
        intermediate: usize,
        heads: usize,
        eps: f64,
        activation: Activation,
    ) -> Result<Self, ModelError> {
        let linear_load = |name: &str, out: usize| -> Result<Linear, ModelError> {
            linear(hidden, out, vb.pp(name)).map_err(|error| ModelError::load(error.to_string()))
        };
        let norm = |name: &str| -> Result<LayerNorm, ModelError> {
            layer_norm(hidden, eps, vb.pp(name))
                .map_err(|error| ModelError::load(error.to_string()))
        };
        Ok(Self {
            attention: BertAttention::load(vb.clone(), hidden, heads)?,
            norm1: norm("attention.output.LayerNorm")?,
            intermediate: linear_load("intermediate.dense", intermediate)?,
            output: linear(intermediate, hidden, vb.pp("output.dense"))
                .map_err(|error| ModelError::load(error.to_string()))?,
            norm2: norm("output.LayerNorm")?,
            activation,
        })
    }

    fn forward(&self, xs: &Tensor, bias: &Tensor) -> Result<Tensor, ModelError> {
        let attention = self.attention.forward(xs, bias)?;
        let xs = xs
            .add(&attention)
            .map_err(inference_error)?
            .apply(&self.norm1)
            .map_err(inference_error)?;
        let feed_forward = xs
            .apply(&self.intermediate)
            .map_err(inference_error)?
            .apply(&self.activation)
            .map_err(inference_error)?
            .apply(&self.output)
            .map_err(inference_error)?;
        xs.add(&feed_forward)
            .map_err(inference_error)?
            .apply(&self.norm2)
            .map_err(inference_error)
    }
}

/// A BERT encoder producing `[batch, length, hidden]` last hidden states.
pub(crate) struct BertEncoder {
    word_embeddings: Embedding,
    position_embeddings: Embedding,
    token_type_embeddings: Embedding,
    norm: LayerNorm,
    layers: Vec<BertLayer>,
}

impl BertEncoder {
    pub fn load(vb: candle_nn::VarBuilder, config: &BertConfig) -> Result<Self, ModelError> {
        config.validate()?;
        let prefix = ["", "bert.", "model."]
            .into_iter()
            .find(|prefix| {
                vb.contains_tensor(&format!("{prefix}embeddings.word_embeddings.weight"))
            })
            .ok_or_else(|| ModelError::load("BERT encoder has no word embeddings"))?;
        let root = if prefix.is_empty() {
            vb
        } else {
            vb.pp(prefix.trim_end_matches('.'))
        };
        let hidden = config.hidden_size;
        let eps = config.layer_norm_eps;
        let word_embeddings = embedding(
            config.vocab_size,
            hidden,
            root.pp("embeddings.word_embeddings"),
        )
        .map_err(|error| ModelError::load(error.to_string()))?;
        let position_embeddings = embedding(
            config.max_position_embeddings,
            hidden,
            root.pp("embeddings.position_embeddings"),
        )
        .map_err(|error| ModelError::load(error.to_string()))?;
        let token_type_embeddings = embedding(
            config.type_vocab_size,
            hidden,
            root.pp("embeddings.token_type_embeddings"),
        )
        .map_err(|error| ModelError::load(error.to_string()))?;
        let norm = layer_norm(hidden, eps, root.pp("embeddings.LayerNorm"))
            .map_err(|error| ModelError::load(error.to_string()))?;
        let activation = config.activation();
        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for index in 0..config.num_hidden_layers {
            layers.push(BertLayer::load(
                root.pp(format!("encoder.layer.{index}")),
                hidden,
                config.intermediate_size,
                config.num_attention_heads,
                eps,
                activation,
            )?);
        }
        Ok(Self {
            word_embeddings,
            position_embeddings,
            token_type_embeddings,
            norm,
            layers,
        })
    }

    pub(crate) fn forward(&self, ids: &Tensor, mask: &Tensor) -> Result<Tensor, ModelError> {
        let (batch, length) = ids.dims2().map_err(inference_error)?;
        let device = ids.device();
        let positions = Tensor::arange(0u32, length as u32, device)
            .map_err(inference_error)?
            .unsqueeze(0)
            .map_err(inference_error)?
            .broadcast_as((batch, length))
            .map_err(inference_error)?;
        let token_types =
            Tensor::zeros((batch, length), DType::U32, device).map_err(inference_error)?;
        let mut xs = self
            .word_embeddings
            .forward(ids)
            .map_err(inference_error)?
            .add(
                &self
                    .position_embeddings
                    .forward(&positions)
                    .map_err(inference_error)?,
            )
            .map_err(inference_error)?
            .add(
                &self
                    .token_type_embeddings
                    .forward(&token_types)
                    .map_err(inference_error)?,
            )
            .map_err(inference_error)?
            .apply(&self.norm)
            .map_err(inference_error)?;
        let bias = attention_bias(mask, xs.dtype())?;
        for layer in &self.layers {
            xs = layer.forward(&xs, &bias)?;
        }
        Ok(xs)
    }
}

fn attention_bias(mask: &Tensor, dtype: DType) -> Result<Tensor, ModelError> {
    mask.to_dtype(DType::F32)
        .map_err(inference_error)?
        .affine(-1.0, 1.0)
        .map_err(inference_error)?
        .affine(-ATTENTION_SCALE, 0.0)
        .map_err(inference_error)?
        .unsqueeze(1)
        .map_err(inference_error)?
        .unsqueeze(2)
        .map_err(inference_error)?
        .to_dtype(dtype)
        .map_err(inference_error)
}

fn inference_error(error: candle_core::Error) -> ModelError {
    ModelError::inference(error.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_multilingual_e5_small_config() {
        let bytes = br#"{
            "architectures": ["BertModel"],
            "hidden_act": "gelu",
            "hidden_size": 384,
            "intermediate_size": 1536,
            "layer_norm_eps": 1e-12,
            "max_position_embeddings": 512,
            "model_type": "bert",
            "num_attention_heads": 12,
            "num_hidden_layers": 12,
            "type_vocab_size": 2,
            "vocab_size": 250037
        }"#;
        let config = BertConfig::from_slice(bytes).expect("config");
        config.validate().expect("valid");
        assert_eq!(config.hidden_size, 384);
        assert_eq!(config.num_attention_heads, 12);
        assert!(matches!(config.activation(), Activation::Gelu));
    }

    #[test]
    fn rejects_non_gelu_activations() {
        let config = BertConfig {
            vocab_size: 8,
            hidden_size: 8,
            num_attention_heads: 2,
            num_hidden_layers: 1,
            intermediate_size: 16,
            max_position_embeddings: 8,
            layer_norm_eps: 1e-12,
            type_vocab_size: 2,
            hidden_act: Some("relu".to_owned()),
            cls_token_id: None,
            sep_token_id: None,
        };
        assert!(config.validate().is_err());
    }
}
