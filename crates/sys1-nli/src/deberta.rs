//! DeBERTa-v3 disentangled encoder for the zeroshot-v2.0 NLI checkpoints.
//!
//! The model uses relative position buckets instead of absolute position
//! embeddings, and every attention layer adds content-to-position and
//! position-to-content scores on top of content-to-content attention. The
//! relative-position table is shared by all layers and layer-normalized once
//! (`norm_rel_ebd = "layer_norm"`).

use candle_core::{D, DType, Device, Tensor};
use candle_nn::{Activation, Embedding, LayerNorm, Linear, Module, embedding, layer_norm, linear};
use serde::Deserialize;
use sys1_core::ModelError;

fn default_layer_norm_eps() -> f64 {
    1e-7
}

fn default_max_position_embeddings() -> usize {
    512
}

fn default_position_buckets() -> i64 {
    256
}

fn default_max_relative_positions() -> i64 {
    -1
}

fn default_true() -> bool {
    true
}

/// The DeBERTa-v3 fields needed by the encoder and the classification head.
#[derive(Clone, Debug, Deserialize)]
pub struct DebertaConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    #[serde(default = "default_max_position_embeddings")]
    pub max_position_embeddings: usize,
    #[serde(default = "default_layer_norm_eps")]
    pub layer_norm_eps: f64,
    #[serde(default = "default_position_buckets")]
    pub position_buckets: i64,
    #[serde(default = "default_max_relative_positions")]
    pub max_relative_positions: i64,
    #[serde(default)]
    pub pos_att_type: Vec<String>,
    #[serde(default)]
    pub share_att_key: bool,
    #[serde(default = "default_true")]
    pub relative_attention: bool,
    #[serde(default)]
    pub position_biased_input: bool,
    #[serde(default)]
    pub type_vocab_size: usize,
    #[serde(default)]
    pub norm_rel_ebd: String,
    #[serde(default)]
    pub hidden_act: Option<String>,
    #[serde(default)]
    pub pooler_hidden_act: Option<String>,
    #[serde(default)]
    pub pad_token_id: Option<u32>,
}

impl DebertaConfig {
    pub fn validate(&self) -> Result<(), ModelError> {
        if self.hidden_size == 0
            || self.num_attention_heads == 0
            || !self.hidden_size.is_multiple_of(self.num_attention_heads)
        {
            return Err(ModelError::load(
                "DeBERTa has invalid hidden size or attention heads",
            ));
        }
        if !self.relative_attention {
            return Err(ModelError::unsupported(
                "DeBERTa checkpoints without relative attention",
            ));
        }
        if self.position_biased_input {
            return Err(ModelError::unsupported(
                "DeBERTa checkpoints with position-biased input embeddings",
            ));
        }
        if self.type_vocab_size != 0 {
            return Err(ModelError::unsupported(
                "DeBERTa checkpoints with token-type embeddings",
            ));
        }
        if !self.share_att_key {
            return Err(ModelError::unsupported(
                "DeBERTa checkpoints without share_att_key",
            ));
        }
        if self.position_buckets <= 0 || self.max_position_embeddings < 8 {
            return Err(ModelError::load("DeBERTa position buckets are invalid"));
        }
        for kind in ["c2p", "p2c"] {
            if !self.pos_att_type.iter().any(|item| item == kind) {
                return Err(ModelError::unsupported(format!(
                    "DeBERTa without {kind} disentangled attention"
                )));
            }
        }
        if !self.norm_rel_ebd.is_empty()
            && !self
                .norm_rel_ebd
                .split('|')
                .any(|part| part.trim() == "layer_norm")
        {
            return Err(ModelError::unsupported(format!(
                "DeBERTa norm_rel_ebd {:?}",
                self.norm_rel_ebd
            )));
        }
        match self.hidden_act.as_deref().unwrap_or("gelu") {
            "gelu" | "gelu_new" | "gelu_pytorch_tanh" => {}
            other => {
                return Err(ModelError::unsupported(format!(
                    "DeBERTa activation {other:?}"
                )));
            }
        }
        Ok(())
    }

    pub fn hidden_size(&self) -> usize {
        self.hidden_size
    }

    pub fn max_position_embeddings(&self) -> usize {
        self.max_position_embeddings
    }

    pub fn layer_norm_eps(&self) -> f64 {
        self.layer_norm_eps
    }

    pub fn max_relative_positions(&self) -> usize {
        if self.max_relative_positions < 1 {
            self.max_position_embeddings
        } else {
            self.max_relative_positions as usize
        }
    }

    pub fn activation(&self) -> Activation {
        match self.hidden_act.as_deref().unwrap_or("gelu") {
            "gelu_new" | "gelu_pytorch_tanh" => Activation::GeluPytorchTanh,
            _ => Activation::Gelu,
        }
    }

    pub fn pooler_activation(&self) -> Activation {
        match self.pooler_hidden_act.as_deref().unwrap_or("gelu") {
            "gelu_new" | "gelu_pytorch_tanh" => Activation::GeluPytorchTanh,
            _ => Activation::Gelu,
        }
    }
}

struct DebertaLayer {
    query: Linear,
    key: Linear,
    value: Linear,
    output: Linear,
    output_norm: LayerNorm,
    intermediate: Linear,
    ff_output: Linear,
    ff_norm: LayerNorm,
    heads: usize,
    head_dim: usize,
    activation: Activation,
}

impl DebertaLayer {
    fn load(vb: candle_nn::VarBuilder, config: &DebertaConfig) -> Result<Self, ModelError> {
        let hidden = config.hidden_size;
        let load = |name: &str, out: usize, input: usize| -> Result<Linear, ModelError> {
            linear(input, out, vb.pp(name)).map_err(|error| ModelError::load(error.to_string()))
        };
        let norm = |name: &str| -> Result<LayerNorm, ModelError> {
            layer_norm(hidden, config.layer_norm_eps, vb.pp(name))
                .map_err(|error| ModelError::load(error.to_string()))
        };
        Ok(Self {
            query: load("attention.self.query_proj", hidden, hidden)?,
            key: load("attention.self.key_proj", hidden, hidden)?,
            value: load("attention.self.value_proj", hidden, hidden)?,
            output: load("attention.output.dense", hidden, hidden)?,
            output_norm: norm("attention.output.LayerNorm")?,
            intermediate: load("intermediate.dense", config.intermediate_size, hidden)?,
            ff_output: load("output.dense", hidden, config.intermediate_size)?,
            ff_norm: norm("output.LayerNorm")?,
            heads: config.num_attention_heads,
            head_dim: hidden / config.num_attention_heads,
            activation: config.activation(),
        })
    }

    fn forward(
        &self,
        xs: &Tensor,
        attention_bias: &Tensor,
        rel_embeddings: &Tensor,
        c2p_positions: &Tensor,
        p2c_positions: &Tensor,
        scale_factor: usize,
    ) -> Result<Tensor, ModelError> {
        let (batch, length, hidden) = xs.dims3().map_err(inference_error)?;
        let heads = self.heads;
        let head_dim = self.head_dim;
        let query = xs
            .apply(&self.query)
            .map_err(inference_error)?
            .reshape((batch, length, heads, head_dim))
            .map_err(inference_error)?
            .transpose(1, 2)
            .map_err(inference_error)?
            .contiguous()
            .map_err(inference_error)?;
        let key = xs
            .apply(&self.key)
            .map_err(inference_error)?
            .reshape((batch, length, heads, head_dim))
            .map_err(inference_error)?
            .transpose(1, 2)
            .map_err(inference_error)?
            .contiguous()
            .map_err(inference_error)?;
        let value = xs
            .apply(&self.value)
            .map_err(inference_error)?
            .reshape((batch, length, heads, head_dim))
            .map_err(inference_error)?
            .transpose(1, 2)
            .map_err(inference_error)?
            .contiguous()
            .map_err(inference_error)?;

        let positions = rel_embeddings.dim(0).map_err(inference_error)?;
        let position_query = self
            .query
            .forward(rel_embeddings)
            .map_err(inference_error)?
            .reshape((positions, heads, head_dim))
            .map_err(inference_error)?
            .transpose(0, 1)
            .map_err(inference_error)?
            .unsqueeze(0)
            .map_err(inference_error)?
            .broadcast_as((batch, heads, positions, head_dim))
            .map_err(inference_error)?
            .contiguous()
            .map_err(inference_error)?;
        let position_key = self
            .key
            .forward(rel_embeddings)
            .map_err(inference_error)?
            .reshape((positions, heads, head_dim))
            .map_err(inference_error)?
            .transpose(0, 1)
            .map_err(inference_error)?
            .unsqueeze(0)
            .map_err(inference_error)?
            .broadcast_as((batch, heads, positions, head_dim))
            .map_err(inference_error)?
            .contiguous()
            .map_err(inference_error)?;

        let scale = ((head_dim * scale_factor) as f64).sqrt();
        let content = query
            .matmul(
                &key.transpose(2, 3)
                    .map_err(inference_error)?
                    .contiguous()
                    .map_err(inference_error)?,
            )
            .map_err(inference_error)?;
        let content_to_position = query
            .matmul(
                &position_key
                    .transpose(2, 3)
                    .map_err(inference_error)?
                    .contiguous()
                    .map_err(inference_error)?,
            )
            .map_err(inference_error)?
            .contiguous()
            .map_err(inference_error)?
            .gather(
                &c2p_positions
                    .unsqueeze(0)
                    .map_err(inference_error)?
                    .unsqueeze(0)
                    .map_err(inference_error)?
                    .broadcast_as((batch, heads, length, length))
                    .map_err(inference_error)?
                    .contiguous()
                    .map_err(inference_error)?,
                D::Minus1,
            )
            .map_err(inference_error)?;
        let position_to_content = key
            .matmul(
                &position_query
                    .transpose(2, 3)
                    .map_err(inference_error)?
                    .contiguous()
                    .map_err(inference_error)?,
            )
            .map_err(inference_error)?
            .contiguous()
            .map_err(inference_error)?
            .gather(
                &p2c_positions
                    .unsqueeze(0)
                    .map_err(inference_error)?
                    .unsqueeze(0)
                    .map_err(inference_error)?
                    .broadcast_as((batch, heads, length, length))
                    .map_err(inference_error)?
                    .contiguous()
                    .map_err(inference_error)?,
                D::Minus1,
            )
            .map_err(inference_error)?
            .transpose(2, 3)
            .map_err(inference_error)?;

        let scores = content
            .add(&content_to_position)
            .map_err(inference_error)?
            .add(&position_to_content)
            .map_err(inference_error)?
            .affine(scale.recip(), 0.0)
            .map_err(inference_error)?
            .broadcast_add(attention_bias)
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
        let context = probabilities
            .matmul(&value)
            .map_err(inference_error)?
            .transpose(1, 2)
            .map_err(inference_error)?
            .reshape((batch, length, hidden))
            .map_err(inference_error)?;
        let attention_output = xs
            .add(&context.apply(&self.output).map_err(inference_error)?)
            .map_err(inference_error)?
            .apply(&self.output_norm)
            .map_err(inference_error)?;
        let feed_forward = attention_output
            .apply(&self.intermediate)
            .map_err(inference_error)?
            .apply(&self.activation)
            .map_err(inference_error)?
            .apply(&self.ff_output)
            .map_err(inference_error)?;
        attention_output
            .add(&feed_forward)
            .map_err(inference_error)?
            .apply(&self.ff_norm)
            .map_err(inference_error)
    }
}

/// A DeBERTa-v3 encoder producing `[batch, length, hidden]` states.
pub(crate) struct DebertaEncoder {
    word_embeddings: Embedding,
    embeddings_norm: LayerNorm,
    rel_embeddings: Tensor,
    layers: Vec<DebertaLayer>,
    c2p_positions: Tensor,
    p2c_positions: Tensor,
    scale_factor: usize,
    max_hidden: usize,
}

impl DebertaEncoder {
    pub fn load(
        vb: candle_nn::VarBuilder,
        config: &DebertaConfig,
        device: &Device,
    ) -> Result<Self, ModelError> {
        config.validate()?;
        let hidden = config.hidden_size;
        let word_embeddings = embedding(
            config.vocab_size,
            hidden,
            vb.pp("embeddings.word_embeddings"),
        )
        .map_err(|error| ModelError::load(error.to_string()))?;
        let embeddings_norm =
            layer_norm(hidden, config.layer_norm_eps, vb.pp("embeddings.LayerNorm"))
                .map_err(|error| ModelError::load(error.to_string()))?;
        let rel_weight = vb
            .get(
                (2 * config.position_buckets as usize, config.hidden_size),
                "encoder.rel_embeddings.weight",
            )
            .map_err(|error| ModelError::load(error.to_string()))?;
        let rel_embeddings = if config.norm_rel_ebd.trim().is_empty() {
            rel_weight
        } else {
            let norm = layer_norm(hidden, config.layer_norm_eps, vb.pp("encoder.LayerNorm"))
                .map_err(|error| ModelError::load(error.to_string()))?;
            rel_weight
                .apply(&norm)
                .map_err(|error| ModelError::load(error.to_string()))?
        };
        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for index in 0..config.num_hidden_layers {
            layers.push(DebertaLayer::load(
                vb.pp(format!("encoder.layer.{index}")),
                config,
            )?);
        }
        let (c2p, p2c) = relative_positions(
            config.max_position_embeddings,
            config.position_buckets as usize,
            config.max_relative_positions(),
        );
        let c2p_positions = Tensor::from_vec(
            c2p,
            (
                config.max_position_embeddings,
                config.max_position_embeddings,
            ),
            device,
        )
        .map_err(inference_error)?;
        let p2c_positions = Tensor::from_vec(
            p2c,
            (
                config.max_position_embeddings,
                config.max_position_embeddings,
            ),
            device,
        )
        .map_err(inference_error)?;
        Ok(Self {
            word_embeddings,
            embeddings_norm,
            rel_embeddings,
            layers,
            c2p_positions,
            p2c_positions,
            scale_factor: 1 + config
                .pos_att_type
                .iter()
                .filter(|kind| kind.as_str() == "c2p" || kind.as_str() == "p2c")
                .count(),
            max_hidden: config.max_position_embeddings,
        })
    }

    pub fn forward(&self, ids: &Tensor, mask: &Tensor) -> Result<Tensor, ModelError> {
        let (_batch, length) = ids.dims2().map_err(inference_error)?;
        if length > self.max_hidden {
            return Err(ModelError::inference(format!(
                "DeBERTa row is {length} tokens; the checkpoint reads {}",
                self.max_hidden
            )));
        }
        let dtype = self.rel_embeddings.dtype();
        let mut xs = self
            .word_embeddings
            .forward(ids)
            .map_err(inference_error)?
            .apply(&self.embeddings_norm)
            .map_err(inference_error)?
            .broadcast_mul(
                &mask
                    .unsqueeze(2)
                    .map_err(inference_error)?
                    .to_dtype(dtype)
                    .map_err(inference_error)?,
            )
            .map_err(inference_error)?;
        let pair = mask
            .unsqueeze(2)
            .map_err(inference_error)?
            .broadcast_mul(&mask.unsqueeze(1).map_err(inference_error)?)
            .map_err(inference_error)?;
        let attention_bias = pair
            .to_dtype(DType::F32)
            .map_err(inference_error)?
            .affine(-1.0, 1.0)
            .map_err(inference_error)?
            .affine(-1e9, 0.0)
            .map_err(inference_error)?
            .unsqueeze(1)
            .map_err(inference_error)?
            .to_dtype(dtype)
            .map_err(inference_error)?;
        let c2p = self
            .c2p_positions
            .narrow(0, 0, length)
            .map_err(inference_error)?
            .narrow(1, 0, length)
            .map_err(inference_error)?;
        let p2c = self
            .p2c_positions
            .narrow(0, 0, length)
            .map_err(inference_error)?
            .narrow(1, 0, length)
            .map_err(inference_error)?;
        for layer in &self.layers {
            xs = layer.forward(
                &xs,
                &attention_bias,
                &self.rel_embeddings,
                &c2p,
                &p2c,
                self.scale_factor,
            )?;
        }
        Ok(xs)
    }
}

fn relative_positions(length: usize, buckets: usize, max_position: usize) -> (Vec<u32>, Vec<u32>) {
    let mid = buckets / 2;
    let limit = 2 * buckets;
    let denominator = ((max_position - 1) as f64 / mid as f64).ln();
    let mut c2p = Vec::with_capacity(length * length);
    let mut p2c = Vec::with_capacity(length * length);
    for query in 0..length {
        for key in 0..length {
            let relative = query as i64 - key as i64;
            let abs_pos = if relative < mid as i64 && relative > -(mid as i64) {
                mid as i64 - 1
            } else {
                relative.abs()
            };
            let bucket = if abs_pos <= mid as i64 {
                relative
            } else {
                let sign = relative.signum();
                let log_pos = ((abs_pos as f64 / mid as f64).ln() / denominator * (mid - 1) as f64)
                    .ceil() as i64
                    + mid as i64;
                log_pos * sign
            };
            c2p.push((bucket + buckets as i64).clamp(0, limit as i64 - 1) as u32);
            p2c.push((-bucket + buckets as i64).clamp(0, limit as i64 - 1) as u32);
        }
    }
    (c2p, p2c)
}

fn inference_error(error: candle_core::Error) -> ModelError {
    ModelError::inference(error.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn relative_positions_use_the_log_buckets() {
        let (c2p, p2c) = relative_positions(3, 256, 512);
        // Row 0: relative positions 0, -1, -2.
        assert_eq!(&c2p[..3], &[256, 255, 254]);
        assert_eq!(&p2c[..3], &[256, 257, 258]);
        // Row 2 starts with +2, +1, 0.
        assert_eq!(&c2p[6..9], &[258, 257, 256]);
        assert_eq!(&p2c[6..9], &[254, 255, 256]);
    }
}
