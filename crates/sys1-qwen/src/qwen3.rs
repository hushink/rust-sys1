use crate::config::Qwen3Config;
use crate::weights::WeightStore;
use candle_core::{D, DType, Device, Result, Tensor};
use candle_nn::{Linear, Module, ops::softmax};

/// `Qwen3RMSNorm`: `w * x / sqrt(mean(x²) + eps)`.
pub struct RmsNorm {
    weight: Tensor,
    eps: f32,
}

impl RmsNorm {
    pub fn load(
        store: &WeightStore,
        name: &str,
        dim: usize,
        eps: f32,
        zero_centered: bool,
    ) -> Result<Self> {
        let mut weight = store.get(name, &[dim])?;
        if zero_centered {
            weight = weight.affine(1.0, 1.0)?;
        }
        Ok(Self { weight, eps })
    }

    pub fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        candle_nn::ops::rms_norm_slow(xs, &self.weight, self.eps)
    }
}

/// `Qwen3MLP`: `down(silu(gate(x)) * up(x))`.
pub struct Mlp {
    gate: Linear,
    up: Linear,
    down: Linear,
}

impl Mlp {
    pub fn load(
        store: &WeightStore,
        prefix: &str,
        hidden: usize,
        intermediate: usize,
    ) -> Result<Self> {
        Ok(Self {
            gate: store.linear(&format!("{prefix}gate_proj"), intermediate, hidden, false)?,
            up: store.linear(&format!("{prefix}up_proj"), intermediate, hidden, false)?,
            down: store.linear(&format!("{prefix}down_proj"), hidden, intermediate, false)?,
        })
    }

    pub fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let gate = self.gate.forward(xs)?.silu()?;
        let up = self.up.forward(xs)?;
        self.down.forward(&(gate * up)?)
    }
}

/// Rotary cosine/sine tables for positions `0..len`, `dim / 2` frequencies.
pub fn rotary_tables(
    theta: f64,
    dim: usize,
    len: usize,
    dtype: DType,
    device: &Device,
) -> Result<(Tensor, Tensor)> {
    let half = dim / 2;
    let inv_freq: Vec<f64> = (0..half)
        .map(|index| 1.0 / theta.powf(2.0 * index as f64 / dim as f64))
        .collect();
    let inv_freq = Tensor::from_vec(inv_freq, (1, half), device)?;
    let positions = Tensor::arange(0u32, len as u32, device)?
        .to_dtype(DType::F64)?
        .reshape((len, 1))?;
    let frequencies = positions.matmul(&inv_freq)?;
    Ok((
        frequencies.cos()?.to_dtype(dtype)?,
        frequencies.sin()?.to_dtype(dtype)?,
    ))
}

/// Apply rotary embeddings to the first `dim` channels of `xs`.
pub fn apply_rope(xs: &Tensor, cos: &Tensor, sin: &Tensor, dim: usize) -> Result<Tensor> {
    let total = xs.dim(D::Minus1)?;
    if dim == total {
        return candle_nn::rotary_emb::rope(&xs.contiguous()?, cos, sin);
    }
    let rotated = xs.narrow(D::Minus1, 0, dim)?.contiguous()?;
    let rotated = candle_nn::rotary_emb::rope(&rotated, cos, sin)?;
    let rest = xs.narrow(D::Minus1, dim, total - dim)?;
    Tensor::cat(&[rotated, rest], D::Minus1)
}

/// Repeat grouped-query key/value heads with `repeat_interleave` semantics.
pub fn repeat_kv(xs: &Tensor, repeats: usize) -> Result<Tensor> {
    if repeats == 1 {
        return Ok(xs.clone());
    }
    let (batch, heads, length, dim) = xs.dims4()?;
    xs.unsqueeze(2)?
        .broadcast_as((batch, heads, repeats, length, dim))?
        .reshape((batch, heads * repeats, length, dim))
}

/// Causal attention bias added before softmax: `0` on and below the diagonal,
/// `-1e9` above it.
pub fn causal_bias(length: usize, dtype: DType, device: &Device) -> Result<Tensor> {
    let mask = Tensor::tril2(length, DType::F32, device)?;
    let bias = mask.affine(1e9, -1e9)?;
    bias.to_dtype(dtype)?.reshape((1, 1, length, length))
}

fn attention(q: &Tensor, k: &Tensor, v: &Tensor, bias: &Tensor, scale: f64) -> Result<Tensor> {
    let dtype = q.dtype();
    let scores = (q.contiguous()?.matmul(&k.transpose(2, 3)?.contiguous()?)? * scale)?;
    let scores = scores.broadcast_add(bias)?;
    let probabilities = if matches!(dtype, DType::F16 | DType::BF16) {
        softmax(&scores.to_dtype(DType::F32)?, D::Minus1)?.to_dtype(dtype)?
    } else {
        softmax(&scores, D::Minus1)?
    };
    probabilities.matmul(&v.contiguous()?)
}

struct Layer {
    input_norm: RmsNorm,
    q: Linear,
    k: Linear,
    v: Linear,
    o: Linear,
    q_norm: RmsNorm,
    k_norm: RmsNorm,
    post_norm: RmsNorm,
    mlp: Mlp,
}

/// The dense Qwen3 decoder (Qwen3Guard and similar checkpoints).
pub struct Qwen3 {
    heads: usize,
    kv_heads: usize,
    head_dim: usize,
    embed: Tensor,
    layers: Vec<Layer>,
    final_norm: RmsNorm,
    rope_theta: f64,
    device: Device,
}

impl Qwen3 {
    pub fn load(config: &Qwen3Config, store: &WeightStore) -> Result<Self> {
        let prefix = &config.prefix;
        let hidden = config.hidden_size;
        let eps = config.rms_norm_eps as f32;
        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for index in 0..config.num_hidden_layers {
            let layer = format!("{prefix}layers.{index}");
            let attention = format!("{layer}.self_attn");
            layers.push(Layer {
                input_norm: RmsNorm::load(
                    store,
                    &format!("{layer}.input_layernorm.weight"),
                    hidden,
                    eps,
                    false,
                )?,
                q: store.linear(
                    &format!("{attention}.q_proj"),
                    config.num_attention_heads * config.head_dim,
                    hidden,
                    config.attention_bias,
                )?,
                k: store.linear(
                    &format!("{attention}.k_proj"),
                    config.num_key_value_heads * config.head_dim,
                    hidden,
                    config.attention_bias,
                )?,
                v: store.linear(
                    &format!("{attention}.v_proj"),
                    config.num_key_value_heads * config.head_dim,
                    hidden,
                    config.attention_bias,
                )?,
                o: store.linear(
                    &format!("{attention}.o_proj"),
                    hidden,
                    config.num_attention_heads * config.head_dim,
                    config.attention_bias,
                )?,
                q_norm: RmsNorm::load(
                    store,
                    &format!("{attention}.q_norm.weight"),
                    config.head_dim,
                    eps,
                    false,
                )?,
                k_norm: RmsNorm::load(
                    store,
                    &format!("{attention}.k_norm.weight"),
                    config.head_dim,
                    eps,
                    false,
                )?,
                post_norm: RmsNorm::load(
                    store,
                    &format!("{layer}.post_attention_layernorm.weight"),
                    hidden,
                    eps,
                    false,
                )?,
                mlp: Mlp::load(
                    store,
                    &format!("{layer}.mlp."),
                    hidden,
                    config.intermediate_size,
                )?,
            });
        }
        let embed = store.get(
            &format!("{prefix}embed_tokens.weight"),
            &[config.vocab_size, hidden],
        )?;
        let device = embed.device().clone();
        Ok(Self {
            heads: config.num_attention_heads,
            kv_heads: config.num_key_value_heads,
            head_dim: config.head_dim,
            embed,
            layers,
            final_norm: RmsNorm::load(store, &format!("{prefix}norm.weight"), hidden, eps, false)?,
            rope_theta: config.rope_theta,
            device,
        })
    }

    pub fn output_embedding(&self) -> &Tensor {
        &self.embed
    }

    pub fn forward(&self, ids: &Tensor) -> Result<Tensor> {
        let (rows, seq) = ids.dims2()?;
        let hidden = self.embed.dim(1)?;
        let dtype = self.embed.dtype();
        let flat = ids.flatten_all()?;
        let mut xs = self
            .embed
            .index_select(&flat, 0)?
            .reshape((rows, seq, hidden))?;
        let (cos, sin) = rotary_tables(self.rope_theta, self.head_dim, seq, dtype, &self.device)?;
        let bias = causal_bias(seq, dtype, &self.device)?;
        let scale = (self.head_dim as f64).powf(-0.5);
        let repeats = self.heads / self.kv_heads;
        let heads = |xs: Tensor, count: usize, norm: &RmsNorm| -> Result<Tensor> {
            let xs = xs.reshape((rows, seq, count, self.head_dim))?;
            let xs = norm.forward(&xs)?;
            xs.transpose(1, 2)
        };

        for layer in &self.layers {
            let normed = layer.input_norm.forward(&xs)?;
            let q = heads(layer.q.forward(&normed)?, self.heads, &layer.q_norm)?;
            let k = heads(layer.k.forward(&normed)?, self.kv_heads, &layer.k_norm)?;
            let v = layer
                .v
                .forward(&normed)?
                .reshape((rows, seq, self.kv_heads, self.head_dim))?
                .transpose(1, 2)?;
            let q = apply_rope(&q, &cos, &sin, self.head_dim)?;
            let k = apply_rope(&k, &cos, &sin, self.head_dim)?;
            let k = repeat_kv(&k, repeats)?;
            let v = repeat_kv(&v, repeats)?;
            let attention = attention(&q, &k, &v, &bias, scale)?
                .transpose(1, 2)?
                .reshape((rows, seq, self.heads * self.head_dim))?;
            let attention = layer.o.forward(&attention)?;
            xs = (xs + attention)?;
            let feed_forward = layer.mlp.forward(&layer.post_norm.forward(&xs)?)?;
            xs = (xs + feed_forward)?;
        }
        self.final_norm.forward(&xs)
    }
}

/// A model's normed hidden state at one position per row, multiplied by the
/// rows of `embedding` named in `ids` (a tied or untied LM head restricted to
/// candidate tokens).
pub fn candidate_logits(
    hidden: &Tensor,
    positions: &Tensor,
    embedding: &Tensor,
    ids: &Tensor,
) -> Result<Tensor> {
    let (rows, _seq, width) = hidden.dims3()?;
    let at = positions
        .reshape((rows, 1, 1))?
        .broadcast_as((rows, 1, width))?;
    let selected = hidden
        .contiguous()?
        .gather(&at.contiguous()?, 1)?
        .reshape((rows, width))?;
    let rows = embedding
        .index_select(&ids.contiguous()?, 0)?
        .transpose(0, 1)?;
    selected.matmul(&rows)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn causal_bias_masks_future_tokens() {
        let bias = causal_bias(2, DType::F32, &Device::Cpu).unwrap();
        let values = bias.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert_eq!(values, vec![0.0, -1e9, 0.0, 0.0], "{values:?}");
    }

    #[test]
    fn repeat_kv_interleaves_heads() {
        let xs = Tensor::new(&[[[[1f32]], [[2f32]]]], &Device::Cpu).unwrap();
        let repeated = repeat_kv(&xs, 3).unwrap();
        let values = repeated.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert_eq!(values, vec![1.0, 1.0, 1.0, 2.0, 2.0, 2.0]);
    }
}
