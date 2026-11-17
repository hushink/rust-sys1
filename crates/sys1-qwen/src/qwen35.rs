use crate::config::Qwen35Config;
use crate::qwen3::{Mlp, RmsNorm, apply_rope, causal_bias, repeat_kv, rotary_tables};
use crate::weights::WeightStore;
use candle_core::{D, DType, Device, Result, Tensor};
use candle_nn::{Linear, Module, ops::softmax};

const CHUNK: usize = 64;

fn rms_norm_gated(xs: &Tensor, weight: &Tensor, eps: f32) -> Result<Tensor> {
    candle_nn::ops::rms_norm_slow(xs, weight, eps)
}

/// `log(1 + exp(x))` in its stable form.
fn softplus(xs: &Tensor) -> Result<Tensor> {
    let tail = xs.abs()?.neg()?.exp()?;
    let tail = (tail + 1.0)?.log()?;
    xs.relu()? + tail
}

/// `x / sqrt(sum(x²) + 1e-6)` over the last axis.
fn l2norm(xs: &Tensor) -> Result<Tensor> {
    let squared = xs.sqr()?.sum_keepdim(D::Minus1)?;
    let norm = (squared + 1e-6)?.sqrt()?;
    xs.broadcast_div(&norm)
}

/// Precomputed masks for the gated delta rule's 64-token chunks.
struct ChunkMasks {
    lower: Tensor,
    strict: Tensor,
    doubling: Vec<Tensor>,
}

impl ChunkMasks {
    fn new(dtype: DType, device: &Device) -> Result<Self> {
        let mut lower = vec![0f32; CHUNK * CHUNK];
        let mut strict = vec![0f32; CHUNK * CHUNK];
        for query in 0..CHUNK {
            for key in 0..CHUNK {
                lower[query * CHUNK + key] = if key <= query { 1.0 } else { 0.0 };
                strict[query * CHUNK + key] = if key < query { 1.0 } else { 0.0 };
            }
        }
        let lower = Tensor::from_vec(lower, (CHUNK, CHUNK), device)?.to_dtype(dtype)?;
        let strict = Tensor::from_vec(strict, (CHUNK, CHUNK), device)?.to_dtype(dtype)?;
        let mut doubling = Vec::new();
        let mut size = 1;
        while size < CHUNK {
            let block = (2 * size) as f32;
            let values: Vec<f32> = (0..CHUNK * CHUNK)
                .map(|index| {
                    let row = index / CHUNK;
                    let column = index % CHUNK;
                    let same = (row as f32 / block).floor() == (column as f32 / block).floor();
                    let lower_left = (row % (2 * size)) >= size && (column % (2 * size)) < size;
                    if same && lower_left { 1.0 } else { 0.0 }
                })
                .collect();
            doubling.push(Tensor::from_vec(values, (CHUNK, CHUNK), device)?.to_dtype(dtype)?);
            size *= 2;
        }
        Ok(Self {
            lower,
            strict,
            doubling,
        })
    }

    fn masked_bias(mask: &Tensor, _device: &Device) -> Result<Tensor> {
        // 1 -> 0, 0 -> -1e9
        mask.affine(1e9, -1e9)
    }

    /// `(I + A)^-1` for strictly lower-triangular `A`.
    fn unit_lower_inverse(&self, a: &Tensor, dtype: DType, device: &Device) -> Result<Tensor> {
        let dims = a.dims().to_vec();
        let mut x = Tensor::eye(CHUNK, dtype, device)?.broadcast_as(dims.clone())?;
        for mask in &self.doubling {
            let mask = mask.broadcast_as(dims.clone())?;
            let block = a.broadcast_mul(&mask)?;
            let product = mm(&mm(&x, &block)?, &x)?;
            x = (x - product)?;
        }
        Ok(x)
    }
}

struct DeltaNet {
    qkv: Linear,
    conv: Tensor,
    kernel: usize,
    z: Linear,
    b: Linear,
    a: Linear,
    neg_a: Tensor,
    dt_bias: Tensor,
    norm_weight: Tensor,
    out: Linear,
    key_heads: usize,
    value_heads: usize,
    key_dim: usize,
    value_dim: usize,
    eps: f32,
}

impl DeltaNet {
    fn load(store: &WeightStore, prefix: &str, config: &Qwen35Config) -> Result<Self> {
        let hidden = config.hidden_size;
        let key_dim = config.linear_num_key_heads * config.linear_key_head_dim;
        let value_dim = config.linear_num_value_heads * config.linear_value_head_dim;
        let conv_dim = 2 * key_dim + value_dim;
        let kernel = config.linear_conv_kernel_dim;
        let neg_a = store
            .get(&format!("{prefix}A_log"), &[config.linear_num_value_heads])?
            .exp()?
            .neg()?;
        Ok(Self {
            qkv: store.linear(&format!("{prefix}in_proj_qkv"), conv_dim, hidden, false)?,
            conv: store.get(&format!("{prefix}conv1d.weight"), &[conv_dim, 1, kernel])?,
            kernel,
            z: store.linear(&format!("{prefix}in_proj_z"), value_dim, hidden, false)?,
            b: store.linear(
                &format!("{prefix}in_proj_b"),
                config.linear_num_value_heads,
                hidden,
                false,
            )?,
            a: store.linear(
                &format!("{prefix}in_proj_a"),
                config.linear_num_value_heads,
                hidden,
                false,
            )?,
            neg_a,
            dt_bias: store.get(
                &format!("{prefix}dt_bias"),
                &[config.linear_num_value_heads],
            )?,
            norm_weight: store.get(
                &format!("{prefix}norm.weight"),
                &[config.linear_value_head_dim],
            )?,
            out: store.linear(&format!("{prefix}out_proj"), hidden, value_dim, false)?,
            key_heads: config.linear_num_key_heads,
            value_heads: config.linear_num_value_heads,
            key_dim: config.linear_key_head_dim,
            value_dim: config.linear_value_head_dim,
            eps: config.rms_norm_eps as f32,
        })
    }

    fn forward(&self, xs: &Tensor, masks: &ChunkMasks) -> Result<Tensor> {
        let (rows, seq, _hidden) = xs.dims3()?;
        let dtype = xs.dtype();
        let device = xs.device();
        let key_width = self.key_heads * self.key_dim;
        let value_width = self.value_heads * self.value_dim;
        let projected = self.qkv.forward(xs)?;
        let mixed = depthwise_conv1d(&projected.transpose(1, 2)?, &self.conv, self.kernel)?
            .transpose(1, 2)?
            .silu()?;

        let q = mixed
            .narrow(D::Minus1, 0, key_width)?
            .reshape((rows, seq, self.key_heads, self.key_dim))?
            .transpose(1, 2)?;
        let k = mixed
            .narrow(D::Minus1, key_width, key_width)?
            .reshape((rows, seq, self.key_heads, self.key_dim))?
            .transpose(1, 2)?;
        let v = mixed
            .narrow(D::Minus1, 2 * key_width, value_width)?
            .reshape((rows, seq, self.value_heads, self.value_dim))?
            .transpose(1, 2)?;
        let repeats = self.value_heads / self.key_heads;
        let q = repeat_kv(&q, repeats)?;
        let k = repeat_kv(&k, repeats)?;

        let beta = candle_nn::ops::sigmoid(&self.b.forward(xs)?)?.transpose(1, 2)?;
        let gate = softplus(&self.a.forward(xs)?.broadcast_add(&self.dt_bias)?)?
            .broadcast_mul(&self.neg_a)?
            .transpose(1, 2)?;
        let core = gated_delta_rule(&q, &k, &v, &gate, &beta, masks, dtype, device)?;
        let core = core
            .transpose(1, 2)?
            .reshape((rows, seq, self.value_heads, self.value_dim))?;
        let z = self
            .z
            .forward(xs)?
            .reshape((rows, seq, self.value_heads, self.value_dim))?;
        let gate = z.silu()?;
        let normalized = rms_norm_gated(&core, &self.norm_weight, self.eps)?
            .mul(&gate)?
            .reshape((rows, seq, value_width))?;
        self.out.forward(&normalized)
    }
}

#[allow(clippy::too_many_arguments)]
fn gated_delta_rule(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    gate: &Tensor,
    beta: &Tensor,
    masks: &ChunkMasks,
    dtype: DType,
    device: &Device,
) -> Result<Tensor> {
    let (batch, heads, seq, _) = q.dims4()?;
    let key_dim = q.dim(3)?;
    let value_dim = v.dim(3)?;
    let chunks = seq / CHUNK;
    let q = l2norm(q)?.affine((key_dim as f64).powf(-0.5), 0.0)?;
    let k = l2norm(k)?;
    let beta = beta.unsqueeze(D::Minus1)?;
    let v_beta = v.broadcast_mul(&beta)?;
    let k_beta = k.broadcast_mul(&beta)?;
    let q = q.reshape((batch, heads, chunks, CHUNK, key_dim))?;
    let k = k.reshape((batch, heads, chunks, CHUNK, key_dim))?;
    let k_beta = k_beta.reshape((batch, heads, chunks, CHUNK, key_dim))?;
    let v_beta = v_beta.reshape((batch, heads, chunks, CHUNK, value_dim))?;
    let gate = gate.reshape((batch, heads, chunks, CHUNK))?;
    let cumulative = gate.cumsum(D::Minus1)?;

    let lower_bias = ChunkMasks::masked_bias(&masks.lower, device)?
        .broadcast_as((batch, heads, chunks, CHUNK, CHUNK))?;
    let difference = cumulative
        .unsqueeze(D::Minus1)?
        .broadcast_sub(&cumulative.unsqueeze(D::Minus2)?)?;
    let pair = difference.broadcast_add(&lower_bias)?.exp()?;
    let k_transposed = k.transpose(3, 4)?;
    let ut = mm(&k_beta, &k_transposed)?.broadcast_mul(&pair)?;
    let intra = mm(&q, &k_transposed)?.broadcast_mul(&pair)?;
    let exp_cumulative = cumulative.exp()?.unsqueeze(D::Minus1)?;
    let decayed_k_beta = k_beta.broadcast_mul(&exp_cumulative)?;
    let strict = masks.strict.broadcast_as(ut.dims().to_vec())?;
    let strict_ut = ut.broadcast_mul(&strict)?;
    let inverse = masks.unit_lower_inverse(&strict_ut, dtype, device)?;
    let new_v = mm(&inverse, &v_beta)?;
    let k_cumdecay = mm(&inverse, &decayed_k_beta)?;
    let q = q.broadcast_mul(&exp_cumulative)?;
    let last = cumulative.narrow(3, CHUNK - 1, 1)?;
    let decayed = (last.broadcast_sub(&cumulative)?)
        .exp()?
        .unsqueeze(D::Minus1)?;
    let k = k.broadcast_mul(&decayed)?;
    let chunk_decay = last.exp()?.reshape((batch, heads, chunks, 1, 1))?;

    let mut state = Tensor::zeros((batch, heads, key_dim, value_dim), dtype, device)?;
    let mut outputs = Vec::with_capacity(chunks);
    let at = |tensor: &Tensor, index: usize, width: usize| -> Result<Tensor> {
        tensor
            .narrow(2, index, 1)?
            .squeeze(2)?
            .reshape((batch, heads, CHUNK, width))
    };
    for index in 0..chunks {
        let new_v_i = at(&new_v, index, value_dim)?;
        let k_cumdecay_i = at(&k_cumdecay, index, key_dim)?;
        let q_i = at(&q, index, key_dim)?;
        let intra_i = at(&intra, index, CHUNK)?;
        let k_i = at(&k, index, key_dim)?;
        let v_new = new_v_i.broadcast_sub(&mm(&k_cumdecay_i, &state)?)?;
        let out = mm(&q_i, &state)?.broadcast_add(&mm(&intra_i, &v_new)?)?;
        let decay = chunk_decay.narrow(2, index, 1)?.squeeze(2)?;
        state = state
            .broadcast_mul(&decay)?
            .broadcast_add(&mm(&k_i.transpose(2, 3)?, &v_new)?)?;
        outputs.push(out);
    }
    let refs: Vec<&Tensor> = outputs.iter().collect();
    Tensor::cat(&refs, 2)
}

struct Attention {
    q: Linear,
    k: Linear,
    v: Linear,
    o: Linear,
    q_norm: RmsNorm,
    k_norm: RmsNorm,
}

impl Attention {
    fn load(store: &WeightStore, prefix: &str, config: &Qwen35Config) -> Result<Self> {
        let hidden = config.hidden_size;
        let head_dim = config.head_dim;
        let eps = config.rms_norm_eps as f32;
        Ok(Self {
            q: store.linear(
                &format!("{prefix}q_proj"),
                config.num_attention_heads * head_dim * 2,
                hidden,
                config.attention_bias,
            )?,
            k: store.linear(
                &format!("{prefix}k_proj"),
                config.num_key_value_heads * head_dim,
                hidden,
                config.attention_bias,
            )?,
            v: store.linear(
                &format!("{prefix}v_proj"),
                config.num_key_value_heads * head_dim,
                hidden,
                config.attention_bias,
            )?,
            o: store.linear(
                &format!("{prefix}o_proj"),
                hidden,
                config.num_attention_heads * head_dim,
                config.attention_bias,
            )?,
            q_norm: RmsNorm::load(
                store,
                &format!("{prefix}q_norm.weight"),
                head_dim,
                eps,
                true,
            )?,
            k_norm: RmsNorm::load(
                store,
                &format!("{prefix}k_norm.weight"),
                head_dim,
                eps,
                true,
            )?,
        })
    }

    fn forward(
        &self,
        xs: &Tensor,
        config: &Qwen35Config,
        cos: &Tensor,
        sin: &Tensor,
        bias: &Tensor,
    ) -> Result<Tensor> {
        let (rows, seq, _) = xs.dims3()?;
        let (heads, kv_heads, head_dim) = (
            config.num_attention_heads,
            config.num_key_value_heads,
            config.head_dim,
        );
        let qg = self
            .q
            .forward(xs)?
            .reshape((rows, seq, heads, 2 * head_dim))?;
        let q = qg.narrow(3, 0, head_dim)?;
        let gate = qg
            .narrow(3, head_dim, head_dim)?
            .reshape((rows, seq, heads * head_dim))?;
        let q = self.q_norm.forward(&q)?.transpose(1, 2)?;
        let k = self
            .k
            .forward(xs)?
            .reshape((rows, seq, kv_heads, head_dim))?;
        let k = self.k_norm.forward(&k)?.transpose(1, 2)?;
        let v = self
            .v
            .forward(xs)?
            .reshape((rows, seq, kv_heads, head_dim))?
            .transpose(1, 2)?;
        let rotary_dim = config.rotary_dim();
        let q = apply_rope(&q, cos, sin, rotary_dim)?;
        let k = apply_rope(&k, cos, sin, rotary_dim)?;
        let repeats = heads / kv_heads;
        let k = repeat_kv(&k, repeats)?;
        let v = repeat_kv(&v, repeats)?;
        let dtype = q.dtype();
        let scores = (q.contiguous()?.matmul(&k.transpose(2, 3)?.contiguous()?)?
            * (head_dim as f64).powf(-0.5))?;
        let scores = scores.broadcast_add(bias)?;
        let probabilities = if matches!(dtype, DType::F16 | DType::BF16) {
            softmax(&scores.to_dtype(DType::F32)?, D::Minus1)?.to_dtype(dtype)?
        } else {
            softmax(&scores, D::Minus1)?
        };
        let attention = probabilities
            .matmul(&v.contiguous()?)?
            .transpose(1, 2)?
            .reshape((rows, seq, heads * head_dim))?
            .mul(&candle_nn::ops::sigmoid(&gate)?)?;
        self.o.forward(&attention)
    }
}

enum Mixer {
    Linear(DeltaNet),
    Full(Attention),
}

struct Layer {
    input_norm: RmsNorm,
    mixer: Mixer,
    post_norm: RmsNorm,
    mlp: Mlp,
}

/// Causal depthwise convolution over `[batch, channels, length]`.
///
/// Candle's generic `conv1d` is very slow on CUDA for this shape; the kernel is
/// at most 4 taps, so shifted multiplies are both simpler and much faster.
fn depthwise_conv1d(xs: &Tensor, weight: &Tensor, kernel: usize) -> Result<Tensor> {
    let (batch, channels, length) = xs.dims3()?;
    let coefficient =
        |tap: usize| -> Result<Tensor> { weight.narrow(2, tap, 1)?.reshape((1, channels, 1)) };
    if kernel <= 1 {
        return xs.broadcast_mul(&coefficient(0)?);
    }
    let padding = Tensor::zeros((batch, channels, kernel - 1), xs.dtype(), xs.device())?;
    let padded = Tensor::cat(&[&padding, xs], 2)?;
    let mut output = Tensor::zeros((batch, channels, length), xs.dtype(), xs.device())?;
    for tap in 0..kernel {
        let shifted = padded.narrow(2, tap, length)?;
        let term = shifted.broadcast_mul(&coefficient(tap)?)?;
        output = output.add(&term)?;
    }
    Ok(output)
}

/// The hybrid Qwen3.5 decoder: Gated DeltaNet and gated full-attention layers.
fn mm(lhs: &Tensor, rhs: &Tensor) -> Result<Tensor> {
    let lhs_shape = lhs.dims().to_vec();
    let rhs_shape = rhs.dims().to_vec();
    let (rows, inner) = match lhs_shape.as_slice() {
        [.., rows, inner] => (*rows, *inner),
        _ => candle_core::bail!("matmul lhs needs at least two dimensions"),
    };
    let out = match rhs_shape.as_slice() {
        [.., rhs_inner, columns] => {
            if *rhs_inner != inner {
                candle_core::bail!("matmul inner dimensions differ: {inner} vs {rhs_inner}");
            }
            *columns
        }
        _ => candle_core::bail!("matmul rhs needs at least two dimensions"),
    };
    let batch: usize = lhs_shape[..lhs_shape.len() - 2].iter().product();
    let lhs = lhs.reshape((batch, rows, inner))?.contiguous()?;
    let rhs = rhs.reshape((batch, inner, out))?.contiguous()?;
    let product = lhs.matmul(&rhs)?;
    let mut shape = lhs_shape[..lhs_shape.len() - 2].to_vec();
    shape.push(rows);
    shape.push(out);
    product.reshape(shape)
}

pub struct Qwen35 {
    config: Qwen35Config,
    embed: Tensor,
    lm_head: Option<Tensor>,
    layers: Vec<Layer>,
    final_norm: RmsNorm,
    masks: ChunkMasks,
    device: Device,
}

impl Qwen35 {
    pub fn load(config: Qwen35Config, store: &WeightStore) -> Result<Self> {
        let prefix = config.prefix.clone();
        let prefix = &prefix;
        let hidden = config.hidden_size;
        let eps = config.rms_norm_eps as f32;
        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for (index, kind) in config.layer_types.iter().enumerate() {
            let layer = format!("{prefix}layers.{index}");
            let mixer = match kind.as_str() {
                "linear_attention" => Mixer::Linear(DeltaNet::load(
                    store,
                    &format!("{layer}.linear_attn."),
                    &config,
                )?),
                "full_attention" => Mixer::Full(Attention::load(
                    store,
                    &format!("{layer}.self_attn."),
                    &config,
                )?),
                other => candle_core::bail!("unknown Qwen3.5 layer type {other:?}"),
            };
            layers.push(Layer {
                input_norm: RmsNorm::load(
                    store,
                    &format!("{layer}.input_layernorm.weight"),
                    hidden,
                    eps,
                    true,
                )?,
                mixer,
                post_norm: RmsNorm::load(
                    store,
                    &format!("{layer}.post_attention_layernorm.weight"),
                    hidden,
                    eps,
                    true,
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
        let lm_head = if !config.tie_word_embeddings {
            let name = if store.contains("lm_head.weight") {
                "lm_head.weight".to_owned()
            } else {
                format!("{prefix}lm_head.weight")
            };
            if !store.contains(&name) {
                candle_core::bail!("untied Qwen3.5 checkpoint is missing lm_head.weight");
            }
            Some(store.get(&name, &[config.vocab_size, hidden])?)
        } else {
            None
        };
        Ok(Self {
            masks: ChunkMasks::new(embed.dtype(), embed.device())?,
            device: embed.device().clone(),
            config,
            embed,
            lm_head,
            layers,
            final_norm: RmsNorm::load(store, &format!("{prefix}norm.weight"), hidden, eps, true)?,
        })
    }

    pub fn config(&self) -> &Qwen35Config {
        &self.config
    }

    pub fn output_embedding(&self) -> &Tensor {
        self.lm_head.as_ref().unwrap_or(&self.embed)
    }

    pub fn forward(&self, ids: &Tensor) -> Result<Tensor> {
        let (rows, seq) = ids.dims2()?;
        let padded = seq.div_ceil(CHUNK) * CHUNK;
        let ids = if padded == seq {
            ids.clone()
        } else {
            let padding = Tensor::zeros((rows, padded - seq), DType::U32, ids.device())?;
            Tensor::cat(&[ids, &padding], 1)?
        };
        let hidden = self.embed.dim(1)?;
        let flat = ids.flatten_all()?;
        let mut xs = self
            .embed
            .index_select(&flat, 0)?
            .reshape((rows, padded, hidden))?;
        let dtype = self.embed.dtype();
        let (cos, sin) = rotary_tables(
            self.config.rope_theta,
            self.config.rotary_dim(),
            padded,
            dtype,
            &self.device,
        )?;
        let bias = causal_bias(padded, dtype, &self.device)?;
        for layer in &self.layers {
            let normed = layer.input_norm.forward(&xs)?;
            let mixed = match &layer.mixer {
                Mixer::Linear(delta) => delta.forward(&normed, &self.masks)?,
                Mixer::Full(attention) => {
                    attention.forward(&normed, &self.config, &cos, &sin, &bias)?
                }
            };
            xs = (xs + mixed)?;
            let feed_forward = layer.mlp.forward(&layer.post_norm.forward(&xs)?)?;
            xs = (xs + feed_forward)?;
        }
        let hidden = self.final_norm.forward(&xs)?;
        if padded == seq {
            Ok(hidden)
        } else {
            hidden.narrow(1, 0, seq)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn masks_have_the_expected_triangle() {
        let masks = ChunkMasks::new(DType::F32, &Device::Cpu).unwrap();
        let lower = masks.lower.to_vec2::<f32>().unwrap();
        assert_eq!(lower[0][0], 1.0);
        assert_eq!(lower[1][0], 1.0);
        assert_eq!(lower[0][1], 0.0);
        assert_eq!(masks.doubling.len(), 6);
    }

    #[test]
    fn causal_depthwise_conv_shifts_its_taps() {
        let xs = Tensor::new(&[[[1f32, 2.0, 3.0, 4.0]]], &Device::Cpu).unwrap();
        let weight = Tensor::new(&[[[1f32, 10.0, 100.0]]], &Device::Cpu).unwrap();
        let output = depthwise_conv1d(&xs, &weight, 3).unwrap();
        let values = output.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert_eq!(values, vec![100.0, 210.0, 321.0, 432.0]);
    }
}
