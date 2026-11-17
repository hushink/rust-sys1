use crate::ModernBertConfig;
use candle_core::{D, DType, Device, Result, Tensor};
use candle_nn::{
    Embedding, LayerNorm, Linear, Module, VarBuilder, embedding, layer_norm_no_bias, ops::softmax,
};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

const ATTENTION_MASK_VALUE: f32 = -10_000.0;

struct RotaryEmbedding {
    sin: Tensor,
    cos: Tensor,
}

impl RotaryEmbedding {
    fn new(
        dtype: DType,
        config: &ModernBertConfig,
        rope_theta: f64,
        device: &Device,
    ) -> Result<Self> {
        let dim = config.hidden_size / config.num_attention_heads;
        let inv_freq: Vec<_> = (0..dim)
            .step_by(2)
            .map(|index| 1f32 / rope_theta.powf(index as f64 / dim as f64) as f32)
            .collect();
        let inv_freq_len = inv_freq.len();
        let inv_freq = Tensor::from_vec(inv_freq, (1, inv_freq_len), device)?.to_dtype(dtype)?;
        let positions = Tensor::arange(0u32, config.max_position_embeddings as u32, device)?
            .to_dtype(dtype)?
            .reshape((config.max_position_embeddings, 1))?;
        let frequencies = positions.matmul(&inv_freq)?;
        Ok(Self {
            sin: frequencies.sin()?,
            cos: frequencies.cos()?,
        })
    }

    fn apply(&self, q: &Tensor, k: &Tensor) -> Result<(Tensor, Tensor)> {
        let dtype = q.dtype();
        let rotary_dtype = self.cos.dtype();
        let q = candle_nn::rotary_emb::rope(
            &q.to_dtype(rotary_dtype)?.contiguous()?,
            &self.cos,
            &self.sin,
        )?
        .to_dtype(dtype)?;
        let k = candle_nn::rotary_emb::rope(
            &k.to_dtype(rotary_dtype)?.contiguous()?,
            &self.cos,
            &self.sin,
        )?
        .to_dtype(dtype)?;
        Ok((q, k))
    }
}

struct Attention {
    qkv: Linear,
    projection: Linear,
    heads: usize,
    head_size: usize,
    rotary: Arc<RotaryEmbedding>,
    compute_dtype: DType,
}

impl Attention {
    fn load(
        vb: VarBuilder,
        config: &ModernBertConfig,
        rotary: Arc<RotaryEmbedding>,
        compute_dtype: DType,
    ) -> Result<Self> {
        Ok(Self {
            qkv: linear_no_bias_dtype(
                config.hidden_size,
                config.hidden_size * 3,
                vb.pp("Wqkv"),
                compute_dtype,
            )?,
            projection: linear_no_bias_dtype(
                config.hidden_size,
                config.hidden_size,
                vb.pp("Wo"),
                compute_dtype,
            )?,
            heads: config.num_attention_heads,
            head_size: config.hidden_size / config.num_attention_heads,
            rotary,
            compute_dtype,
        })
    }

    fn forward(&self, xs: &Tensor, mask: Option<&Tensor>) -> Result<Tensor> {
        let (batch, length, hidden) = xs.dims3()?;
        let qkv = xs
            .to_dtype(self.compute_dtype)?
            .apply(&self.qkv)?
            .reshape((batch, length, 3, self.heads, self.head_size))?
            .permute((2, 0, 3, 1, 4))?;
        let q = qkv.get(0)?;
        let k = qkv.get(1)?;
        let v = qkv.get(2)?;
        let (q, k) = self.rotary.apply(&q, &k)?;
        let scale = (self.head_size as f64).powf(-0.5);
        let scores = (q * scale)?.matmul(&k.transpose(D::Minus2, D::Minus1)?)?;
        let scores = match mask {
            Some(mask) => scores.to_dtype(mask.dtype())?.broadcast_add(mask)?,
            None => scores,
        };
        let probabilities = attention_softmax(&scores)?;
        probabilities
            .to_dtype(v.dtype())?
            .matmul(&v)?
            .transpose(1, 2)?
            .reshape((batch, length, hidden))?
            .apply(&self.projection)
    }

    /// Exact sliding-window attention for ModernBERT's local layers.
    ///
    /// Each query gathers keys/values in `[query - window, query + window]`;
    /// out-of-range positions and padded keys are masked. This avoids the full
    /// `length × length` score matrix while matching the local attention mask.
    fn forward_local(&self, xs: &Tensor, attention_mask: &Tensor, window: usize) -> Result<Tensor> {
        const BLOCK: usize = LOCAL_WINDOW_BLOCK;
        let (batch, length, hidden) = xs.dims3()?;
        let qkv = xs
            .to_dtype(self.compute_dtype)?
            .apply(&self.qkv)?
            .reshape((batch, length, 3, self.heads, self.head_size))?
            .permute((2, 0, 3, 1, 4))?;
        let q = qkv.get(0)?;
        let k = qkv.get(1)?;
        let v = qkv.get(2)?;
        let (q, k) = self.rotary.apply(&q, &k)?;
        let scale = (self.head_size as f64).powf(-0.5);
        let q = (q * scale)?.contiguous()?;
        let device = xs.device();
        let attention_mask = attention_mask
            .reshape((batch, length))?
            .to_dtype(self.compute_dtype)?;

        let mut outputs = Vec::new();
        let mut start = 0;
        while start < length {
            let query_len = (length - start).min(BLOCK);
            let key_start = start.saturating_sub(window);
            let key_end = (start + query_len + window).min(length);
            let key_len = key_end - key_start;

            let q_block = q.narrow(2, start, query_len)?.contiguous()?;
            let k_block = k.narrow(2, key_start, key_len)?.contiguous()?;
            let v_block = v.narrow(2, key_start, key_len)?.contiguous()?;
            let scores = q_block.matmul(&k_block.transpose(2, 3)?)?;

            let mut valid = vec![0f32; query_len * key_len];
            for row in 0..query_len {
                let query_position = start + row;
                for column in 0..key_len {
                    let key_position = key_start + column;
                    if query_position.abs_diff(key_position) <= window {
                        valid[row * key_len + column] = 1.0;
                    }
                }
            }
            let window_valid = Tensor::from_vec(valid, (query_len, key_len), device)?
                .to_dtype(self.compute_dtype)?
                .unsqueeze(0)?
                .broadcast_as((batch, query_len, key_len))?;
            let key_valid = attention_mask
                .narrow(1, key_start, key_len)?
                .unsqueeze(1)?
                .broadcast_as((batch, query_len, key_len))?;
            let valid = (key_valid * window_valid)?;
            let bias = ((1.0 - valid)? * ATTENTION_MASK_VALUE as f64)?
                .to_dtype(self.compute_dtype)?
                .unsqueeze(1)?;
            let probabilities = attention_softmax(&scores.broadcast_add(&bias)?)?;
            outputs.push(probabilities.matmul(&v_block)?);
            start += query_len;
        }

        Tensor::cat(&outputs, 2)?
            .transpose(1, 2)?
            .reshape((batch, length, hidden))?
            .apply(&self.projection)
    }
}

const LOCAL_WINDOW_BLOCK: usize = 128;
const LOCAL_WINDOW_MIN_LENGTH: usize = 512;
const LOCAL_WINDOW_MAX_ELEMENTS: usize = 512 * 1024 * 1024;

fn attention_softmax(scores: &Tensor) -> Result<Tensor> {
    if matches!(scores.dtype(), DType::F16 | DType::BF16) {
        softmax(&scores.to_dtype(DType::F32)?, D::Minus1)?.to_dtype(scores.dtype())
    } else {
        softmax(scores, D::Minus1)
    }
}

struct Mlp {
    input: Linear,
    output: Linear,
    compute_dtype: DType,
}

impl Mlp {
    fn load(vb: VarBuilder, config: &ModernBertConfig, compute_dtype: DType) -> Result<Self> {
        Ok(Self {
            input: linear_no_bias_dtype(
                config.hidden_size,
                config.intermediate_size * 2,
                vb.pp("Wi"),
                compute_dtype,
            )?,
            output: linear_no_bias_dtype(
                config.intermediate_size,
                config.hidden_size,
                vb.pp("Wo"),
                compute_dtype,
            )?,
            compute_dtype,
        })
    }
}

impl Module for Mlp {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let parts = xs
            .to_dtype(self.compute_dtype)?
            .apply(&self.input)?
            .chunk(2, D::Minus1)?;
        (&parts[0].gelu_erf()? * &parts[1])?.apply(&self.output)
    }
}

struct Layer {
    attention: Attention,
    mlp: Mlp,
    attention_norm: Option<LayerNorm>,
    mlp_norm: LayerNorm,
    uses_local_attention: bool,
}

impl Layer {
    fn load(
        vb: VarBuilder,
        config: &ModernBertConfig,
        rotary: Arc<RotaryEmbedding>,
        uses_local_attention: bool,
        compute_dtype: DType,
    ) -> Result<Self> {
        Ok(Self {
            attention: Attention::load(vb.pp("attn"), config, rotary, compute_dtype)?,
            mlp: Mlp::load(vb.pp("mlp"), config, compute_dtype)?,
            attention_norm: layer_norm_no_bias(
                config.hidden_size,
                config.layer_norm_eps(),
                vb.pp("attn_norm"),
            )
            .ok(),
            mlp_norm: layer_norm_no_bias(
                config.hidden_size,
                config.layer_norm_eps(),
                vb.pp("mlp_norm"),
            )?,
            uses_local_attention,
        })
    }

    fn forward(
        &self,
        xs: &Tensor,
        global_mask: Option<&Tensor>,
        local_mask: &Tensor,
        attention_mask: &Tensor,
        local_window: Option<usize>,
    ) -> Result<Tensor> {
        let normalized = match &self.attention_norm {
            Some(norm) => xs.apply(norm)?,
            None => xs.clone(),
        };
        let mask = if self.uses_local_attention {
            Some(match global_mask {
                Some(global_mask) => global_mask.broadcast_add(local_mask)?,
                None => local_mask.clone(),
            })
        } else {
            global_mask.cloned()
        };
        let (batch, length, _) = xs.dims3()?;
        let window = local_window.filter(|&window| {
            let block = length.min(LOCAL_WINDOW_BLOCK);
            let span = block.saturating_add(window.saturating_mul(2));
            self.uses_local_attention
                && xs.device().is_cuda()
                && length >= LOCAL_WINDOW_MIN_LENGTH
                && length > span
                && batch
                    .saturating_mul(self.attention.heads)
                    .saturating_mul(block)
                    .saturating_mul(span)
                    .saturating_mul(self.attention.head_size)
                    <= LOCAL_WINDOW_MAX_ELEMENTS
        });
        let attention = match window {
            Some(window) => self
                .attention
                .forward_local(&normalized, attention_mask, window)?,
            None => self.attention.forward(&normalized, mask.as_ref())?,
        }
        .to_dtype(xs.dtype())?;
        let xs = (attention + xs)?;
        let mlp = xs.apply(&self.mlp_norm)?.apply(&self.mlp)?;
        let dtype = xs.dtype();
        xs + mlp.to_dtype(dtype)?
    }
}

pub struct Encoder {
    embeddings: Embedding,
    norm: LayerNorm,
    layers: Vec<Layer>,
    final_norm: LayerNorm,
    local_attention_size: usize,
    use_local_window: bool,
    dtype: DType,
    local_masks: Mutex<HashMap<usize, Tensor>>,
}

impl Encoder {
    /// Load the encoder from `vb`.
    ///
    /// `path_prefix` is the Hugging Face module prefix, e.g. `model` for
    /// Laya/NLI or `model.encoder_model` for GLiClass.
    pub fn load(
        vb: VarBuilder,
        config: &ModernBertConfig,
        compute_dtype: DType,
        use_local_window: bool,
        path_prefix: &str,
    ) -> Result<Self> {
        let global_rotary = Arc::new(RotaryEmbedding::new(
            vb.dtype(),
            config,
            config.global_rope_theta(),
            vb.device(),
        )?);
        let local_rotary = Arc::new(RotaryEmbedding::new(
            vb.dtype(),
            config,
            config.local_rope_theta(),
            vb.device(),
        )?);
        let path = |suffix: &str| {
            if path_prefix.is_empty() {
                suffix.to_owned()
            } else {
                format!("{path_prefix}.{suffix}")
            }
        };
        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for index in 0..config.num_hidden_layers {
            let local = index % config.global_attn_every_n_layers != 0;
            layers.push(Layer::load(
                vb.pp(path(&format!("layers.{index}"))),
                config,
                if local {
                    local_rotary.clone()
                } else {
                    global_rotary.clone()
                },
                local,
                compute_dtype,
            )?);
        }
        let eps = config.layer_norm_eps();
        Ok(Self {
            embeddings: embedding(
                config.vocab_size,
                config.hidden_size,
                vb.pp(path("embeddings.tok_embeddings")),
            )?,
            norm: layer_norm_no_bias(config.hidden_size, eps, vb.pp(path("embeddings.norm")))?,
            layers,
            final_norm: layer_norm_no_bias(config.hidden_size, eps, vb.pp(path("final_norm")))?,
            local_attention_size: config.local_attention,
            use_local_window,
            dtype: vb.dtype(),
            local_masks: Mutex::new(HashMap::new()),
        })
    }

    /// Raw token embeddings, before the embedding norm.
    pub fn token_embeddings(&self, ids: &Tensor) -> Result<Tensor> {
        ids.apply(&self.embeddings)
    }

    /// Run the encoder from already-computed token embeddings. The embedding
    /// norm is applied here, so families can add segment/type embeddings
    /// first.
    pub fn forward_embeds(
        &self,
        embeddings: Tensor,
        mask: &Tensor,
        has_padding: bool,
    ) -> Result<Tensor> {
        let length = embeddings.dim(1)?;
        let global_mask = has_padding
            .then(|| global_attention_mask(mask, length, self.dtype))
            .transpose()?;
        let local_mask = self.local_mask(length, embeddings.device())?;
        let local_window = self
            .use_local_window
            .then_some(self.local_attention_size / 2);
        let mut xs = embeddings.apply(&self.norm)?;
        for layer in &self.layers {
            xs = layer.forward(&xs, global_mask.as_ref(), &local_mask, mask, local_window)?;
        }
        xs.apply(&self.final_norm)
    }

    pub fn forward(&self, ids: &Tensor, mask: &Tensor, has_padding: bool) -> Result<Tensor> {
        self.forward_embeds(self.token_embeddings(ids)?, mask, has_padding)
    }

    fn local_mask(&self, length: usize, device: &Device) -> Result<Tensor> {
        let mut masks = self
            .local_masks
            .lock()
            .map_err(|_| candle_core::Error::Msg("local attention mask cache poisoned".into()))?;
        if let Some(mask) = masks.get(&length) {
            return Ok(mask.clone());
        }
        let mask = local_attention_mask(length, self.local_attention_size / 2, self.dtype, device)?;
        masks.insert(length, mask.clone());
        Ok(mask)
    }
}

fn linear_no_bias_dtype(
    input: usize,
    output: usize,
    vb: VarBuilder,
    dtype: DType,
) -> Result<Linear> {
    Ok(Linear::new(
        vb.get((output, input), "weight")?.to_dtype(dtype)?,
        None,
    ))
}

fn global_attention_mask(mask: &Tensor, target_length: usize, dtype: DType) -> Result<Tensor> {
    let (batch, source_length) = mask.dims2()?;
    let expanded = mask
        .unsqueeze(1)?
        .unsqueeze(2)?
        .expand((batch, 1, target_length, source_length))?
        .to_dtype(dtype)?;
    ((1.0 - expanded)? * ATTENTION_MASK_VALUE as f64)?.to_dtype(dtype)
}

fn local_attention_mask(
    length: usize,
    max_distance: usize,
    dtype: DType,
    device: &Device,
) -> Result<Tensor> {
    let mask: Vec<_> = (0..length)
        .flat_map(|left| {
            (0..length).map(move |right| {
                if left.abs_diff(right) > max_distance {
                    ATTENTION_MASK_VALUE
                } else {
                    0.0
                }
            })
        })
        .collect();
    Tensor::from_vec(mask, (length, length), device)?.to_dtype(dtype)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn local_attention_mask_only_exposes_nearby_tokens() -> Result<()> {
        let mask = local_attention_mask(4, 1, DType::F32, &Device::Cpu)?.to_vec2::<f32>()?;

        assert_eq!(
            mask,
            vec![
                vec![0.0, 0.0, ATTENTION_MASK_VALUE, ATTENTION_MASK_VALUE],
                vec![0.0, 0.0, 0.0, ATTENTION_MASK_VALUE],
                vec![ATTENTION_MASK_VALUE, 0.0, 0.0, 0.0],
                vec![ATTENTION_MASK_VALUE, ATTENTION_MASK_VALUE, 0.0, 0.0],
            ]
        );
        Ok(())
    }

    #[test]
    fn global_attention_mask_hides_padding_tokens() -> Result<()> {
        let mask = Tensor::new(&[[1u32, 1, 0]], &Device::Cpu)?;
        let mask = global_attention_mask(&mask, 2, DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()?;

        assert_eq!(
            mask,
            vec![
                0.0,
                0.0,
                ATTENTION_MASK_VALUE,
                0.0,
                0.0,
                ATTENTION_MASK_VALUE
            ]
        );
        Ok(())
    }
}
