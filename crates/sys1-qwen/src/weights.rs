use candle_core::{DType, Device, Result, Tensor};
use candle_nn::{Embedding, Linear, VarBuilder, embedding};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

/// A checkpoint plus an optional PEFT LoRA adapter.
pub struct WeightStore {
    vb: VarBuilder<'static>,
    dtype: DType,
    device: Device,
    lora: Option<LoraAdapter>,
}

impl WeightStore {
    /// Memory-map one or more safetensors files.
    #[allow(unsafe_code)]
    pub fn from_safetensors(paths: &[PathBuf], dtype: DType, device: &Device) -> Result<Self> {
        if paths.is_empty() {
            candle_core::bail!("no safetensors files to load");
        }
        // The mmap owns the mapping, so the builder does not borrow `paths`.
        let vb: VarBuilder<'static> =
            unsafe { VarBuilder::from_mmaped_safetensors(paths, dtype, device)? };
        Ok(Self {
            vb,
            dtype,
            device: device.clone(),
            lora: None,
        })
    }

    /// Load a PyTorch zip checkpoint through candle's restricted pickle reader.
    pub fn from_pth(path: &Path, dtype: DType, device: &Device) -> Result<Self> {
        let vb: VarBuilder<'static> = VarBuilder::from_pth(path, dtype, device)?;
        Ok(Self {
            vb,
            dtype,
            device: device.clone(),
            lora: None,
        })
    }

    /// Load a nested state dict from a PyTorch zip checkpoint.
    pub fn from_pth_with_state(
        path: &Path,
        dtype: DType,
        device: &Device,
        key: &str,
    ) -> Result<Self> {
        let vb: VarBuilder<'static> = VarBuilder::from_pth_with_state(path, dtype, key, device)?;
        Ok(Self {
            vb,
            dtype,
            device: device.clone(),
            lora: None,
        })
    }

    /// Build from an in-memory tensor map.
    pub fn from_tensors(tensors: HashMap<String, Tensor>, dtype: DType, device: &Device) -> Self {
        Self {
            vb: VarBuilder::from_tensors(tensors, dtype, device),
            dtype,
            device: device.clone(),
            lora: None,
        }
    }

    pub fn with_lora(mut self, lora: LoraAdapter) -> Self {
        self.lora = Some(lora);
        self
    }

    pub fn dtype(&self) -> DType {
        self.dtype
    }

    pub fn var_builder(&self) -> VarBuilder<'static> {
        self.vb.clone()
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    pub fn contains(&self, name: &str) -> bool {
        self.vb.contains_tensor(name)
    }

    pub fn get(&self, name: &str, shape: &[usize]) -> Result<Tensor> {
        self.vb
            .get(shape, name)
            .map_err(|error| candle_core::Error::Msg(format!("{name}: {error}")))?
            .to_dtype(self.dtype)
    }

    pub fn get_unchecked(&self, name: &str) -> Result<Tensor> {
        self.vb.get_unchecked(name)?.to_dtype(self.dtype)
    }

    pub fn linear(
        &self,
        name: &str,
        output_dim: usize,
        input_dim: usize,
        bias: bool,
    ) -> Result<Linear> {
        let weight = self.get(&format!("{name}.weight"), &[output_dim, input_dim])?;
        let weight = match &self.lora {
            Some(lora) => lora.merge(name, weight)?,
            None => weight,
        };
        let bias = if bias && self.contains(&format!("{name}.bias")) {
            Some(self.get(&format!("{name}.bias"), &[output_dim])?)
        } else if bias {
            // Some exported heads keep a zero bias implicitly.
            Some(Tensor::zeros(output_dim, self.dtype, &self.device)?)
        } else {
            None
        };
        Ok(Linear::new(weight, bias))
    }

    pub fn embedding(&self, name: &str, vocab: usize, hidden: usize) -> Result<Embedding> {
        embedding(vocab, hidden, self.vb.pp(name))
    }

    /// The raw embedding tensor, for tied LM heads and lexical readouts.
    pub fn embedding_tensor(&self, name: &str, vocab: usize, hidden: usize) -> Result<Tensor> {
        self.get(name, &[vocab, hidden])
    }

    /// The model's output embedding matrix: `lm_head` when untied, otherwise
    /// the token embedding.
    pub fn output_embedding(
        &self,
        has_lm_head: bool,
        prefix: &str,
        vocab: usize,
        hidden: usize,
    ) -> Result<Tensor> {
        if has_lm_head && self.contains(&format!("{prefix}lm_head.weight")) {
            self.get(&format!("{prefix}lm_head.weight"), &[vocab, hidden])
        } else if self.contains("lm_head.weight") {
            self.get("lm_head.weight", &[vocab, hidden])
        } else {
            self.embedding_tensor(&format!("{prefix}embed_tokens.weight"), vocab, hidden)
        }
    }

    /// A plain linear without a bias, used by several exported heads.
    pub fn linear_no_bias(
        &self,
        name: &str,
        output_dim: usize,
        input_dim: usize,
    ) -> Result<Linear> {
        self.linear(name, output_dim, input_dim, false)
    }
}

/// One PEFT LoRA adapter, merged lazily into the base linear weights.
pub struct LoraAdapter {
    a: HashMap<String, Tensor>,
    b: HashMap<String, Tensor>,
    scaling: f32,
    dtype: DType,
    device: Device,
}

impl LoraAdapter {
    /// Load `adapter_model.safetensors`; `alpha / rank` is the merge scaling.
    pub fn load(path: &Path, scaling: f32, dtype: DType, device: &Device) -> Result<Self> {
        let tensors = candle_core::safetensors::load(path, device)?;
        let mut a = HashMap::new();
        let mut b = HashMap::new();
        for (name, tensor) in tensors {
            if let Some(module) = adapter_module(&name, "lora_A") {
                a.insert(module, tensor.to_dtype(dtype)?);
            } else if let Some(module) = adapter_module(&name, "lora_B") {
                b.insert(module, tensor.to_dtype(dtype)?);
            }
        }
        Ok(Self {
            a,
            b,
            scaling,
            dtype,
            device: device.clone(),
        })
    }

    pub fn scaling(&self) -> f32 {
        self.scaling
    }

    pub fn len(&self) -> usize {
        self.a.len()
    }

    pub fn is_empty(&self) -> bool {
        self.a.is_empty()
    }

    /// `W + scaling * B A`, computed in float32 as PEFT's merge does.
    pub fn merge(&self, base_name: &str, weight: Tensor) -> Result<Tensor> {
        let Some(module) = adapt_module_name(base_name) else {
            return Ok(weight);
        };
        let (Some(a), Some(b)) = (self.a.get(&module), self.b.get(&module)) else {
            return Ok(weight);
        };
        let delta = b
            .to_dtype(DType::F32)?
            .matmul(&a.to_dtype(DType::F32)?)?
            .affine(f64::from(self.scaling), 0.0)?
            .to_dtype(self.dtype)?;
        weight.add(&delta)
    }

    pub fn device(&self) -> &Device {
        &self.device
    }
}

/// Strip PEFT's prefixes and the projection suffix, keeping `layers.N...`.
fn adapter_module(name: &str, which: &str) -> Option<String> {
    let stem = name
        .strip_suffix(&format!(".{which}.weight"))
        .or_else(|| name.strip_suffix(&format!(".{which}.default.weight")))?;
    adapt_module_name(stem)
}

fn adapt_module_name(name: &str) -> Option<String> {
    let stem = name
        .strip_suffix(".weight")
        .or_else(|| name.strip_suffix(".bias"))
        .unwrap_or(name);
    let start = stem.find("layers.")?;
    Some(stem[start..].to_owned())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalizes_peft_names() {
        assert_eq!(
            adapter_module(
                "base_model.model.model.language_model.layers.3.self_attn.q_proj.lora_A.weight",
                "lora_A"
            )
            .as_deref(),
            Some("layers.3.self_attn.q_proj")
        );
        assert_eq!(
            adapt_module_name("model.language_model.layers.0.mlp.down_proj.weight").as_deref(),
            Some("layers.0.mlp.down_proj")
        );
        assert!(adapter_module("base_model.model.lm_head.lora_A.weight", "lora_A").is_none());
    }
}
