# sys1-modernbert

[简体中文](README.zh-CN.md)

Shared candle ModernBERT encoder used by rust-sys1 model families (`sys1-laya`,
`sys1-nli`, `sys1-gliclass`). It is not a decision model by itself.

## What it provides

- `ModernBertConfig`: parses Hugging Face `config.json`, including both the
  nested `rope_parameters` form used by Laya and the flat
  `global_rope_theta` / `local_rope_theta` form used by ModernBERT checkpoints.
- `Encoder`: embedding, local/global RoPE attention, gated MLP, and final norm.
- `Encoder::token_embeddings`: raw token embeddings before the embedding norm,
  for families that add segment/type embeddings (`sys1-gliclass`).
- `Encoder::forward_embeds`: runs the encoder from pre-computed embeddings.
- `Encoder::forward`: the usual `ids -> hidden states` path.

The encoder takes a `path_prefix` so one implementation can read both
`model.*` (Laya/NLI) and `model.encoder_model.*` (GLiClass) checkpoints.

## Use as a library

```toml
[dependencies]
sys1-modernbert = { git = "https://github.com/hushink/rust-sys1" }
candle-core = "0.11"
candle-nn = "0.11"
# Or from a checkout:
# sys1-modernbert = { path = "../rust-sys1/crates/sys1-modernbert" }
```

This crate is an encoder building block, not a `DecisionModel`; most users
should depend on a family crate or on `sys1`.

## Usage

```rust
use candle_core::{DType, Device};
use candle_nn::VarBuilder;
use sys1_modernbert::{Encoder, ModernBertConfig};

let config = ModernBertConfig::load(&directory.join("config.json"))?;
let vb = VarBuilder::from_tensors(
    candle_core::safetensors::load(&directory.join("model.safetensors"), &device)?,
    dtype,
    &device,
);
let encoder = Encoder::load(vb, &config, DType::F32, false, "model")?;
let hidden = encoder.forward(&input_ids, &attention_mask, true)?;
```

`has_padding = true` enables the key-padding mask. The local-window fast path is
an optimization; standard eager attention remains numerically equivalent.
