# sys1-modernbert

[English](README.md)

rust-sys1 各模型族（`sys1-laya`、`sys1-nli`、`sys1-gliclass`）共用的 candle
ModernBERT 编码器；本身不是决策模型。

## 提供内容

- `ModernBertConfig`：解析 Hugging Face `config.json`，同时支持 Laya 使用的嵌套
  `rope_parameters` 与 ModernBERT checkpoint 使用的扁平 `global_rope_theta` /
  `local_rope_theta`。
- `Encoder`：embedding、local/global RoPE attention、gated MLP 与 final norm。
- `Encoder::token_embeddings`：embedding norm 之前的原始 token 表示，供需要叠加
  segment/type embedding 的模型族使用（`sys1-gliclass`）。
- `Encoder::forward_embeds`：从已有 embedding 继续运行编码器。
- `Encoder::forward`：常规 `ids -> hidden states` 路径。

编码器接收 `path_prefix`，同一份实现可读取 `model.*`（Laya/NLI）与
`model.encoder_model.*`（GLiClass）checkpoint。

## 作为库使用

```toml
[dependencies]
sys1-modernbert = { git = "https://github.com/hushink/rust-sys1" }
candle-core = "0.11"
candle-nn = "0.11"
# 或从本地 checkout 引入：
# sys1-modernbert = { path = "../rust-sys1/crates/sys1-modernbert" }
```

本 crate 是编码器组件，不是 `DecisionModel`；大多数用户应依赖 family crate 或
`sys1`。

## 用法

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

`has_padding = true` 会启用 padding key mask。local-window 快速路径只是优化，与
标准 eager attention 数值等价。
