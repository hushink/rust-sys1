# sys1-qwen

[简体中文](README.zh-CN.md)

Candle Qwen3 and Qwen3.5 decoder backbones shared by several rust-sys1 model
families. The crate is inference-only and returns final normed hidden states;
families read option positions and apply their own heads.

## Backbones

- **Qwen3** — dense decoder with RMSNorm, grouped-query attention, per-head
  q/k RMSNorm, RoPE and SwiGLU.
- **Qwen3.5** — hybrid decoder alternating Gated DeltaNet layers (depthwise
  causal conv, chunked gated delta rule, gated RMSNorm) and gated
  full-attention layers with partial RoPE, followed by SwiGLU. Rows may have
  any length: the backbone right-pads to 64-token chunks internally and slices
  the result back. The 3–4 tap depthwise convolution is evaluated as shifted
  multiplies; candle's generic `conv1d` is roughly two orders of magnitude
  slower on CUDA for this shape.

```rust
use candle_core::{DType, Device};
use sys1_qwen::{Qwen3, Qwen3Config, WeightStore};

let mut config = Qwen3Config::load(std::path::Path::new("model/config.json"))?;
let paths = sys1_qwen::safetensors_files(std::path::Path::new("model"))?;
let store = WeightStore::from_safetensors(&paths, DType::F32, &Device::Cpu)?;
config.prefix = sys1_qwen::detect_prefix(&store, &["model.", ""], "embed_tokens.weight")
    .expect("checkpoint prefix");
let model = Qwen3::load(&config, &store)?;
# Ok::<(), Box<dyn std::error::Error>>(())
```

## Weights

`WeightStore` memory-maps one or more safetensors shards, reads a PyTorch zip
checkpoint through candle's restricted pickle reader, or accepts an in-memory
tensor map. `LoraAdapter` merges a PEFT adapter (`W + scaling · B A`) lazily
into every adapted linear layer, matching by the `layers.N...` module suffix.

Qwen3.5 supports tied and untied LM heads. `output_embedding()` returns the
untied `lm_head.weight` when present, otherwise the token embedding.

## Test

```bash
cargo test --offline -p sys1-qwen
$env:SYS1_QWEN3_TEST_DIR="C:/models/Qwen3Guard-Gen-0.6B"
cargo test --offline -p sys1-qwen --test qwen3_forward -- --ignored --nocapture
```
