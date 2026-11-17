# sys1-qwen

[English](README.md)

rust-sys1 多个模型族共用的 Candle Qwen3 / Qwen3.5 解码器骨干。只做推理，
返回最终归一化后的隐藏状态；各模型族自行读取选项位置并应用头部。

## 骨干

- **Qwen3**：RMSNorm、GQA、逐 head q/k RMSNorm、RoPE、SwiGLU 的 dense decoder。
- **Qwen3.5**：Gated DeltaNet 层（depthwise 因果卷积、分块 gated delta rule、
  gated RMSNorm）与带 output gate、部分 RoPE 的 full-attention 层交替，后接
  SwiGLU。输入长度任意：内部右填充到 64 token 的整数倍，计算后切回原长度。
  3–4 tap 的 depthwise 卷积用移位乘加实现；candle 通用 `conv1d` 在这种形状下
  的 CUDA 性能约慢两个数量级。

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

## 权重

`WeightStore` 支持 memory-map 一个或多个 safetensors 分片、通过 candle 受限
pickle reader 读取 PyTorch zip checkpoint，或直接接收内存中的 tensor map。
`LoraAdapter` 按 `layers.N...` 后缀匹配，把 PEFT adapter 惰性合并进线性层
（`W + scaling · B A`）。

Qwen3.5 同时支持 tied / untied LM head；`output_embedding()` 优先返回
`lm_head.weight`，否则返回 token embedding。

## 测试

```bash
cargo test --offline -p sys1-qwen
$env:SYS1_QWEN3_TEST_DIR="C:/models/Qwen3Guard-Gen-0.6B"
cargo test --offline -p sys1-qwen --test qwen3_forward -- --ignored --nocapture
```
