# sys1-families

[English](README.md)

为已编译的 rust-sys1 模型族提供统一的 `ModelFactory`：

| 模型族 | Crate | 识别方式 |
|---|---|---|
| Laya | `sys1-laya` | `rl_agent_config.json` + `encoder/config.json`（或 ONNX 导出） |
| NLI | `sys1-nli` | 带 `entailment` 标签的 ModernBERT `config.json` |
| GLiClass | `sys1-gliclass` | `model_type = "GLiClass"` 且编码器为 ModernBERT |
| Qwen3Guard | `sys1-qwen3guard` | Qwen3 config，且 tokenizer chat template 含 `Safety: Safe` |
| Von | `sys1-von` | ModernBERT config + `option_marker.pt` |
| Decima | `sys1-decima` | `decima.json` + 编码器与 `head.safetensors` |
| Kev | `sys1-kev` | adapter + `head.pt`，Qwen3.5 base 在旁或 `base/` 下 |
| Decision | `sys1-decision` | `decision_config.json` + `decision_head.safetensors` + `backbone/` |
| Clef | `sys1-clef` | Qwen3.5 config + `joint_head.safetensors` + `joint_head_config.json` |

CLI 与 `sys1` umbrella engine 用它注册本地目录或已拉取的 Hugging Face snapshot，
调用方不需要提前知道目录属于哪个模型族。

## 用法

```rust
use sys1_families::{FamilyFactory, register_bundle};

let factory = FamilyFactory::new();
let manifests = register_bundle(
    &factory,
    &pulled.manifest,
    &pulled.directory,
    &options,
    Some("nli"),
)?;
// 把 manifests 加入 ModelRegistry，并使用 factory 作为 ModelFactory
```

`load_directory` / `load_manifest` 用于直接加载单个模型（`sys1 run`）；
`local_manifest` 为没有 `model.json` 的受支持 checkpoint 目录生成 manifest，
包含嵌套文件与分片。

## Features

- `onnx` 通过同一 dispatcher 支持 Laya ONNX 加载。

本 crate 不负责下载模型；物化由 `sys1-hub` 与 CLI 负责。
