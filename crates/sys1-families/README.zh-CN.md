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
| Kev | `sys1-kev` | adapter + `head.pt`；Qwen3.5 base 在旁、`base/` 下，或由适配器元数据声明 |
| Decision | `sys1-decision` | `decision_config.json` + `decision_head.safetensors` + `backbone/` |
| Clef | `sys1-clef` | Qwen3.5 config + `joint_head.safetensors` + `joint_head_config.json` |

CLI 与 `sys1` umbrella engine 用它注册本地目录或已拉取的 Hugging Face snapshot，
调用方不需要提前知道目录属于哪个模型族。

## 作为库使用

```toml
[dependencies]
sys1-families = { git = "https://github.com/hushink/rust-sys1" }
sys1-core = { git = "https://github.com/hushink/rust-sys1" }
sys1-hub = { git = "https://github.com/hushink/rust-sys1" }
# 或从本地 checkout 引入：
# sys1-families = { path = "../rust-sys1/crates/sys1-families" }
```

大多数应用应使用 `sys1` 的 `InferenceEngine`，而不是直接调用分发器；本 crate
面向自定义注册表或工具。

## 用法

```rust
use sys1_families::{
    BundleOptions, FamilyFactory, FamilyOptions, register_bundle,
    resolve_bundle_overrides,
};

let factory = FamilyFactory::new();
let families = FamilyOptions::default();
let overrides = resolve_bundle_overrides(&pulled.directory, &families, |reference| {
    hub.pull(reference)
        .map(|pulled| pulled.directory)
        .map_err(|error| sys1_core::ModelError::load(error.to_string()))
})?;
let bundle = BundleOptions { families, overrides };
let manifests = register_bundle(
    &factory,
    &pulled.manifest,
    &pulled.directory,
    &options,
    &bundle,
    Some("nli"),
)?;
// 把 manifests 加入 ModelRegistry，并使用 factory 作为 ModelFactory
```

`resolve_bundle_overrides` 负责物化不属于 checkpoint 目录的模型族附加项。对 Kev
而言，它按 `[families.kev] base_dir`、相邻 `base/`，或 `kev.json`、`head.pt`、
`provenance.json`、`adapter_config.json` 中声明的引用解析 Qwen3.5 base，必要时
通过回调拉取。

`FamilyOptions` 还承载编码器 batch 大小、各模型族的 `max_len`、Decima 的
`choice_fit`、Kev/Decision 温度等运行时覆盖项。

`load_directory` / `load_manifest` 用于直接加载单个模型（`sys1 run`）；
`local_manifest` 为没有 `model.json` 的受支持 checkpoint 目录生成 manifest，
包含嵌套文件与分片。

## Features

- `onnx` 通过同一 dispatcher 支持 Laya ONNX 加载。

本 crate 不负责下载模型；物化由 `sys1-hub` 与 CLI 负责。
