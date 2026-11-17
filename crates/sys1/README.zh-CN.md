# sys1

[English](README.md)

rust-sys1 进程内 API 的 umbrella crate。

按 feature 重新导出核心 trait、协议类型与 hub，并提供把全部核心组件串起来的
`InferenceEngine`。

```toml
sys1 = { path = "../sys1" }
```

默认 feature：`laya`、`hub`、`families`。`families` 增加 Laya/NLI/GLiClass
统一分发。可选后端/加速 feature：`onnx`、`cuda`、`cudnn`、`metal`、`accelerate`，
以及 ONNX Runtime 执行提供器 `ort-cuda`、`ort-tensorrt`、`ort-directml`、
`ort-coreml`。`InferenceEngine` 需要同时启用 `laya`、`hub` 与 `families`
（即默认集合）。

## 示例

```rust
use std::path::Path;
use sys1::{EngineOptions, InferenceEngine, Questions, State};

let engine = InferenceEngine::from_directory(
    Path::new("/path/to/laya-multilingual"),
    EngineOptions::default(),
)?;
let state = State::text("I was charged twice. Please refund me today.");
let questions = Questions::new()
    .choice("tone", [("angry", "Hostile"), ("calm", "Neutral")])
    .noul("billing", "Is this about billing?");

let output = engine.predict("laya-multilingual", &state, &questions)?;
let tone = output.answers.choice("tone")?;
let billing = output.answers.noul("billing")?;
# Ok::<(), Box<dyn std::error::Error>>(())
```

`InferenceEngine` 也接受 NLI 与 GLiClass 目录，模型族会从 checkpoint 自动识别；
`predict` 的模型参数使用 `models()` 返回的目录名或目录名称。

原来的直接 checkpoint API 保持不变，可与 engine 并存：

```rust
use std::path::Path;
use sys1::{DecisionModel, LayaModel, LayaOptions, Questions, State};

let model = LayaModel::load_dir(Path::new("/path/to/laya"), LayaOptions::default())?;
let state = State::text("I was charged twice. Please refund me today.");
let questions = Questions::new()
    .choice("tone", [("angry", "Hostile"), ("calm", "Neutral")])
    .noul("billing", "Is this about billing?");

let output = model.predict(&state, &questions)?;
let tone = output.answers.choice("tone")?;
let billing = output.answers.noul("billing")?;
# Ok::<(), Box<dyn std::error::Error>>(())
```

启用 `features = ["onnx"]` 后，可直接加载 ONNX 导出，也可通过
`InferenceEngine::from_directory` 使用：

```rust
use std::path::Path;
use sys1::{DecisionModel, LayaOnnxModel, LayaOptions, Questions, State};

let model = LayaOnnxModel::load_dir(
    Path::new("/path/to/laya-onnx"),
    LayaOptions::default(),
)?;
let output = model.predict(
    &State::text("I was charged twice."),
    &Questions::new().noul("billing", "Is this about billing?"),
)?;
# Ok::<(), Box<dyn std::error::Error>>(())
```

计算图契约与 `onnx.json` 选项见
[`sys1-laya`](../sys1-laya/README.zh-CN.md#onnx-runtime-后端)。`sys1::ort`
重新导出底层 `ScoringSession`，供其他模型 family 复用。

## 进程内引擎

- `EngineOptions` 按 `hub`（缓存、端点、凭据）、`registry`（LRU/keep-alive）与
  `model`（`ModelOptions`：device、dtype、上下文长度、confidence、attention）分组。
- `InferenceEngine::new` 创建空目录；`from_reference` / `from_directory` 立即注册模型。
- `add` / `add_as` 通过 hub 拉取引用、物化 checkpoint、注册后端并写入注册表。
  Laya 仓库根目录会注册 router 与各变体；NLI 与 GLiClass 目录会自动识别。
- `load` / `unload` / `loaded_models` / `is_loaded` 控制常驻；`models` / `model`
  查看目录；`remove` 卸载并移除。
- `predict` 返回 typed `DecisionOutput`；`systemone` 返回官方
  `SystemOneResponse`；`predict_batch` 调用模型原生 batch；`predict_long`
  用重叠窗口扫描长 state；`predict_with` / `predict_batch_with` 应用
  `PredictionOptions`（例如 `min_confidence`）。
  `DecisionOutput::actions` 暴露可选的逐问题动作概率（Laya 的
  `act_probability`），官方 wire 响应不包含。
  `DecisionOutput::windows` / `scored_windows` 暴露决定窗口与窗口数；
  `abstentions` 暴露可选的逐答案 gate 结果。
- Engine 实现了 `ModelResolver`，可传给 `sys1-server` 的
  `router_with_resolver`，支持 `/api/models` 运行时管理。

更底层的接口仍可通过 `sys1::core`、`sys1::protocol`、`sys1::hub`、`sys1::laya`
使用。

## 重新导出

- `sys1::core` — `sys1-core` 全部
- `sys1::protocol` — `sys1-protocol` 全部
- `sys1::hub` — `sys1-hub` 全部（feature `hub`）
- `sys1::laya` — `sys1-laya` 全部（feature `laya`）
- `sys1::families` — `sys1-families` 全部（feature `families`）
- `sys1::ort` — ONNX Runtime session 层（feature `onnx`）
- 扁平别名：`DecisionModel`、`ModelResolver`、`ModelRegistry`、`ModelManifest`、
  `Questions`、`Answers`、`State`、`Question`、`Answer`、`EngineOptions`、
  `EngineError`、`InferenceEngine`、`ModelOptions`、`ModelOrigin`、
  `ConfidenceMode`、`LayaModel`、`LayaOptions`、`LayaFactory`、`LayaRouter`、
  `LayaOnnxModel`（feature `onnx`）、`Hub`、`HubConfig`。

Engine 是增量能力：原有的 `LayaModel` / `LayaOptions` 直接 API 继续可用，
Laya 类型也可通过 `sys1::laya` 访问。

## 基准

Engine 只增加注册表查找与句柄管理开销：

```bash
cargo run --release -p sys1 --example engine_overhead -- /path/to/laya-multilingual 30
```

CPU F32 multilingual、30 次交替测量：直接 checkpoint 推理约 121.7 ms 均值 /
119.9 ms p50；engine 约 122.6 ms / 120.5 ms，即约 **0.9 ms 均值、0.65 ms p50
（<1%）**。无需改为 `Arc<ModelManifest>`。

## 测试

```bash
cargo test --offline -p sys1
```

`tests/naming.rs` 会检查除 `sys1-laya` 外的 crate 不定义 Laya 前缀项；
`sys1` 根部的兼容再导出允许保留。真实 checkpoint 的 engine 测试默认忽略：

```powershell
$env:SYS1_LAYA_TEST_ROOT="C:/path/to/laya-multilingual"
cargo test --offline -p sys1 --test engine -- --ignored --nocapture
```
