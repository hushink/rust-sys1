# sys1

[English](README.md)

rust-sys1 进程内 API 的 umbrella crate。

按 feature 重新导出核心 trait、协议类型与 hub，并提供把全部核心组件串起来的
`InferenceEngine`。

```toml
sys1 = { path = "../sys1" }
```

默认 feature：`laya`、`hub`。可选加速 feature：`cuda`、`cudnn`、`metal`、
`accelerate`。`InferenceEngine` 需要同时启用 `laya` 与 `hub`（即默认集合）。

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

## 进程内引擎

- `EngineOptions` 按 `hub`（缓存、端点、凭据）、`registry`（LRU/keep-alive）与
  `model`（`ModelOptions`：device、dtype、上下文长度、confidence、attention）分组。
- `InferenceEngine::new` 创建空目录；`from_reference` / `from_directory` 立即注册模型。
- `add` / `add_as` 通过 hub 拉取引用、物化 checkpoint、注册后端并写入注册表。
  仓库根目录会注册 router 与各变体。
- `load` / `unload` / `loaded_models` / `is_loaded` 控制常驻；`models` / `model`
  查看目录；`remove` 卸载并移除。
- `predict` 返回 typed `DecisionOutput`；`systemone` 返回官方
  `SystemOneResponse`；`predict_batch` 调用模型原生 batch。
- Engine 实现了 `ModelResolver`，可传给 `sys1-server` 的
  `router_with_resolver`，支持 `/api/models` 运行时管理。

更底层的接口仍可通过 `sys1::core`、`sys1::protocol`、`sys1::hub`、`sys1::laya`
使用。

## 重新导出

- `sys1::core` — `sys1-core` 全部
- `sys1::protocol` — `sys1-protocol` 全部
- `sys1::hub` — `sys1-hub` 全部（feature `hub`）
- `sys1::laya` — `sys1-laya` 全部（feature `laya`）
- 扁平别名：`DecisionModel`、`ModelResolver`、`ModelRegistry`、`ModelManifest`、
  `Questions`、`Answers`、`State`、`Question`、`Answer`、`EngineOptions`、
  `EngineError`、`InferenceEngine`、`ModelOptions`、`ConfidenceMode`、
  `LayaModel`、`LayaOptions`、`LayaFactory`、`LayaRouter`、`Hub`、`HubConfig`。

Engine 是增量能力：原有的 `LayaModel` / `LayaOptions` 直接 API 继续可用，
Laya 类型也可通过 `sys1::laya` 访问。

## 测试

```bash
cargo test --offline -p sys1
```

真实 checkpoint 的 engine 测试默认忽略：

```powershell
$env:SYS1_LAYA_TEST_ROOT="C:/path/to/laya-multilingual"
cargo test --offline -p sys1 --test engine -- --ignored --nocapture
```
