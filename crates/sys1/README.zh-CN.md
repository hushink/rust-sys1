# sys1

[English](README.md)

rust-sys1 进程内 API 的 umbrella crate。

按 feature 重新导出核心 trait、协议类型、Laya 模型与 hub：

```toml
sys1 = { path = "../sys1" }
```

默认 feature：`laya`、`hub`。可选加速 feature：`cuda`、`cudnn`、`metal`、
`accelerate`。

## 示例

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

## 重新导出

- `sys1::core` — `sys1-core` 全部
- `sys1::protocol` — `sys1-protocol` 全部
- `sys1::hub` — `sys1-hub` 全部（feature `hub`）
- `sys1::laya` — `sys1-laya` 全部（feature `laya`）
- 扁平别名：`DecisionModel`、`ModelRegistry`、`Questions`、`Answers`、`State`、
  `Question`、`Answer`、`LayaModel`、`LayaOptions`、`LayaFactory`、`Hub`、`HubConfig`。

## 测试

```bash
cargo test --offline -p sys1
```
