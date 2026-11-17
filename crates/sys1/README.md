# sys1

[简体中文](README.zh-CN.md)

Umbrella crate for the rust-sys1 in-process API.

It re-exports the core traits, protocol types, Laya model, and hub behind
feature flags:

```toml
sys1 = { path = "../sys1" }
```

Default features: `laya`, `hub`. Optional accelerator features: `cuda`, `cudnn`,
`metal`, `accelerate`.

## Example

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

## Re-exports

- `sys1::core` — all of `sys1-core`
- `sys1::protocol` — all of `sys1-protocol`
- `sys1::hub` — all of `sys1-hub` (feature `hub`)
- `sys1::laya` — all of `sys1-laya` (feature `laya`)
- Flat aliases: `DecisionModel`, `ModelResolver`, `ModelRegistry`, `Questions`,
  `Answers`, `State`, `Question`, `Answer`, `LayaModel`, `LayaOptions`,
  `LayaFactory`, `Hub`, `HubConfig`.

## Test

```bash
cargo test --offline -p sys1
```
