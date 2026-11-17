# sys1

[简体中文](README.zh-CN.md)

Umbrella crate for the rust-sys1 in-process API.

It re-exports the core traits, protocol types, and hub behind feature flags,
and provides `InferenceEngine`, which wires all core components together.

```toml
sys1 = { path = "../sys1" }
```

Default features: `laya`, `hub`. Optional accelerator features: `cuda`, `cudnn`,
`metal`, `accelerate`. `InferenceEngine` requires both `laya` and `hub` (the
default set).

## Example

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

## In-process engine

- `EngineOptions` groups `hub` (cache, endpoints, credentials), `registry`
  (LRU/keep-alive), and `model` (`ModelOptions`: device, dtype, context length,
  confidence, attention).
- `InferenceEngine::new` starts empty; `from_reference` and `from_directory`
  register a model immediately.
- `add` / `add_as` pull references through the hub, materialize the checkpoint,
  register the backend, and add manifests to the registry. A repository root
  registers a router plus its variants.
- `load` / `unload` / `loaded_models` / `is_loaded` control residency;
  `models` / `model` inspect the catalog; `remove` unloads and unregisters.
- `predict` returns typed `DecisionOutput`; `systemone` returns the official
  `SystemOneResponse`; `predict_batch` runs the model's native batch path.
- The engine implements `ModelResolver`, so it can be passed to
  `sys1-server`'s `router_with_resolver` for runtime `/api/models` support.

Lower-level access is still available through `sys1::core`, `sys1::protocol`,
`sys1::hub`, and `sys1::laya`.

## Re-exports

- `sys1::core` — all of `sys1-core`
- `sys1::protocol` — all of `sys1-protocol`
- `sys1::hub` — all of `sys1-hub` (feature `hub`)
- `sys1::laya` — all of `sys1-laya` (feature `laya`)
- Flat aliases: `DecisionModel`, `ModelResolver`, `ModelRegistry`,
  `ModelManifest`, `Questions`, `Answers`, `State`, `Question`, `Answer`,
  `EngineOptions`, `EngineError`, `InferenceEngine`, `ModelOptions`,
  `ConfidenceMode`, `Hub`, `HubConfig`.

Laya-prefixed types are intentionally not exported from the crate root; use
`sys1::laya` when you need the concrete implementation.

## Test

```bash
cargo test --offline -p sys1
```

The real-checkpoint engine test is ignored by default:

```powershell
$env:SYS1_LAYA_TEST_ROOT="C:/path/to/laya-multilingual"
cargo test --offline -p sys1 --test engine -- --ignored --nocapture
```
