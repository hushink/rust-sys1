# sys1

[简体中文](README.zh-CN.md)

Umbrella crate for the rust-sys1 in-process API.

It re-exports the core traits, protocol types, and hub behind feature flags,
and provides `InferenceEngine`, which wires all core components together.

```toml
sys1 = { path = "../sys1" }
```

Default features: `laya`, `hub`, `families`. `families` adds the shared
dispatcher for Laya, NLI, GLiClass, Qwen3Guard, Von, Decima, Kev, Decision and
Clef. Optional backend/accelerator features: `onnx`,
`cuda`, `cudnn`, `metal`, `accelerate`, and the ONNX Runtime execution
providers `ort-cuda`, `ort-tensorrt`, `ort-directml`, `ort-coreml`.
`InferenceEngine` requires `laya`, `hub`, and `families` (the default set).

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

`InferenceEngine` also accepts NLI, GLiClass, Qwen3Guard, Von, Decima, Kev,
Decision and Clef directories; the family is detected from the checkpoint. Use
the catalog name returned by `models()` (or the directory name) as the `predict`
model argument.

The direct checkpoint API is unchanged and remains available alongside the
engine:

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

With `features = ["onnx"]`, an ONNX export can be loaded directly or through
`InferenceEngine::from_directory`:

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

See [`sys1-laya`](../sys1-laya/README.md#onnx-runtime-backend) for the graph
contract and `onnx.json` options. `sys1::ort` re-exports the raw
`ScoringSession` layer for other model families.

## In-process engine

- `EngineOptions` groups `hub` (cache, endpoints, credentials), `registry`
  (LRU/keep-alive), `model` (`ModelOptions`: device, dtype, context length,
  confidence, attention), and `families` (`FamilyOptions`: encoder batch size,
  per-family `max_len`, Decima `choice_fit`, Kev base/temperature, Decision
  temperature). Kev adapters without a local base pull the repository declared
  by `head.pt` / `adapter_config.json` through the engine's hub.
- `InferenceEngine::new` starts empty; `from_reference` and `from_directory`
  register a model immediately.
- `add` / `add_as` pull references through the hub, materialize the checkpoint,
  register the backend, and add manifests to the registry. A Laya repository
  root registers a router plus its variants; the other candle families are
  auto-detected from their checkpoint layouts.
- `load` / `unload` / `loaded_models` / `is_loaded` control residency;
  `models` / `model` inspect the catalog; `remove` unloads and unregisters.
- `predict` returns typed `DecisionOutput`; `systemone` returns the official
  `SystemOneResponse`; `predict_batch` runs the model's native batch path;
  `predict_long` scans a long state in overlapping windows; `predict_with` /
  `predict_batch_with` apply `PredictionOptions` such as `min_confidence`.
  `DecisionOutput::actions` exposes optional per-question action probabilities
  (Laya's `act_probability`); the official wire response omits them.
  `DecisionOutput::windows` / `scored_windows` expose the deciding token window
  and window count; `abstentions` exposes the opt-in gate result per answer.
- The engine implements `ModelResolver`, so it can be passed to
  `sys1-server`'s `router_with_resolver` for runtime `/api/models` support.

Lower-level access is still available through `sys1::core`, `sys1::protocol`,
`sys1::hub`, `sys1::laya`, and `sys1::families`.

## Re-exports

- `sys1::core` — all of `sys1-core`
- `sys1::protocol` — all of `sys1-protocol`
- `sys1::hub` — all of `sys1-hub` (feature `hub`)
- `sys1::laya` — all of `sys1-laya` (feature `laya`)
- `sys1::families` — all of `sys1-families` (feature `families`)
- `sys1::ort` — ONNX Runtime session layer (feature `onnx`)
- Flat aliases: `DecisionModel`, `ModelResolver`, `ModelRegistry`,
  `ModelManifest`, `Questions`, `Answers`, `State`, `Question`, `Answer`,
  `EngineOptions`, `EngineError`, `InferenceEngine`, `ModelOptions`,
  `ModelOrigin`, `ConfidenceMode`, `LayaModel`, `LayaOptions`, `LayaFactory`,
  `LayaRouter`, `LayaOnnxModel` (feature `onnx`), `Hub`, `HubConfig`.

The engine is additive: the existing `LayaModel` / `LayaOptions` direct API
keeps working, and Laya types are also available under `sys1::laya`.

## Benchmark

The engine adds registry lookup and handle bookkeeping only:

```bash
cargo run --release -p sys1 --example engine_overhead -- /path/to/laya-multilingual 30
```

On CPU F32 multilingual (30 interleaved iterations), direct checkpoint
prediction was ~121.7 ms mean / 119.9 ms p50; the engine was ~122.6 ms /
120.5 ms — **~0.9 ms mean, ~0.65 ms p50 (<1%)**. No `Arc<ModelManifest>`
optimization was needed.

## Test

```bash
cargo test --offline -p sys1
```

`tests/naming.rs` enforces that no crate except `sys1-laya` defines
Laya-prefixed items; compatibility re-exports at the `sys1` root are allowed.
The real-checkpoint engine test is ignored by default:

```powershell
$env:SYS1_LAYA_TEST_ROOT="C:/path/to/laya-multilingual"
cargo test --offline -p sys1 --test engine -- --ignored --nocapture
```
