# sys1-core

[简体中文](README.zh-CN.md)

Core traits, manifest types, model registry, typed in-process API, and device
selection for rust-sys1.

## What is included

- `DecisionModel`: synchronous `predict(&State, &Questions) -> DecisionOutput`.
  The optional `predict_long` scans a state in overlapping windows; see
  `LongContextOptions` and `WindowAttribution`.
- `ModelFactory`: lazy loading from a `ModelManifest`.
- `ModelResolver`: optional runtime resolution of references into loadable manifests.
- `ModelRegistry`: name/alias resolution, lazy loading, LRU, `keep_alive`, model unload,
  plus runtime `register` / `unregister` of catalog entries.
- `Questions` / `Answers`: typed builders and typed accessors.
- `PredictionOptions` / `MinConfidence`: opt-in abstention, scalar or keyed by
  question-type/option-count buckets.
- `Abstention` / `AbstentionState`: the gate result for each answer
  (`passed`, `abstained`, or `unevaluated`).
- `ModelManifest` / `ModelFile`: the model bundle contract.
- `DevicePreference` / `ResolvedDevice`: runtime device selection helpers.

## Use as a library

```toml
[dependencies]
sys1-core = { git = "https://github.com/hushink/rust-sys1" }
sys1-protocol = { git = "https://github.com/hushink/rust-sys1" }
# Or from a checkout:
# sys1-core = { path = "../rust-sys1/crates/sys1-core" }
# sys1-protocol = { path = "../rust-sys1/crates/sys1-protocol" }
```

`sys1-core` is the trait/registry layer; implement `DecisionModel` to plug your
own backend in, or use it with a family crate such as `sys1-nli`.

## Example

```rust
use std::sync::Arc;
use sys1_core::{
    Answers, DecisionModel, DecisionOutput, ModelError, ModelRegistry, Questions,
    RegistryConfig,
};
use sys1_protocol::{Answer, NoulAnswer, State, Usage};

struct Echo;

impl DecisionModel for Echo {
    fn predict(&self, _state: &State, questions: &Questions) -> Result<DecisionOutput, ModelError> {
        let answers = questions
            .iter()
            .map(|(id, _)| (id.clone(), Answer::Noul(NoulAnswer::new(0.5))))
            .collect();
        Ok(DecisionOutput::new(Answers::new(answers), Usage::new(1, 0)))
    }
}

let registry = ModelRegistry::new(
    vec![manifest],          // Vec<ModelManifest>
    Arc::new(factory),       // Arc<dyn ModelFactory>
    RegistryConfig::default(),
)?;

let response = registry.predict("laya-latest", &state, &questions)?;
```

## Registry semantics

- The model name is resolved through canonical names and aliases.
- Loading is lazy; a `OnceLock` per model prevents duplicate inserts.
- `max_loaded` evicts the oldest model that is not held by an external `Arc`.
- `keep_alive` removes idle models on `maintenance()` / `predict()`.
- In-use models may temporarily exceed `max_loaded`.
- `register(manifest)` adds a catalog entry as `ModelOrigin::Runtime`;
  `register_configured` marks a config-sourced addition. Duplicates and alias
  conflicts are rejected without changing the catalog. `unregister(name)`
  removes the entry (by name or alias) and drops its resident instance.
- `is_loaded(name)` reports residency without touching the LRU timestamp.
- `origin(name)` returns `ModelOrigin::Configured` for manifests passed to
  `new` or `ModelOrigin::Runtime` for `register` additions; aliases resolve to
  their canonical entry.
- `predict_output` returns typed `DecisionOutput`; `predict` wraps it into the
  official `SystemOneResponse` with the canonical model name.
- `DecisionOutput::actions` carries optional per-question action probabilities
  (for example Laya's `action.act_probability`). It is a typed extension; the
  official wire response does not include it.
- `DecisionOutput::windows` / `scored_windows` carry long-context attribution:
  the deciding token window per answer and the number of windows scored
  (`0` for `predict`, `1` for a state that fit one window). `predict_long_output`
  runs the same validation and registry path for long calls.
- `predict_with` / `predict_batch_with` apply `PredictionOptions`; with
  `min_confidence` set, `DecisionOutput::abstentions` holds one result per
  answer and the answer payload is left untouched. `predict_output_with` is the
  registry-level form.

```rust
use sys1_core::{MinConfidence, PredictionOptions};

let options = PredictionOptions {
    min_confidence: Some(MinConfidence::scalar(0.8)?),
};
let output = model.predict_with(&state, &questions, &options)?;
let gate = output.abstention("tone").expect("gate ran");
assert!(gate.low_confidence == (gate.state == sys1_core::AbstentionState::Abstained));
```
- A `ModelResolver` is not part of the registry: callers resolve a reference to
  manifests, register them, and pass a factory that can load them. The CLI uses
  this to add hub models while the server is running.

## Test

```bash
cargo test --offline -p sys1-core
```
