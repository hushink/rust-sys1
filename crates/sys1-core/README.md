# sys1-core

[简体中文](README.zh-CN.md)

Core traits, manifest types, model registry, typed in-process API, and device
selection for rust-sys1.

## What is included

- `DecisionModel`: synchronous `predict(&State, &Questions) -> DecisionOutput`.
- `ModelFactory`: lazy loading from a `ModelManifest`.
- `ModelRegistry`: name/alias resolution, lazy loading, LRU, `keep_alive`, model unload.
- `Questions` / `Answers`: typed builders and typed accessors.
- `ModelManifest` / `ModelFile`: the model bundle contract.
- `DevicePreference` / `ResolvedDevice`: runtime device selection helpers.

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

## Test

```bash
cargo test --offline -p sys1-core
```
