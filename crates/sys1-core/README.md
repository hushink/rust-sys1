# sys1-core

[简体中文](README.zh-CN.md)

Core traits, manifest types, model registry, typed in-process API, and device
selection for rust-sys1.

## What is included

- `DecisionModel`: synchronous `predict(&State, &Questions) -> DecisionOutput`.
- `ModelFactory`: lazy loading from a `ModelManifest`.
- `ModelResolver`: optional runtime resolution of references into loadable manifests.
- `ModelRegistry`: name/alias resolution, lazy loading, LRU, `keep_alive`, model unload,
  plus runtime `register` / `unregister` of catalog entries.
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
- `register(manifest)` adds a catalog entry; duplicates and alias conflicts are
  rejected without changing the catalog. `unregister(name)` removes the entry
  (by name or alias) and drops its resident instance.
- `is_loaded(name)` reports residency without touching the LRU timestamp.
- `predict_output` returns typed `DecisionOutput`; `predict` wraps it into the
  official `SystemOneResponse` with the canonical model name.
- A `ModelResolver` is not part of the registry: callers resolve a reference to
  manifests, register them, and pass a factory that can load them. The CLI uses
  this to add hub models while the server is running.

## Test

```bash
cargo test --offline -p sys1-core
```
