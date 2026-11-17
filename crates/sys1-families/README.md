# sys1-families

[简体中文](README.zh-CN.md)

One `ModelFactory` for the compiled rust-sys1 families:

| Family | Crate | Detected by |
|---|---|---|
| Laya | `sys1-laya` | `rl_agent_config.json` + `encoder/config.json` (or ONNX export) |
| NLI | `sys1-nli` | ModernBERT `config.json` with an `entailment` label |
| GLiClass | `sys1-gliclass` | `model_type = "GLiClass"` with a ModernBERT encoder |

The CLI and the `sys1` umbrella engine use it to register a local directory or
a pulled Hugging Face snapshot without knowing which family it contains.

## Usage

```rust
use sys1_families::{FamilyFactory, register_bundle};

let factory = FamilyFactory::new();
let manifests = register_bundle(
    &factory,
    &pulled.manifest,
    &pulled.directory,
    &options,
    Some("nli"),
)?;
// add `manifests` to a ModelRegistry using `factory` as the ModelFactory
```

`load_directory` and `load_manifest` load one model directly (used by
`sys1 run`). `local_manifest` builds a manifest for a bare NLI/GLiClass
directory without `model.json`.

## Features

- `onnx` enables Laya ONNX loading through the same dispatcher.

The crate does not download models; `sys1-hub` and the CLI own materialization.
