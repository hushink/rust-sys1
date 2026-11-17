# sys1-families

[简体中文](README.zh-CN.md)

One `ModelFactory` for the compiled rust-sys1 families:

| Family | Crate | Detected by |
|---|---|---|
| Laya | `sys1-laya` | `rl_agent_config.json` + `encoder/config.json` (or ONNX export) |
| NLI | `sys1-nli` | ModernBERT `config.json` with an `entailment` label |
| GLiClass | `sys1-gliclass` | `model_type = "GLiClass"` with a ModernBERT encoder |
| Qwen3Guard | `sys1-qwen3guard` | Qwen3 config + tokenizer chat template with `Safety: Safe` |
| Von | `sys1-von` | ModernBERT config + `option_marker.pt` |
| Decima | `sys1-decima` | `decima.json` + encoder and `head.safetensors` |
| Kev | `sys1-kev` | adapter + `head.pt`; the Qwen3.5 base beside them, under `base/`, or declared in the adapter metadata |
| Decision | `sys1-decision` | `decision_config.json` + `decision_head.safetensors` + `backbone/` |
| Clef | `sys1-clef` | Qwen3.5 config + `joint_head.safetensors` + `joint_head_config.json` |

The CLI and the `sys1` umbrella engine use it to register a local directory or
a pulled Hugging Face snapshot without knowing which family it contains.

## Usage

```rust
use sys1_families::{
    BundleOptions, FamilyFactory, FamilyOptions, register_bundle,
    resolve_bundle_overrides,
};

let factory = FamilyFactory::new();
let families = FamilyOptions::default();
let overrides = resolve_bundle_overrides(&pulled.directory, &families, |reference| {
    hub.pull(reference)
        .map(|pulled| pulled.directory)
        .map_err(|error| sys1_core::ModelError::load(error.to_string()))
})?;
let bundle = BundleOptions { families, overrides };
let manifests = register_bundle(
    &factory,
    &pulled.manifest,
    &pulled.directory,
    &options,
    &bundle,
    Some("nli"),
)?;
// add `manifests` to a ModelRegistry using `factory` as the ModelFactory
```

`resolve_bundle_overrides` materializes family-specific extras that are not
part of the checkpoint directory. For Kev, it resolves the Qwen3.5 base from
`[families.kev] base_dir`, a `base/` sibling, or the reference declared in
`kev.json`, `head.pt`, `provenance.json`, or `adapter_config.json`, pulling it
through the callback when needed.

`FamilyOptions` also carries encoder batch size, per-family `max_len`, Decima
`choice_fit`, Kev/Decision temperatures, and other runtime overrides.

`load_directory` and `load_manifest` load one model directly (used by
`sys1 run`). `local_manifest` builds a manifest for a bare supported checkpoint
directory without `model.json`, including nested files and shards.

## Features

- `onnx` enables Laya ONNX loading through the same dispatcher.

The crate does not download models; `sys1-hub` and the CLI own materialization.
