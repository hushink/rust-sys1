# sys1-hub

[简体中文](README.zh-CN.md)

Model sources, manifest resolution, and the on-disk cache for rust-sys1.

## Sources

| Scheme | Example |
|---|---|
| Local path | `local:/models/laya-multilingual` or an existing path |
| Hugging Face | `hf:convaiinnovations/laya@main#multilingual` |
| ModelScope | `modelscope:owner/name@master` |
| Static HTTP center | `https://models.example.com/laya/model.json` |

Syntax: `scheme:location[@revision][#subfolder]`.

## Manifest

```json
{
  "schema_version": 1,
  "name": "laya-multilingual",
  "description": "Laya multilingual decision model",
  "release_date": "2026-01-01",
  "family": "laya",
  "variant": "multilingual",
  "backend": "candle",
  "format": "safetensors",
  "version": "main",
  "files": [
    {"path": "model.safetensors", "url": "...", "sha256": "...", "size": 643835514}
  ]
}
```

`sha256` is optional; HF non-LFS files fall back to `git_sha1`. Paths must be
relative, use forward slashes, and contain no `..`.

## Example

```rust
use sys1_hub::{Hub, HubConfig};

let hub = Hub::new(HubConfig::from_env())?;
let pulled = hub.pull("hf:convaiinnovations/laya@main#multilingual")?;
println!("{} at {}", pulled.manifest.name, pulled.directory.display());
```

`pull` returns a `PulledModel` with the manifest, directory, and per-file paths.
`resolve` returns a manifest without downloading files.

## Cache and network behavior

- Cache keys are stable, filesystem-safe, and hashed.
- Downloads use `.partial` files, HTTP Range resume, and sha256 verification.
- Concurrent pulls of the same model are serialized in-process.
- Offline mode reads cached manifests and verifies existing files.
- `HF_ENDPOINT` supports mirrors (`https://hf-mirror.com`).
- `MODELSCOPE_API_TOKEN`, `HF_TOKEN`, and `SYS1_HUB_TOKEN` are respected.
- The ModelScope file listing handles its 3000-entry silent truncation by splitting directories.

Custom centers implement `ModelSource` and are passed to `Hub::with_sources`.

## Test

```bash
cargo test --offline -p sys1-hub
```
