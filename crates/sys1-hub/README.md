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

## Cache quota

Set `HubConfig::max_cache_bytes` (`None` or `0` = unlimited). Before materializing
a remote model, `Hub::pull` estimates the missing bytes and evicts
least-recently-used, non-resident cache directories until the new files fit.

- LRU order uses the `model.json` modification time, refreshed on every
  successful pull (including cache hits).
- Install `CacheResidency` with `Hub::set_residency` to protect models that the
  process currently has loaded. `InferenceEngine` and `sys1-cli` install
  registry-backed implementations automatically.
- Active pulls and the entry being pulled are never evicted; `remove` /
  `remove_cached` return `HubError::CacheBusy` while a download is active and
  `HubError::CacheResident` when the residency provider reports the model loaded.
- A model larger than the quota, or space blocked by resident entries, returns
  `HubError::CacheQuotaExceeded` with `limit`, `used`, `additional`, and
  `reclaimable` bytes.
- `cache_entries()` reports actual directory sizes, timestamps, and
  `evictable`; `cache_usage()` returns `limit_bytes`, `used_bytes`,
  `free_bytes`, `reclaimable_bytes`, and counts. `enforce_quota(additional)`
  applies the policy without a pull.
- Local `local:` sources are not copied into the cache and do not consume
  quota.

## Test

```bash
cargo test --offline -p sys1-hub
```

Integration tests start loopback model centers and cover:

- static HTTP manifests, relative URLs, Bearer tokens, Range-resumed downloads;
- Hugging Face tree pagination (`Link` headers), subfolders, LFS sha256,
  revision/path URL encoding, release dates, and token headers;
- ModelScope `Root`/`Recursive` listing, token + cookie headers, and the
  3000-entry recursive listing split;
- quota eviction by LRU, resident protection, oversized-model rejection, and
  eviction triggered by a pull.
