# sys1-server

[简体中文](README.zh-CN.md)

Strict TypeSafe-compatible System One HTTP server built on axum.

## Endpoints

| Method | Path | Meaning |
|---|---|---|
| `POST` | `/v1/systemone` | Official decision endpoint |
| `GET` | `/v1/models` | Official model/alias list |
| `GET` | `/health` | Operational health check, outside `/v1` |

There is intentionally no `/v1/decisions`.

## Example

```rust
use std::sync::Arc;
use sys1_core::{ModelRegistry, RegistryConfig};
use sys1_server::{ServerConfig, router, serve};

let registry = Arc::new(ModelRegistry::new(
    manifests,
    factory,
    RegistryConfig::default(),
)?);

// For embedding in another axum app:
let app = router(registry.clone(), ServerConfig::default());

// Or run a standalone server:
serve("127.0.0.1:8080".parse()?, registry, ServerConfig::default()).await?;
```

## Behavior

- Bearer auth is enabled when `ServerConfig::api_key` is set.
- Request bodies are limited to 32 MiB; gzip is decompressed; zstd returns `415`.
- Responses include `x-typesafe-request-id`.
- Validation failures use `422 {"detail": [...]}`.
- Unknown models use `400 {"detail": "..."}`.
- `box` / `assets` are parsed but unsupported by Laya: `422 unsupported`.
- Missing `TypeSafe-Danger-Zone-Images: true` for box/assets returns `400`.
- Long state returns `400 max_tokens_exceeded`.
- Inference and model loading run in `tokio::task::spawn_blocking`.
- Cross-request batching is enabled by default: `ServerConfig::batching` carries
  a `BatchConfig { max_batch: 32, max_wait: 2 ms }`. Set `batching: None` to
  disable it. Models can opt out through
  `DecisionModel::supports_cross_request_batching`; CPU Laya returns `false`
  because independent CPU forwards already use the thread pool. A lone GPU
  request only pays a 0.5 ms yield grace period.

## Benchmark

i5-12400F (6C/12T) + RTX 3060 12 GiB, `laya-multilingual` (mmBERT-base, 322M),
distinct states per concurrent request:

| Scenario | CPU mean | CPU p95 | CPU req/s | CUDA F32 mean | CUDA F32 p95 | CUDA req/s |
|---|---:|---:|---:|---:|---:|---:|
| `GET /health` | 0.29 ms | 0.47 ms | 2534 | 0.31 ms | 0.54 ms | 2254 |
| `GET /v1/models` | 0.23 ms | 0.35 ms | 3879 | 0.28 ms | 0.43 ms | 3263 |
| `POST /v1/systemone` (noul) | 84.62 ms | 99.89 ms | 11.81 | 17.39 ms | 21.12 ms | 57.32 |
| `POST /v1/systemone` (noul+score+choice) | 199.72 ms | 214.56 ms | 5.01 | 21.26 ms | 24.06 ms | 46.96 |
| `POST /v1/systemone` (noul), 8 concurrent | 316.50 ms | 348.79 ms | 23.92 | 35.06 ms | 56.99 ms | 163.96 |
| `POST /v1/systemone` (noul), 32 concurrent | 1108.97 ms | 1486.42 ms | 25.82 | 88.49 ms | 152.53 ms | 209.25 |

CPU is unbatched and saturates at ~24–26 req/s. CUDA batches and reaches
~209 req/s at 32 concurrent in the PowerShell bench. The Rust `loadgen` example
reaches ~345 req/s saturation with F16 and `--batch-max 32`; `--batch-max 64`
with 256 in-flight reaches ~359 req/s. F16 probabilities were within ~0.0005 in
spot checks. Full concurrency, F16, option-count
and reproduction details are in the root README.

cuDNN links on this machine, but Laya has no convolution ops; distinct-state
32-concurrent throughput is ~211 vs ~209 req/s for plain CUDA, within noise.

## Test

```bash
cargo test --offline -p sys1-server
```
