# rust-sys1

[简体中文](README.zh-CN.md)

Type-safe in-process System One inference plus a TypeSafe-compatible HTTP service.

`rust-sys1` runs open decision models locally and exposes the official
`POST /v1/systemone` / `GET /v1/models` wire contract. v1 ships Laya
(`english`, `multilingual`, `typed-decisions`) with a lazy `laya-latest` router.

## Features

- Strict official API: `POST /v1/systemone`, `GET /v1/models`; official `detail` error shape.
- Process-in inference: typed `Questions` / `Answers`, no HTTP or serialization required.
- `InferenceEngine`: one object wires the hub, manifest, backend factory, registry, and typed inference.
- Laya in candle: ModernBERT encoder, decision head, calibration, batch, confidence modes.
- Optional ONNX Runtime backend for Laya exports; CUDA / TensorRT / DirectML / CoreML execution providers via features.
- Model sources: local path, Hugging Face, ModelScope, static HTTP manifest center.
- Content-addressed-ish cache: sha256, HTTP Range resume, offline mode, concurrent pull lock.
- Single-process server: registry, lazy loading, LRU, keep_alive, `spawn_blocking` inference.
- Runtime model management: load, unload, reload, and add/remove models without restarting.
- Observability: Prometheus/JSON metrics, replaceable telemetry, W3C `traceparent`, text/JSON/file logs.
- Accelerator feature skeleton: CPU / CUDA / cuDNN / Metal / Accelerate.

## Workspace

| Crate | Role |
|---|---|
| [`sys1-protocol`](crates/sys1-protocol/README.md) | Official wire types, validation, typed questions/answers |
| [`sys1-core`](crates/sys1-core/README.md) | `DecisionModel`, registry, manifest, typed in-process API, device selection |
| [`sys1-hub`](crates/sys1-hub/README.md) | Model sources and cache (Local / HF / ModelScope / HTTP) |
| [`sys1-laya`](crates/sys1-laya/README.md) | Clean-room Laya implementation, router, ONNX export path |
| [`sys1-ort`](crates/sys1-ort/README.md) | Dynamically loaded ONNX Runtime session and token-scoring layer |
| [`sys1-server`](crates/sys1-server/README.md) | Axum server for `/v1/*` and `/health` |
| [`sys1-cli`](crates/sys1-cli/README.md) | `sys1` binary: serve/pull/list/show/rm/run/doctor |
| [`sys1`](crates/sys1/README.md) | Umbrella in-process engine API |

Each crate has an English `README.md` and a Chinese `README.zh-CN.md`.

## Quick start

### Build

```bash
cargo build --release -p sys1-cli
```

CPU is the default. CUDA / cuDNN / Metal builds:

```bash
cargo build --release -p sys1-cli --features cuda
cargo build --release -p sys1-cli --features cudnn
cargo build --release -p sys1-cli --features metal   # macOS
cargo build --release -p sys1-cli --features ort-cuda
cargo build --release -p sys1-cli --features ort-tensorrt
```

ONNX support is a default CLI feature. ONNX Runtime is loaded dynamically at
model load time; set `ORT_DYLIB_PATH` or install `onnxruntime` on the platform
loader path. Builds do not download runtime binaries.

### Laya checkpoint layout

```text
model.safetensors
encoder/config.json
rl_agent_config.json
tokenizer/tokenizer.json
```

An ONNX export instead provides `model.onnx` (or the file named by
`onnx.json`), `rl_agent_config.json`, and a tokenizer. See the
[`sys1-laya` ONNX contract](crates/sys1-laya/README.md#onnx-runtime-backend).

### Serve

```bash
# One checkpoint
sys1 serve --model laya-multilingual=local:/path/to/laya-multilingual

# Laya repository root: registers laya-english / laya-multilingual /
# laya-typed-decisions and enables the lazy laya-latest router
sys1 serve --model laya-latest=local:/path/to/convaiinnovations/laya
```

Default bind is `127.0.0.1:8080`. Optional Bearer auth:

```bash
sys1 serve --bind 0.0.0.0:8080 --api-key secret \
  --model laya-latest=local:/path/to/laya
```

Cross-request batching is enabled by default (`--batch-max 32
--batch-wait-ms 2`). Use `--no-batch` for the unbatched baseline, or raise
`--batch-max` (for example 64) for high-concurrency GPU throughput. On CUDA,
`--dtype f16` is the fastest high-throughput mode; `--tf32` is an optional
lower-risk F32 speedup. Exact local-window attention is on by default for CUDA
states of at least 512 tokens; use `--no-local-window` for the eager baseline.

Use `--preload laya-latest` (repeatable) or `[server] preload` to load models
before the listener starts. `/metrics` (Prometheus) and `/api/metrics` (JSON)
are enabled by default; `--no-metrics` disables them. Logs default to text on
stdout; use `--log-format json`, `--log-file`, and `--log-level` to change the
sink. `Ctrl-C` / `SIGTERM` drain in-flight requests before exit. CORS is
permissive by default; restrict it with `--cors-origin` / `[server] cors_origins`
or disable it with `--no-cors` / `cors_origins = []`.

### Runtime model management

The CLI exposes an operational API outside `/v1`. It is enabled by default and
authenticates with the same Bearer key as `/v1/*`; disable it with `--no-admin`
or `[server] admin = false`.

| Method | Path | Meaning |
|---|---|---|
| `GET` | `/api/models` | Catalog, aliases, residency, and LRU limits |
| `POST` | `/api/models/{name}/load` | Load a model into memory now |
| `POST` | `/api/models/{name}/unload` | Drop the resident instance (idempotent) |
| `POST` | `/api/models/{name}/reload` | Unload and load again |
| `POST` | `/api/models` | Resolve a `ModelRef` and add it to the live catalog |
| `DELETE` | `/api/models/{name}` | Unload and remove it from the live catalog |

```bash
curl http://127.0.0.1:8080/api/models
curl -X POST http://127.0.0.1:8080/api/models/laya-latest/load
curl -X POST http://127.0.0.1:8080/api/models \
  -H 'Content-Type: application/json' \
  -d '{"reference":"hf:convaiinnovations/laya-multilingual","load":true}'
curl -X POST http://127.0.0.1:8080/api/models/laya-multilingual/unload
curl -X DELETE http://127.0.0.1:8080/api/models/laya-multilingual

# Disk cache (keys come from GET /api/cache); unload resident models first.
curl http://127.0.0.1:8080/api/cache
curl -X DELETE http://127.0.0.1:8080/api/cache/<key>
```

`POST /api/models` accepts the same references as `sys1 pull`, plus `[models]`
names from `sys1.toml`. `"load": true` loads the primary model before returning;
a failure rolls back the registration. `GET /api/models` marks each entry as
`configured` or `runtime`. Added and removed entries are runtime-only: the
configured `[models]` catalog is authoritative after a restart. In-flight
requests that already hold a model handle finish normally; later requests
trigger a lazy load again.

The cache admin API mirrors `sys1 list` / `sys1 rm` inside the serving process:
`GET /api/cache` reports keys, names, directories, sizes, and files;
`DELETE /api/cache/{key}` removes the cache directory and manifest. A model that
is currently resident returns `409`, so unload it first.

After editing `[models]` in `sys1.toml`, apply safe additions/removals without a
restart:

```bash
curl -X POST http://127.0.0.1:8080/api/config/reload
```

Only `[models]` changes are applied. Hub/server settings and changed references
for existing names return `409 restart required`; invalid TOML or a failed
model resolution returns `400`. Neither case mutates the catalog, and reloaded
config models report `origin: "configured"`.

### Metrics and preload

```bash
# Prometheus scrape
curl -s http://127.0.0.1:8080/metrics | head

# JSON snapshot: request/error/latency, batch size/time, model load/unload,
# per-model request counts, and resident models
curl http://127.0.0.1:8080/api/metrics
```

Both endpoints use the same Bearer key as `/v1/*` when configured. CLI defaults:
metrics on, admin on. Library defaults: `ServerConfig::enable_metrics = false`,
`enable_admin = false`.

```toml
[server]
preload = ["laya-latest"]  # load before accepting requests
metrics = true
```

Preloading avoids paying the cold-load cost (several seconds for CPU Laya) on
the first request; an unknown or failing preload aborts startup with the model
name.

### Logging and tracing

The CLI owns the `tracing` subscriber; the libraries never install one, so they
compose with any logging stack. CLI options:

```bash
sys1 serve --log-level "sys1=debug,tower_http=info" \
  --log-format json --log-file logs/sys1.log
```

Equivalent `[server]` keys: `log_level`, `log_format` (`text`/`json`), and
`log_file` (relative to the config file). Precedence is CLI / `SYS1_LOG_*` >
`RUST_LOG` (filter) > config > defaults. JSON records include `request_id` and
`trace_id`.

For embedding, `sys1-server` exposes a replaceable `Telemetry` sink
(`RequestEvent`, `ModelEvent`, `BatchEvent`, `CacheEvent`) through
`ServerExtensions`. The default is a no-op; `TracingTelemetry` emits structured
events. `/v1/systemone` accepts and returns W3C `traceparent` (the incoming
trace id is preserved, a fresh server span id is returned) alongside the
existing `x-typesafe-request-id`.

### Configuration

All deployment-specific values can live in `sys1.toml`; code contains only
built-in defaults. Precedence is **CLI > environment > `sys1.toml` > defaults**.
Use `--config <file>` or `SYS1_CONFIG`; `./sys1.toml` is loaded automatically
when present. See [`sys1.toml.example`](sys1.toml.example).

```toml
[hub]
cache_dir = "D:/models/sys1-cache"  # model download/cache directory
# hf_endpoint = "https://huggingface.co"
# modelscope_endpoint = "https://modelscope.cn"

[server]
# bind = "127.0.0.1:8080"
# device = "cuda:0"
# dtype = "f16"
# batch_max = 32
# batch_wait_ms = 2
# max_attention_elements = 67108864
# local_window = true
# metrics = true
# preload = ["laya-latest"]
# log_level = "info"
# log_format = "text"          # text | json
# log_file = "logs/sys1.log"   # append instead of stdout
# cors_origins = ["https://app.example"]  # [] disables CORS; omit = permissive

[models]
laya-latest = "local:D:/models/laya"       # local model directory
# laya-multilingual = "hf:convaiinnovations/laya-multilingual"
```

Relative paths in the file resolve from the config file's directory.
`[models]` names are usable by `serve`, `run`, `pull`, `show`, and `rm`.
Credentials are preferably supplied through `HF_TOKEN`,
`MODELSCOPE_API_TOKEN`, and `SYS1_HUB_TOKEN`.

### Call the API

```bash
curl http://127.0.0.1:8080/v1/models

curl http://127.0.0.1:8080/v1/systemone \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "laya-latest",
    "state": "I was charged twice. Please refund the extra charge today.",
    "questions": {
      "billing": {"type": "noul", "instructions": "Is this about billing?"},
      "tone": {
        "type": "choice",
        "instructions": "What is the tone?",
        "criteria": {"angry": "Hostile", "calm": "Neutral"}
      }
    }
  }'
```

### In-process

`InferenceEngine` wires the hub, manifest, backend factory, registry, and typed
protocol together:

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

`EngineOptions` groups `hub` (cache, endpoints, credentials), `registry`
(LRU/keep-alive), and `model` (`ModelOptions`: device, dtype, context length,
confidence, attention). `engine.add("hf:...")` / `add_as` register references,
`load` / `unload` control residency, `models` / `model` inspect the catalog,
`predict_batch` runs a native batch, and `systemone` returns the official wire
response. The engine implements `ModelResolver`, so it can also back the
server's `/api/models` runtime management.

The original direct checkpoint API is unchanged:

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

`LayaModel`, `LayaOptions`, `LayaFactory`, and `LayaRouter` remain re-exported
at the crate root; `InferenceEngine` is an additive convenience layer.

## Compatibility notes

- Only official endpoints are implemented. `/v1/decisions` is intentionally absent.
- Validation errors use `422 {"detail": [...]}`; unknown models use `400 {"detail": "..."}`.
- `box` questions and image `assets` are parsed but Laya returns `422 unsupported`; missing
  `TypeSafe-Danger-Zone-Images: true` returns `400`.
- Request limit is 32 MiB. gzip is supported; zstd currently returns `415` in this build.
- Long state returns `400` with `max_tokens_exceeded`; it is never silently truncated.
- `/v1/*` defaults to TypeSafe/ollaya-normalized confidence; use `--laya-confidence laya`
  for Laya's native entropy confidence.
- `/health` is an operational endpoint outside `/v1`.

## Benchmark

Measured 2026-11-17, release build, single machine. Concurrent scenarios send a
distinct state per request, so Laya's identical-request dedup does not inflate
the numbers.

Cross-request batching (`--batch-max 32 --batch-wait-ms 2`) is used on GPU.
Laya on CPU opts out via `DecisionModel::supports_cross_request_batching`: a CPU
forward already uses the thread pool, and serializing independent requests
through one batched forward reduced throughput in measurements. A lone GPU
request only pays a 0.5 ms yield grace period.

Environment:

- CPU: Intel i5-12400F (6 cores / 12 threads), 32 GiB RAM
- GPU: NVIDIA RTX 3060 12 GiB, driver 617.42 / CUDA 13.4
- Model: `laya-multilingual` (mmBERT-base, 322M)
- Client: PowerShell `HttpClient` and the Rust `loadgen` example
- Server: `spawn_blocking` per batch on GPU, per request on CPU

### CPU vs CUDA (F32, distinct states)

CPU numbers are unbatched; CUDA numbers use the default `--batch-max 32`.

| Scenario | CPU mean | CPU p95 | CPU req/s | CUDA mean | CUDA p95 | CUDA req/s |
|---|---:|---:|---:|---:|---:|---:|
| `GET /health` | 0.29 ms | 0.47 ms | 2534 | 0.31 ms | 0.54 ms | 2254 |
| `GET /v1/models` | 0.23 ms | 0.35 ms | 3879 | 0.28 ms | 0.43 ms | 3263 |
| `POST /v1/systemone` (noul) | 84.62 ms | 99.89 ms | 11.81 | 17.39 ms | 21.12 ms | 57.32 |
| `POST /v1/systemone` (noul+score+choice) | 199.72 ms | 214.56 ms | 5.01 | 21.26 ms | 24.06 ms | 46.96 |
| `POST /v1/systemone` (noul), 4 concurrent | 171.29 ms | 181.46 ms | 22.23 | 24.70 ms | 27.46 ms | 125.53 |
| `POST /v1/systemone` (noul), 8 concurrent | 316.50 ms | 348.79 ms | 23.92 | 35.06 ms | 56.99 ms | 163.96 |

CUDA F32 reduces a single-question request from ~85 ms to ~17 ms (~5x) and a
three-question request from ~200 ms to ~21 ms (~9.4x). With distinct states,
GPU batching raises 8-concurrent throughput from ~36 (unbatched) to ~164 req/s.
Run-to-run variance is about 5–10% for concurrent scenarios.

### Concurrency sweep (noul, F32, distinct states)

| Concurrent | CPU mean | CPU p95 | CPU req/s | CUDA mean | CUDA p95 | CUDA req/s |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 87.70 ms | 90.63 ms | 11.22 | 17.09 ms | 18.03 ms | 53.77 |
| 2 | 112.13 ms | 121.24 ms | 17.55 | 30.97 ms | 37.94 ms | 56.81 |
| 4 | 171.16 ms | 180.03 ms | 22.38 | 26.59 ms | 48.91 ms | 101.03 |
| 8 | 312.40 ms | 345.15 ms | 24.25 | 34.95 ms | 59.60 ms | 152.54 |
| 16 | 597.03 ms | 705.60 ms | 24.87 | 53.39 ms | 98.63 ms | 182.77 |
| 32 | 1108.97 ms | 1486.42 ms | 25.82 | 88.49 ms | 152.53 ms | 209.25 |

The CPU is saturated at ~24–26 req/s; CUDA scales to ~209 req/s at 32
concurrent. Closed-loop PowerShell workers under-report steady-state throughput
because runspace startup is included in the wall time. The Rust load generator
below reaches higher concurrency without that client overhead.

### Rust load generator (F16, closed-loop and open-loop)

Build and run:

```powershell
cargo build --release --offline -p sys1-cli --example loadgen
target/release/examples/loadgen.exe --url http://127.0.0.1:8080 `
  --concurrency 32 --requests 256
target/release/examples/loadgen.exe --url http://127.0.0.1:8080 `
  --rate 300 --concurrency 128 --requests 1500
```

| Mode | Concurrency / rate | Mean | p95 | p99 | req/s | dropped |
|---|---:|---:|---:|---:|---:|---:|
| closed-loop | x8 | 33.46 ms | 39.60 ms | 86.06 ms | 238.34 | 0 |
| closed-loop | x32 | 101.39 ms | 134.38 ms | 135.72 ms | 314.00 | 0 |
| closed-loop | x64 | 184.85 ms | 215.08 ms | 215.79 ms | 334.97 | 0 |
| open-loop | 200 rps | 53.02 ms | 98.55 ms | 121.03 ms | 198.85 | 0 |
| open-loop | 300 rps | 138.76 ms | 348.37 ms | 385.37 ms | 294.58 | 12 |
| open-loop | 400 rps | 295.40 ms | 360.90 ms | 366.94 ms | 343.75 | 184 |

These are F16, `--batch-max 32`, distinct states. The server saturates around
~345 req/s with this closed-loop client; `--batch-max 64` reached ~359 req/s
with 256 in-flight. Open-loop 400 rps overdrives the 128-request in-flight cap,
hence the dropped offers; errors stayed at zero. F32 reaches ~250 req/s
(closed-loop x64 249.7; open-loop 250 offered 243.9). Model-only F16 batch-32
ceiling is ~396 req/s, so HTTP, tokenization, batching and JSON add roughly
10–15%.

### F16 high-throughput mode

`--dtype f16` keeps weights in F32 (for loading) but runs the encoder and head
in F16. It is the largest remaining GPU lever:

| Scenario | F32 mean | F32 req/s | F16 mean | F16 req/s |
|---|---:|---:|---:|---:|
| noul, 1 | 17.39 ms | 57.32 | 19.59 ms | 50.91 |
| noul, 4 concurrent | 24.70 ms | 125.53 | 21.47 ms | 141.90 |
| noul, 8 concurrent | 35.06 ms | 163.96 | 33.28 ms | 183.55 |
| noul, 16 concurrent | 53.39 ms | 182.77 | 40.73 ms | 219.27 |
| noul, 32 concurrent | 88.49 ms | 209.25 | 70.47 ms | 248.47 |

In a three-question check, F16 changed probabilities by at most ~0.0005
(`choice` 0.6343, `noul` 0.9712→0.9711, `score` 1.6681→1.6676). F32 remains the
default for maximum numerical fidelity; use F16 for throughput on Ampere+ GPUs.
BF16 was similar in speed but less accurate in spot checks. `--tf32` enables
TF32 GEMMs for F32; measured 32-concurrent throughput was ~217 vs ~209 req/s
(within noise), with outputs within ~1e-4.

Identical concurrent requests are deduplicated inside Laya's batch preparation.
For example, 8 identical F32 requests measured ~21 ms / ~240 req/s on CUDA. That
is a workload-specific cache-like win, not general batching.

### Long-context attention (CUDA)

Local layers use an exact sliding-window attention by default when the state is
at least 512 tokens: queries only gather the `±64` keys allowed by the local
mask, in 128-query blocks. It is mathematically equivalent to eager attention;
F32 outputs match exactly, F16 differs by at most ~`6e-4` from rounding. Use
`--no-local-window` for the eager baseline. CPU keeps eager attention.

| Noul tokens | Eager | Window | Speedup | Max answer diff |
|---:|---:|---:|---:|---:|
| 266 | 27.00 ms | 27.35 ms | 0.99x (eager path) | 0 |
| 518 | 87.20 ms | 75.18 ms | 1.16x | 6e-4 |
| 932 | 194.33 ms | 141.10 ms | 1.38x | 6e-4 |

F32 at 932 tokens measured 184.92 → 142.06 ms (1.30x) with zero answer diff.
Four concurrent 932-token F16 requests measured ~724 ms / 5.5 req/s eager vs
~433 ms / 8.3 req/s windowed.

```powershell
# Model-level F16/F32 accuracy + latency
$env:SYS1_LAYA_TEST_ROOT="C:/path/to/laya-multilingual"
$env:SYS1_LONG_DEVICE="cuda"
$env:SYS1_LONG_DTYPE="f16"
cargo run --release --offline -p sys1-laya --features cuda `
  --example long_context_bench

# HTTP concurrency: generate a long state and feed it to loadgen
$state = ("I was charged twice for invoice 12345 and need a refund today. " * 50)
target/release/examples/loadgen.exe --url http://127.0.0.1:8080 `
  --concurrency 4 --requests 8 --state $state
```

A FlashAttention prototype cut a 932-token request to ~69 ms (2.6x), but its
answers differed materially from eager/F32 (noul 0.67 vs 0.98), so it remains
disabled. For very long F32 states, lower `--batch-max` (8–16) to bound eager
attention memory.

### Option-count sweep (F32, sequential)

| Question | Options | CPU mean | CUDA mean |
|---|---:|---:|---:|
| choice | 1 | 58.84 ms | 15.08 ms |
| choice | 2 | 66.40 ms | 15.14 ms |
| choice | 3 | 76.82 ms | 16.97 ms |
| choice | 5 | 93.06 ms | 17.32 ms |
| choice | 10 | 137.19 ms | 17.26 ms |
| choice | 20 | 298.44 ms | 23.27 ms |
| choice | 40 | 462.79 ms | 36.14 ms |
| score | 2 | 65.09 ms | 15.84 ms |
| score | 3 | 75.62 ms | 15.34 ms |
| score | 5 | 95.45 ms | 15.85 ms |
| score | 10 | 150.60 ms | 19.19 ms |

Every run returned the expected number of labels and probabilities summing to
1.00. CPU latency grows roughly linearly with option count because the decision
head sees more option markers. CUDA stays near 15–18 ms up to 10 options;
20/40 options add ~8/21 ms.

### cuDNN

The cuDNN build links on this machine (`--features cudnn`, with the cuDNN `lib`
directory on `LIB`). With distinct states, 32-concurrent throughput is ~211 vs
~209 req/s for plain CUDA—within run-to-run noise. Laya's ModernBERT has no
convolution ops, so cuDNN adds no meaningful speedup; the CUDA build is the
relevant accelerator. cuDNN would only matter for future models with
convolutional layers.

### Reproduce

```powershell
# CPU: Laya automatically opts out of cross-request batching
$env:SYS1_BENCH_MODEL="C:/path/to/laya-multilingual"
pwsh ./scripts/bench.ps1 -Device cpu -Dtype f32 -UniqueStates -OptionSweep -ConcurrencySweep

# CUDA: build with the Visual Studio environment first
cargo build --release --offline -p sys1-cli --features cuda --target-dir target-cuda
pwsh ./scripts/bench.ps1 -Exe target-cuda/release/sys1.exe `
  -Device cuda:0 -Dtype f32 -UniqueStates -ConcurrencySweep

# High-throughput F16, or optional TF32 for F32
pwsh ./scripts/bench.ps1 -Exe target-cuda/release/sys1.exe `
  -Device cuda:0 -Dtype f16 -UniqueStates -ConcurrencySweep
pwsh ./scripts/bench.ps1 -Exe target-cuda/release/sys1.exe `
  -Device cuda:0 -Dtype f32 -Tf32 -UniqueStates

# Unbatched baseline for comparison
pwsh ./scripts/bench.ps1 -Exe target-cuda/release/sys1.exe `
  -Device cuda:0 -Dtype f32 -UniqueStates -NoBatch
```

## Testing

```bash
cargo test --offline --workspace

# Require the ONNX Runtime-backed tests instead of skipping when the dylib is absent
$env:ORT_DYLIB_PATH="C:/path/to/onnxruntime.dll"
$env:SYS1_REQUIRE_ORT="1"
cargo test --offline -p sys1-ort
cargo test --offline -p sys1-laya --features onnx
```

Hub tests spin up loopback model centers and cover static HTTP manifests,
Bearer tokens, Range-resumed downloads, HF pagination/subfolders/LFS hashes,
and ModelScope token headers plus 3000-entry truncation splitting. CI runs fmt,
clippy, and tests on Linux, Windows, and macOS (`.github/workflows/ci.yml`). The optional
`.github/workflows/gpu.yml` runs the long-context parity test and a loadgen
smoke test on a self-hosted Linux `gpu` + `cuda` runner; set the repository
variable `SYS1_LAYA_TEST_ROOT` to a cached multilingual checkpoint.

Laya golden tests are ignored by default because they need local checkpoints
and the official golden vectors:

```bash
$env:SYS1_LAYA_TEST_ROOT="C:/.../snapshots/<sha>"
$env:SYS1_LAYA_GOLDEN_DIR="D:/path/to/laya/laya-rust/tests/golden"
cargo test -p sys1-laya -- --ignored --nocapture

# Long-context exact local-window parity (CUDA, cached checkpoint)
cargo test --release -p sys1-laya --features cuda --test long_context -- --ignored --nocapture
```

## Known gaps / roadmap

- zstd request compression.
- A real Laya ONNX export. The optional ORT backend, dynamic loading, synthetic
  contract tests, and factory wiring are implemented; no official ONNX artifact
  is published. TensorRT remains available only through the ORT TensorRT EP.
- TensorRT EP CI requires a self-hosted GPU runner with TensorRT installed.
- Long-context scanning, act head, abstention, parallel option layout (v1.1).
- FlashAttention for long contexts: a prototype built with a hdim64-only
  candle-flash-attn + CUDA 13.3 and ran at ~69 ms vs ~181 ms for a 932-token
  state, but diverged from eager/F32 answers. The upstream full build with
  CUDA 13.4/Windows also hit a `cicc` access violation. Not merged until
  numerical parity is understood.
- CPU int8 quantization / MKL. A candle `Q8_0` GEMM probe was only ~6% faster
  than F32 on a representative 2304×768×1024 layer and `Q4_0` was slower, so it
  was not integrated; MKL crates are not available in the offline cache.
- The `laya-latest` router is heuristic; full official routing golden parity is pending.
- HF / ModelScope live-download integration tests.
- Laya weight license must be checked before redistribution. The repo does not bundle weights.

## Documentation

- [Design](docs/design.md)
- [ADRs](docs/adr/)
- [Glossary](GLOSSARY.md)

## License

Apache-2.0. Laya architecture/behavior references the official Laya repository and
`sys1` 0.0.3 (both Apache-2.0). No model weights are included.
