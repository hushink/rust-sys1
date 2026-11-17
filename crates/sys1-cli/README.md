# sys1-cli

[简体中文](README.zh-CN.md)

The `sys1` binary: model hub, one-shot inference, and TypeSafe-compatible server.

## Commands

| Command | Purpose |
|---|---|
| `sys1 serve` | Run the HTTP server |
| `sys1 pull <ref>` | Download/materialize a model |
| `sys1 list` | List cached models |
| `sys1 show <ref>` | Print a resolved manifest |
| `sys1 rm <ref>` | Remove a cached model |
| `sys1 run <ref> --input request.json` | One-shot inference |
| `sys1 doctor` | Print build/device/cache diagnostics |

## Model references

- `local:/path/to/model` or an existing path
- `hf:convaiinnovations/laya@main#multilingual`
- `modelscope:owner/name@master`
- `https://models.example.com/laya/model.json`

A local Laya directory without `model.json` is accepted directly:

```bash
sys1 run local:/path/to/laya-multilingual --input request.json
```

## ONNX models

The CLI enables the `onnx` feature by default. A directory containing
`model.onnx` (or an `onnx.json` sidecar naming another file),
`rl_agent_config.json`, and a tokenizer is detected as an ONNX Laya export:

```bash
sys1 serve --model laya-onnx=local:/path/to/laya-onnx
sys1 run local:/path/to/laya-onnx --input request.json
```

ONNX Runtime is loaded dynamically at model load time. Set `ORT_DYLIB_PATH` to
the `onnxruntime` shared library, or install it on the platform loader path.
Use a current ONNX Runtime for multi-gigabyte exports. The graph input/output
contract, optional `onnx.json` fields, and the pure NumPy exporter are
documented in [`sys1-laya`](../sys1-laya/README.md#onnx-runtime-backend).
The CLI `pull`/`show` paths also accept ONNX manifests
(`backend = "onnx"`, `format = "onnx"`).

## Serve

```bash
# Single checkpoint
sys1 serve --model laya-multilingual=local:/path/to/laya-multilingual

# Repository root: registers three variants + lazy laya-latest router
sys1 serve --model laya-latest=local:/path/to/convaiinnovations/laya
```

Useful flags:

```text
--bind 127.0.0.1:8080
--api-key <key>            # or SYS1_API_KEY
--device auto|cpu|cuda:N|metal
--dtype auto|f32|f16|bf16   # f16 is faster at high CUDA concurrency
--tf32                     # optional TF32 for F32 CUDA GEMMs
--no-local-window          # use eager attention for local layers
--max-len <tokens>
--laya-confidence typesafe|laya
--laya-router-default english|multilingual|typed-decisions  # router fallback
--laya-auto-task-detection # opt in to typed-decisions workflow matching
--laya-lang-guess <code>   # router-wide language hint, e.g. en or ro
--batch-max 32             # requests per forward pass
--batch-wait-ms 2          # batch collection window
--max-attention-elements 67108864  # chunk long batches by attention memory
--no-batch                 # disable cross-request batching
--no-admin                 # disable /api/models management endpoints
--metrics                  # force /metrics and /api/metrics on
--no-metrics               # disable /metrics and /api/metrics
--preload <name>           # load before serving; repeatable
--log-level <filter>       # or SYS1_LOG_LEVEL / RUST_LOG
--log-format text|json     # or SYS1_LOG_FORMAT
--log-file <file>          # or SYS1_LOG_FILE; append instead of stdout
--cors-origin <origin>     # allowlist; repeatable, overrides config
--no-cors                  # disable CORS headers
--offline
--cache <dir>              # or SYS1_CACHE
--max-cache-bytes <bytes>  # or SYS1_MAX_CACHE_BYTES; 0 = unlimited
```

## Model management API

`sys1 serve` exposes an operational API outside `/v1` for residency and live
catalog changes. It is enabled by default and uses the same Bearer key as
`/v1/*`; disable it with `--no-admin` or `[server] admin = false`.

| Method | Path | Meaning |
|---|---|---|
| `GET` | `/api/models` | Catalog, aliases, residency, LRU limits |
| `POST` | `/api/models/{name}/load` | Load a model into memory now |
| `POST` | `/api/models/{name}/unload` | Drop the resident instance (idempotent) |
| `POST` | `/api/models/{name}/reload` | Unload and load again |
| `POST` | `/api/models` | Resolve a reference and add it to the live catalog |
| `DELETE` | `/api/models/{name}` | Unload and remove it from the live catalog |
| `GET` | `/api/cache` | List cached models, keys, sizes, and files |
| `DELETE` | `/api/cache/{key}` | Remove a cache entry from disk |
| `POST` | `/api/config/reload` | Re-read `sys1.toml` and apply safe model changes |

```bash
curl http://127.0.0.1:8080/api/models
curl -X POST http://127.0.0.1:8080/api/models/laya-latest/load
curl -X POST http://127.0.0.1:8080/api/models \
  -H 'Content-Type: application/json' \
  -d '{"reference":"hf:convaiinnovations/laya-multilingual"}'
curl -X DELETE http://127.0.0.1:8080/api/models/laya-multilingual

# Disk cache: list keys, then delete one by key. Resident models and active
# downloads must finish/unload first (409 otherwise).
curl http://127.0.0.1:8080/api/cache
curl -X DELETE http://127.0.0.1:8080/api/cache/<key>

# After editing [models] in sys1.toml:
curl -X POST http://127.0.0.1:8080/api/config/reload
```

`GET /api/cache` returns `cache_dir`, `limit_bytes`, `used_bytes`,
`free_bytes`, `reclaimable_bytes`, `model_count`, `resident_count`, and the
`models` array. Each model includes `key`, `name`, `directory`, actual
`size_bytes`, `last_used_unix`, `evictable`, and its `files`.

When `[hub] max_cache_bytes` is set, each pull first evicts the
least-recently-used non-resident cache entries until the new files fit.
Resident models in this process and active downloads are protected;
`reclaimable_bytes` shows what eviction could free now. A model larger than
the whole quota, or one blocked by resident entries, fails with
`cache quota exceeded`. Cache hits still enforce the quota. `sys1 rm` is an
explicit deletion and cannot detect models resident in another process; stop
that process or unload the model first.

```bash
# Remove one cached model and show current usage
sys1 rm laya-multilingual
sys1 list
```

Adding a reference materializes it through the hub (unless cached) and makes it
available to `/v1/systemone` immediately. Send `"load": true` to also load the
primary model before the response; a load failure rolls back the registration.
`GET /api/models` marks each entry as `configured` or `runtime`. Runtime
additions and removals are not persisted; `[models]` in `sys1.toml` is
authoritative after a restart. In-flight requests holding a model handle finish
normally.

`--preload NAME` (or `[server] preload`) loads selected models before the
listener starts, so the first request does not pay the cold-load cost. A
preload failure aborts startup with the model name.

`/metrics` (Prometheus) and `/api/metrics` (JSON) are enabled by default in the
CLI; `--no-metrics` disables them, `--metrics` forces them on. They use the same
Bearer key as `/v1/*` when configured. The JSON includes request/error/latency
counters, batch size/latency, model load/unload counts, per-model request counts,
and currently resident models. Preloaded models are already resident when the
HTTP metrics start, so they do not increment `model load` counters.

`Ctrl-C` / `SIGTERM` trigger graceful shutdown: the listener stops accepting,
in-flight requests drain, then the process exits. CORS defaults to permissive;
use `--cors-origin` (repeatable), `--no-cors`, or `[server] cors_origins` to
restrict or disable it.

## Config reload

`POST /api/config/reload` reloads the TOML file passed with `--config` (or the
auto-loaded `sys1.toml`). It applies **only** `[models]` additions/removals:

- Added models are resolved through the hub and registered as `configured`.
- Removed models are unloaded from the catalog (the disk cache is left alone).
- Changing `[hub]`, `[server]`, or an existing model's reference returns
  `409 restart required`; invalid TOML or model resolution failure returns
  `400`. Neither mutates the catalog.
- The endpoint exists only when admin endpoints are enabled and a config file
  was loaded at startup.

## Configuration

`--config <file>` or `SYS1_CONFIG` loads a TOML file; `./sys1.toml` is used
automatically when present. CLI values override environment variables, which
override the file, which overrides built-in defaults.

```toml
[hub]
cache_dir = "D:/models/sys1-cache"
max_cache_bytes = 34359738368   # omit or 0 = unlimited

[server]
device = "cuda:0"
dtype = "f16"
batch_max = 32
max_attention_elements = 67108864
metrics = true
laya_router_default = "multilingual"   # english | multilingual | typed-decisions
laya_auto_task_detection = false       # opt in to workflow matching
# laya_lang_guess = "en"
preload = ["laya-latest"]
log_level = "info"
log_format = "text"          # text | json
# log_file = "logs/sys1.log"
# cors_origins = ["https://app.example"]  # [] disables CORS; omit = permissive

[models]
laya-latest = "local:D:/models/laya"
laya-multilingual = "hf:convaiinnovations/laya-multilingual"
```

See [`sys1.toml.example`](../../sys1.toml.example). Relative local paths resolve
from the config file's directory. `[models]` names also work with `pull`,
`show`, `rm`, and `run`.

## Logging and tracing

The CLI installs the `tracing` subscriber; the libraries only emit events, so
embedders can choose their own subscriber. The CLI supports text or JSON logs,
an optional append-only file, and a filter:

```toml
[server]
log_level = "info"           # sys1=debug,tower_http=info also works
log_format = "json"
log_file = "logs/sys1.log"   # relative to this config file
```

Precedence: CLI flags / `SYS1_LOG_LEVEL` / `SYS1_LOG_FORMAT` /
`SYS1_LOG_FILE` > `RUST_LOG` for the filter > `[server]` > defaults. JSON logs
include `request_id` and `trace_id` fields from the server telemetry events.

`/v1/systemone` accepts a W3C `traceparent` header and echoes a new
`traceparent` (same trace id, fresh server span id) on the response. Every
response also carries `x-typesafe-request-id`; the two ids appear together in
log events and admin model/cache audit events.

## Load generator

```bash
cargo build --release --offline -p sys1-cli --example loadgen
target/release/examples/loadgen --url http://127.0.0.1:8080 \
  --concurrency 64 --requests 512
target/release/examples/loadgen --url http://127.0.0.1:8080 \
  --rate 300 --concurrency 128 --requests 1500
```

Closed-loop mode keeps `--concurrency` requests in flight. `--rate` is an
open-loop offered rate; requests that cannot acquire an in-flight permit are
reported as `dropped`.

## Features

```bash
cargo build --release -p sys1-cli --features cuda
cargo build --release -p sys1-cli --features cudnn
cargo build --release -p sys1-cli --features metal
cargo build --release -p sys1-cli --features ort-cuda
cargo build --release -p sys1-cli --features ort-tensorrt
```

`onnx` is in the default features. Disable it with
`--no-default-features` when ONNX Runtime should not be part of the binary.
`ort-cuda`, `ort-tensorrt`, `ort-directml`, and `ort-coreml` enable ONNX
Runtime execution providers and require the matching SDK/runtime. `sys1 doctor`
prints compiled features and the resolved device.

## Request file

```json
{
  "state": "I was charged twice. Please refund me today.",
  "questions": {
    "billing": {"type": "noul", "instructions": "Is this about billing?"}
  }
}
```

```bash
sys1 run local:/path/to/laya-multilingual --input request.json

# Scan a document longer than the checkpoint's context window
sys1 run local:/path/to/laya-multilingual --input long.json \
  --long --window 512 --stride 256

# Abstain when answer confidence falls below a threshold
sys1 run local:/path/to/laya-multilingual --input request.json \
  --min-confidence 0.8
```

With `--long`, each answer gains a `window` object (`index`, `token_start`,
`token_end`, `count`) and `usage.windows` reports how many windows were
scored. A state that fits one window takes the ordinary single-call path.
`--stride` must not exceed the effective window; `--long-batch` bounds windows
per forward pass.

With `--min-confidence`, each answer gains `abstention` (`passed`,
`abstained`, or `unevaluated`) and `abstention_threshold`; `low_confidence:
true` appears only on abstained answers. The gate compares the largest answer
probability, not the reported confidence, and is not supported with `--long`.
The typed API additionally accepts per-bucket threshold maps.

## Benchmark

```powershell
$env:SYS1_BENCH_MODEL="C:/path/to/laya-multilingual"
pwsh ./scripts/bench.ps1
```

See the root README for results.
