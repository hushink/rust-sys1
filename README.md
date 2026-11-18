# rust-sys1

[简体中文](README.zh-CN.md)

Type-safe in-process System One inference plus a TypeSafe-compatible HTTP service.

`rust-sys1` runs open decision models locally and exposes the official
`POST /v1/systemone` / `GET /v1/models` wire contract. It ships Laya
(`english`, `multilingual`, `typed-decisions`) with a lazy `laya-latest` router,
plus NLI, GLiClass, Qwen3Guard, Von, Decima, Kev, Decision 1.0 and Clef.

## Features

- Strict official API: `POST /v1/systemone`, `GET /v1/models`; official `detail` error shape.
- Process-in inference: typed `Questions` / `Answers`, no HTTP or serialization required.
- `InferenceEngine`: one object wires the hub, manifest, backend factory, registry, and typed inference.
- Typed extensions: long-context scanning, `min_confidence` abstention, and act-head probabilities.
- Multiple families in candle: Laya, NLI, GLiClass, Qwen3Guard, Von, Decima,
  Kev, Decision 1.0 and Clef, with automatic checkpoint detection.
- Decoder backbones: Qwen3 and Qwen3.5 (Gated DeltaNet + gated attention),
  including PEFT LoRA merge and tied/untied LM heads.
- Laya in candle: ModernBERT encoder, decision head, calibration, batch, confidence modes.
- Optional ONNX Runtime backend for Laya exports; CUDA / TensorRT / DirectML / CoreML execution providers via features.
- Pure NumPy Laya-to-ONNX exporter with official golden logits within 2.5e-6.
- Model sources: local path, Hugging Face, ModelScope, static HTTP manifest center
  ([manifest schema](docs/model-manifest.md)).
- Content-addressed-ish cache: sha256, HTTP Range resume, offline mode, concurrent pull lock, byte quota with LRU eviction.
- Single-process server: registry, lazy loading, LRU, keep_alive, `spawn_blocking` inference, gzip/zstd requests.
- Runtime model management: load, unload, reload, and add/remove models without restarting.
- Observability: Prometheus/JSON metrics, replaceable telemetry, W3C `traceparent`, text/JSON/file logs.
- Accelerator feature skeleton: CPU / CUDA / cuDNN / Metal / Accelerate.

## Supported models and family tags

Every checkpoint is detected from its directory layout. The **family tag** in
the first column is the value of `ModelManifest.family` (and `model.json` for a
[custom HTTP center](docs/model-manifest.md)); it selects the implementation in
`sys1-families`. The model id on the left of `--model NAME=REF` (or a
`[models]` key) is independent of the tag — for example
`sys1 serve --model nli=hf:MoritzLaurer/ModernBERT-base-zeroshot-v2.0` uses the
tag `nli` and the id `nli`.

| Family tag | Model | Notes | Reference |
|---|---|---|---|
| `laya` | Laya `english`, `multilingual`, `typed-decisions` | ModernBERT-large 421M / mmBERT-base 322M; a repository root registers all three variants and the lazy `laya-latest` router; optional ONNX backend | `hf:convaiinnovations/laya` |
| `nli` | `ModernBERT-base-zeroshot-v2.0` | 150M, ~299 MB safetensors | `hf:MoritzLaurer/ModernBERT-base-zeroshot-v2.0` |
| `nli` | `ModernBERT-large-zeroshot-v2.0` | 395M, ~792 MB | `hf:MoritzLaurer/ModernBERT-large-zeroshot-v2.0` |
| `nli` | `deberta-v3-large-zeroshot-v2.0` | 435M, ~870 MB F16; disentangled attention | `hf:MoritzLaurer/deberta-v3-large-zeroshot-v2.0` |
| `gliclass` | `gliclass-instruct-edge-v1.0` | 32.7M Ettin/ModernBERT, ~131 MB, Apache-2.0 | `hf:knowledgator/gliclass-instruct-edge-v1.0` |
| `qwen3guard` | `Qwen3Guard-Gen-0.6B` | Dense Qwen3 0.6B; fixed `safety` / `unsafe` / `unsafe_strict` / `category` presets | `hf:Qwen/Qwen3Guard-Gen-0.6B` |
| `von` | Von 1.1 | ModernBERT-large + `option_marker.pt` | `hf:wfzyx/von` |
| `decima` | `decima-small` | multilingual-e5-small BERT encoder, v1 layout | `hf:amyrmahdy/decima-small` |
| `decima` | `decima-base`, `decima-base-clean` | mmBERT-base, v2 layout | `hf:amyrmahdy/decima-base`, `hf:amyrmahdy/decima-base-clean` |
| `decima` | `decima-agent` | mmBERT-base, 8192-token state, `choice_fit` | `hf:amyrmahdy/decima-agent` |
| `decima` | `decima-guard`, `decima-guard-small` | mmBERT-base / mmBERT-small guard checks, `choice_fit` | `hf:amyrmahdy/decima-guard`, `hf:amyrmahdy/decima-guard-small` |
| `kev` | `kev-0.8b`, `kev-4b`, `kev-9b` | Qwen3.5 base + PEFT LoRA + pointer head; the declared base is pulled automatically | `hf:jaredpalmer/kev-0.8b`, `hf:jaredpalmer/kev-4b`, `hf:jaredpalmer/kev-9b` |
| `decision` | `Decision-1.0-Eos-0.8B` | Qwen3.5 0.8B backbone + endpoint head | `hf:vllm-sr/Decision-1.0-Eos-0.8B` |
| `clef` | `clef-flash` | Qwen3.5 9B joint-schema head, ~18.8 GB BF16 | `hf:Cloudflare/clef-flash` |

All candle families support `choice`, `score`, and `noul` questions unless the
family README says otherwise (Qwen3Guard accepts only its presets; box questions
are unsupported). The `sys1.toml.example` and each crate README show the exact
checkpoint layout and the upstream parity tests that exercise these models.

Validation status: Laya's official golden logits, NLI (ModernBERT and
DeBERTa-v3), GLiClass, Qwen3Guard 0.6B, Von, Decision, Decima base/small/
guard-small, Kev 0.8B/4B/9B, and Clef flash have parity tests against the
upstream implementations or recorded decisions. The other checkpoints in the
table share the same architecture and layout but are not individually exercised.

## Workspace

| Crate | Role |
|---|---|
| [`sys1-protocol`](crates/sys1-protocol/README.md) | Official wire types, validation, typed questions/answers |
| [`sys1-core`](crates/sys1-core/README.md) | `DecisionModel`, registry, manifest, typed in-process API, device selection |
| [`sys1-hub`](crates/sys1-hub/README.md) | Model sources and cache (Local / HF / ModelScope / HTTP) |
| [`sys1-laya`](crates/sys1-laya/README.md) | Clean-room Laya implementation, router, ONNX export path |
| [`sys1-modernbert`](crates/sys1-modernbert/README.md) | Shared candle ModernBERT encoder |
| [`sys1-nli`](crates/sys1-nli/README.md) | ModernBERT / DeBERTa-v3 zero-shot NLI decision model |
| [`sys1-gliclass`](crates/sys1-gliclass/README.md) | GLiClass uni-encoder decision model (32M edge) |
| [`sys1-qwen`](crates/sys1-qwen/README.md) | Shared candle Qwen3 / Qwen3.5 decoder backbones |
| [`sys1-qwen3guard`](crates/sys1-qwen3guard/README.md) | Qwen3Guard Gen fixed-preset safety model |
| [`sys1-von`](crates/sys1-von/README.md) | Von 1.1 option-marker encoder |
| [`sys1-decima`](crates/sys1-decima/README.md) | Decima late-interaction encoder |
| [`sys1-kev`](crates/sys1-kev/README.md) | Kev pointer-head decoder |
| [`sys1-decision`](crates/sys1-decision/README.md) | Decision 1.0 endpoint-head decoder |
| [`sys1-clef`](crates/sys1-clef/README.md) | Clef joint-schema decoder |
| [`sys1-families`](crates/sys1-families/README.md) | Family detection and dispatch for all compiled families |
| [`sys1-ort`](crates/sys1-ort/README.md) | Dynamically loaded ONNX Runtime session and token-scoring layer |
| [`sys1-server`](crates/sys1-server/README.md) | Axum server for `/v1/*` and `/health` |
| [`sys1-cli`](crates/sys1-cli/README.md) | `sys1` binary: serve/pull/list/show/rm/run/doctor |
| [`sys1`](crates/sys1/README.md) | Umbrella in-process engine API |

The Python workspace lives in [`python/tools`](python/tools/README.md) and is
managed by uv: ONNX export/check, the ORT fixture generator, and the HTTP
benchmark. Each crate has an English `README.md` and a Chinese
`README.zh-CN.md`; the Python package does too.

## Use as a library

The workspace is not published to crates.io yet. Depend on it through Git or a
local checkout:

```toml
[dependencies]
sys1 = { git = "https://github.com/hushink/rust-sys1" }
# Or, from a checkout:
# sys1 = { path = "../rust-sys1/crates/sys1" }
```

`sys1` is the recommended entry point. It re-exports the protocol, core, hub, and
family types and provides `InferenceEngine`, which handles downloads, checkpoint
detection, lazy loading, batching, and typed answers:

```rust
use sys1::{EngineOptions, InferenceEngine, Questions, State};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let engine = InferenceEngine::new(EngineOptions::default())?;
    engine.add_as("nli", "hf:MoritzLaurer/ModernBERT-base-zeroshot-v2.0")?;

    let state = State::text("I was charged twice and need a refund today.");
    let questions = Questions::new()
        .choice("team", [("billing", "Charges and refunds"), ("technical", "Bugs")])
        .noul("refund", "Is the customer asking for a refund?");

    let output = engine.predict("nli", &state, &questions)?;
    let team = output.answers.choice("team")?;
    let refund = output.answers.noul("refund")?;
    println!("{} {:.3}", team.choice, refund.noul);
    Ok(())
}
```

For an existing local checkpoint, register it with
`InferenceEngine::from_directory(path, EngineOptions::default())?` and use the
directory name as the model id. `add_as` accepts every `local:`, `hf:`,
`modelscope:`, and HTTP-manifest reference described above; Kev adapters pull
their declared Qwen3.5 base automatically.

### Use one family directly

Add the family crate plus the shared traits and wire types:

```toml
[dependencies]
sys1-nli = { path = "../rust-sys1/crates/sys1-nli" }
sys1-core = { path = "../rust-sys1/crates/sys1-core" }
sys1-protocol = { path = "../rust-sys1/crates/sys1-protocol" }
# Or with Git:
# sys1-nli = { git = "https://github.com/hushink/rust-sys1" }
```

```rust
use std::path::Path;
use sys1_core::{DecisionModel, Questions};
use sys1_nli::{NliModel, NliOptions};
use sys1_protocol::State;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model = NliModel::load_dir(Path::new("/models/nli"), NliOptions::default())?;
    let output = model.predict(
        &State::text("I was charged twice."),
        &Questions::new().noul("refund", "Is a refund requested?"),
    )?;
    println!("{:.3}", output.answers.noul("refund")?.noul);
    Ok(())
}
```

Each crate README shows how to depend on it (and a minimal example);
`sys1-cli` documents installing the binary instead. The
[Workspace](#workspace) table maps each goal to a crate; in short:

| Goal | Crate |
|---|---|
| In-process engine over any family | `sys1` |
| Official wire types and validation | `sys1-protocol` |
| Traits, registry, typed questions/answers | `sys1-core` |
| Download, manifest, and cache | `sys1-hub` |
| One model family | `sys1-laya`, `sys1-nli`, `sys1-gliclass`, `sys1-qwen3guard`, `sys1-von`, `sys1-decima`, `sys1-kev`, `sys1-decision`, `sys1-clef` |
| HTTP router or standalone server | `sys1-server` |
| CLI binary | `sys1-cli` (`cargo install --path crates/sys1-cli`) |

### Features

`sys1` enables `laya`, `hub`, and `families` by default. Add `onnx`, `cuda`,
`cudnn`, `metal`, `accelerate`, or the ONNX Runtime providers as needed:

```toml
sys1 = { git = "https://github.com/hushink/rust-sys1", features = ["cuda"] }
```

The workspace requires Rust 1.92 (edition 2024) and is Apache-2.0 licensed.

## Quick start

### Build

```bash
cargo build --release -p sys1-cli

# Python tools (uv workspace, Python 3.10-3.14)
uv sync --all-extras
uv run sys1-bench --help
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
`onnx.json`), `rl_agent_config.json`, a tokenizer, and optionally
`act_head.safetensors`. Export one with the pure NumPy/ONNX translator (no
PyTorch), then check it against the official golden logits:

```bash
uv run --extra onnx sys1-export-onnx \
  /path/to/laya-multilingual /path/to/laya-onnx

uv run --extra check sys1-check-onnx \
  /path/to/laya-onnx /path/to/golden/multilingual/case_dict_state.json
```

All three checkpoints matched the official golden logits with max
`|ONNX - golden|` between 1.7e-6 and 2.5e-6. The exporter also emits a `pooled`
output and copies the act head, so ONNX backend actions match candle within
2e-3. See the
[`sys1-laya` ONNX contract](crates/sys1-laya/README.md#onnx-runtime-backend).
Use a current ONNX Runtime for these multi-gigabyte graphs; 1.17 was unstable
on Windows for the 1.2 GiB multilingual export, while 1.32 worked.

### Serve

```bash
# One checkpoint
sys1 serve --model laya-multilingual=local:/path/to/laya-multilingual

# Laya repository root: registers laya-english / laya-multilingual /
# laya-typed-decisions and enables the lazy laya-latest router
sys1 serve --model laya-latest=local:/path/to/convaiinnovations/laya

# Candle families (auto-detected from the checkpoint layout):
sys1 serve --model nli=hf:MoritzLaurer/ModernBERT-base-zeroshot-v2.0
sys1 serve --model nli=hf:MoritzLaurer/deberta-v3-large-zeroshot-v2.0
sys1 serve --model gliclass=hf:knowledgator/gliclass-instruct-edge-v1.0
sys1 serve --model qwen3guard=hf:Qwen/Qwen3Guard-Gen-0.6B
sys1 serve --model von=hf:wfzyx/von
sys1 serve --model decima=hf:amyrmahdy/decima-base
sys1 serve --model decision=hf:vllm-sr/Decision-1.0-Eos-0.8B
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
# max_cache_bytes = 34359738368    # omit or 0 = unlimited; LRU eviction before pull
# hf_endpoint = "https://huggingface.co"
# modelscope_endpoint = "https://modelscope.cn"
# timeout_secs = 60            # metadata/connection timeout
# download_timeout_secs = 300  # per-attempt; large files resume by range

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

[families]
# max_batch = 32                       # encoder families: NLI/GLiClass/Von/Decima
# [families.qwen3guard]
# max_len = 32768
# [families.von]
# max_len = 8192
# [families.clef]
# max_len = 8192
# [families.decima]
# choice_fit = false                  # override decima.json
# [families.kev]
# base_dir = "D:/models/qwen3.5-0.8b-base"
# temperature = 2.351                 # otherwise kev.json / head.pt
# [families.decision]
# temperature = 1.0

[models]
laya-latest = "local:D:/models/laya"       # local model directory
# laya-multilingual = "hf:convaiinnovations/laya-multilingual"
# kev = "hf:jaredpalmer/kev-0.8b"          # pulls the declared Qwen3.5 base too

# Per-model family overrides (omitted keys inherit [families]):
# [models.nli-small]
# reference = "hf:MoritzLaurer/ModernBERT-base-zeroshot-v2.0"
# [models.nli-small.families]
# max_batch = 8
```

Relative paths in the file resolve from the config file's directory.
`[models]` names are usable by `serve`, `run`, `pull`, `show`, and `rm`.
`/api/config/reload` applies `[models]` additions/removals only; family changes,
including per-model `families` blocks, need a restart. `sys1 list --tree` groups
Kev adapters under their base; `sys1 rm --with-deps` removes the base too when
no other cached adapter declares it.
Kev adapters resolve their Qwen3.5 base from `base_dir`, a `base/` sibling, or
the repository recorded in `head.pt` / `adapter_config.json`; an `hf:` adapter
therefore pulls the base on first use. Credentials are preferably supplied
through `HF_TOKEN`, `MODELSCOPE_API_TOKEN`, and `SYS1_HUB_TOKEN`.

With `[hub] max_cache_bytes` (or `--max-cache-bytes` /
`SYS1_MAX_CACHE_BYTES`), each pull evicts the least-recently-used non-resident
cache entries until the new files fit. Models resident in the serving process
and active downloads are protected; a model larger than the quota fails with
`cache quota exceeded`. `GET /api/cache` reports `limit_bytes`, `used_bytes`,
`free_bytes`, and `reclaimable_bytes`, and `sys1 list` prints the same summary.
`sys1 rm` remains an explicit deletion; stop other serving processes or unload
their models first.

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
`predict_batch` runs a native batch, `predict_long` scans a long state in
overlapping windows with typed window attribution, `predict_with` applies
per-call controls such as `min_confidence` abstention, and `systemone` returns
the official wire response. The engine implements `ModelResolver`, so it can
also back the server's `/api/models` runtime management.

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
- Request limit is 32 MiB compressed and decompressed. gzip and zstd request
  bodies are supported; malformed frames return `400`.
- Long state returns `400` with `max_tokens_exceeded` from `/v1` and ordinary
  `predict`; typed `predict_long` / `sys1 run --long` scan it in overlapping
  windows instead.
- `/v1/*` defaults to TypeSafe/ollaya-normalized confidence; use `--laya-confidence laya`
  for Laya's native entropy confidence.
- `/health` is an operational endpoint outside `/v1`.

## Benchmark

Laya's detailed latency, throughput, concurrency, batching, F16, long-context,
option-sweep, and cuDNN measurements are in
[docs/laya-benchmarks.md](docs/laya-benchmarks.md)
([简体中文](docs/laya-benchmarks.zh-CN.md)).

The newer families are covered in
[docs/benchmarks.en.md](docs/benchmarks.en.md)
([简体中文](docs/benchmarks.md)).

## Testing

```bash
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
```

The default suite needs no network, model weights, or GPU. Real-checkpoint
parity tests are ignored and opt-in. The full matrix — Laya, NLI, GLiClass,
Qwen backbones, Qwen3Guard, Von, Decima, Kev 0.8B/4B/9B, Decision, Clef, the
umbrella engine, every environment variable, the Python tooling, CI jobs, and
troubleshooting — is in **[docs/testing.md](docs/testing.md)**
([简体中文](docs/testing.zh-CN.md)).

CI runs formatting, clippy with `-D warnings`, and the workspace tests on
Linux, Windows, and macOS (`.github/workflows/ci.yml`), plus a Python
lint/fixture job. The optional `.github/workflows/gpu.yml` runs the Laya
long-context parity test and a loadgen smoke test on a self-hosted Linux
`gpu` + `cuda` runner.

## Known gaps / roadmap

- TensorRT EP CI requires a self-hosted GPU runner with TensorRT installed.
- A published upstream Laya ONNX artifact. The repository exporter now produces
  parity-checked models locally; TensorRT remains available only through the
  ORT TensorRT EP.
- Parallel option layout (v1.1).
- FlashAttention for long contexts: a prototype built with a hdim64-only
  candle-flash-attn + CUDA 13.3 and ran at ~69 ms vs ~181 ms for a 932-token
  state, but diverged from eager/F32 answers. The upstream full build with
  CUDA 13.4/Windows also hit a `cicc` access violation. Not merged until
  numerical parity is understood.
- CPU int8 quantization / MKL. A candle `Q8_0` GEMM probe was only ~6% faster
  than F32 on a representative 2304×768×1024 layer and `Q4_0` was slower, so it
  was not integrated; MKL crates are not available in the offline cache.
- Laya weight license must be checked before redistribution. The repo does not bundle weights.

## Documentation

- [Design](docs/design.md)
- [Testing guide](docs/testing.md)
- [Laya benchmarks](docs/laya-benchmarks.md)
- [Model family benchmarks](docs/benchmarks.en.md)
- [Model manifest schema](docs/model-manifest.md)
- [ADRs](docs/adr/)
- [Glossary](GLOSSARY.md)

## License

Apache-2.0. Laya architecture/behavior references the official Laya repository and
`sys1` 0.0.3 (both Apache-2.0). No model weights are included.
