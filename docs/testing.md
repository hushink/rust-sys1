# Testing rust-sys1

[简体中文](testing.zh-CN.md)

This document is the single reference for the test suite: what runs by default,
how to run the real-checkpoint parity tests, which environment variables they
need, and what CI executes. The root README keeps only a short summary.

## Quick start

```bash
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
```

The default suite needs no network, no model weights, and no GPU. Hub tests
start loopback HTTP servers; CLI tests create temporary caches. The few tests
that need external resources are `#[ignore]`d and are opt-in.

Tests can run offline once dependencies are cached:

```bash
cargo test --offline --workspace
```

## Test layers

| Layer | Location | What it covers |
|---|---|---|
| Wire contract | `crates/sys1-protocol/tests/wire.rs` | Official request/response examples, validation, limits, round-tripping |
| Traits and registry | `crates/sys1-core/tests/core.rs` | `DecisionModel`, registry LRU/keep-alive, resolution, abstention, device selection; deterministic JSON property tests |
| Hub | `crates/sys1-hub/tests/{hub,http,hf,modelscope}.rs` | Cache quota/LRU, static HTTP centers, Range resume, truncation retry, HF pagination/LFS, ModelScope listing, progress callbacks |
| Hub properties | `crates/sys1-hub/tests/properties.rs` | Cache keys stay filesystem-safe and stable; repo references round-trip |
| Laya unit | `crates/sys1-laya/tests/*.rs` | Tokenization, routing, calibration, ONNX wiring with synthetic graphs |
| Family units | each family crate | Config parsing, row layout, question validation, answer formatting |
| Server | `crates/sys1-server/tests/*.rs` | `/v1/*`, `/api/*`, metrics, graceful shutdown, CORS |
| CLI | `crates/sys1-cli/tests/config.rs` | Config precedence, model resolution, preload, logs, cache |
| Umbrella engine | `crates/sys1/tests/{engine,families,naming}.rs` | Engine registration/prediction, family dispatch, naming policy |
| Python tools | `python/tools` | Ruff lint/format, fixture generation, ONNX export/check, benchmark CLI |

## Real-checkpoint parity tests

These tests load local checkpoints and compare against upstream PyTorch,
`transformers`, the author's code, or recorded decisions. They are ignored by
default. Set the environment variable, then run the named test with `--ignored
--nocapture`. Use `--test-threads=1` for multi-gigabyte models.

### Laya

```powershell
$env:SYS1_LAYA_TEST_ROOT="C:/models/laya"            # repository root (english, multilingual, typed-decisions)
$env:SYS1_LAYA_GOLDEN_DIR="D:/laya/laya-rust/tests/golden"
$env:SYS1_LAYA_ROUTING_GOLDEN="D:/laya/laya-rust/tests/golden/routing/route_probe.json"
$env:SYS1_LAYA_ONNX_DIR="D:/models/laya-onnx"

cargo test -p sys1-laya --test parity matches_official_golden_answers -- --ignored --nocapture
cargo test -p sys1-laya --test routing_golden matches_the_official_routing_golden -- --nocapture
cargo test -p sys1-laya --test predict_long scans_a_long_state_and_attributes_every_answer -- --ignored --nocapture
cargo test -p sys1-laya --test abstention matches_the_official_scalar_and_bucket_gate -- --ignored --nocapture

# CUDA-only exact local-window check
cargo test --release -p sys1-laya --features cuda --test long_context \
  exact_local_window_matches_eager_on_long_state -- --ignored --nocapture

# Candle vs ONNX on the official golden case
cargo test -p sys1-laya --features onnx --test onnx_parity \
  candle_and_onnx_models_produce_the_same_answers -- --ignored --nocapture
```

`routing_golden` is not ignored; it skips when `SYS1_LAYA_ROUTING_GOLDEN` is
unset. The exporter's logits were within 2.5e-6 of the official golden vectors.

### NLI (ModernBERT and DeBERTa-v3)

```powershell
$env:SYS1_NLI_TEST_DIR="D:/models/nli-modernbert-base"
cargo test -p sys1-nli --test nli predicts_all_question_types -- --ignored --nocapture

$env:SYS1_NLI_DEBERTA_TEST_DIR="D:/models/deberta-v3-large-zeroshot-v2.0"
cargo test -p sys1-nli --test nli predicts_with_the_deberta_checkpoint -- --ignored --nocapture
```

Both are within 1e-4 of the upstream `transformers` F32 results.

### GLiClass

```powershell
$env:SYS1_GLICLASS_TEST_DIR="D:/models/gliclass-edge"
cargo test -p sys1-gliclass --test gliclass -- --ignored --nocapture
```

Choice, score, and noul probabilities match `gliclass==0.1.20` within 1e-4.

### Qwen backbones

```powershell
$env:SYS1_QWEN3_TEST_DIR="D:/models/Qwen3Guard-Gen-0.6B"
cargo test -p sys1-qwen --test qwen3_forward -- --ignored --nocapture

$env:SYS1_QWEN35_TEST_DIR="D:/models/qwen3.5-0.8b-base"
cargo test -p sys1-qwen --test qwen35_forward -- --ignored --nocapture
```

### Qwen3Guard

```powershell
$env:SYS1_QWEN3GUARD_TEST_DIR="D:/models/Qwen3Guard-Gen-0.6B"
cargo test -p sys1-qwen3guard --test qwen3guard -- --ignored --nocapture
```

Compares the preset `safety` / `unsafe` / `category` readouts with the
`transformers` reference.

### Von

```powershell
$env:SYS1_VON_TEST_DIR="D:/models/von"
cargo test -p sys1-von --test von -- --ignored --nocapture
```

Choice, noul, and score probabilities match `von-sdk==1.1.1`.

### Decima

Only one test per checkpoint; each uses a different variable:

```powershell
$env:SYS1_DECIMA_TEST_DIR="D:/models/decima-base"
cargo test -p sys1-decima --test decima predicts_all_question_types -- --ignored --nocapture

$env:SYS1_DECIMA_SMALL_TEST_DIR="D:/models/decima-small"
cargo test -p sys1-decima --test decima predicts_with_the_bert_encoder -- --ignored --nocapture

$env:SYS1_DECIMA_GUARD_SMALL_TEST_DIR="D:/models/decima-guard-small"
cargo test -p sys1-decima --test decima predicts_with_the_guard_small_checkpoint -- --ignored --nocapture
```

The v2 guard-small test covers `choice_fit` option rendering. Log-probabilities
match the author's `decima@1b07634e` within 1e-6.

### Kev

```powershell
$env:SYS1_KEV_TEST_DIR="D:/models/kev-0.8b"
$env:SYS1_KEV_BASE_DIR="D:/models/qwen3.5-0.8b-base"
cargo test -p sys1-kev --test kev predicts_all_question_types -- --ignored --nocapture

$env:SYS1_KEV4B_TEST_DIR="D:/models/kev-4b"          # contains base/
cargo test --release -p sys1-kev --test kev predicts_with_the_4b_checkpoint -- --ignored --nocapture

$env:SYS1_KEV9B_TEST_DIR="D:/models/kev-9b"          # contains base/; about 19 GB RAM
cargo test --release -p sys1-kev --test kev predicts_with_the_9b_checkpoint \
  -- --ignored --nocapture --test-threads=1
```

The 4B F32 path matches the independent PyTorch reference within 1e-6; BF16 on
a 12 GB CUDA card is within 0.003. The 9B CPU BF16 result is within 0.006.

### Decision

```powershell
$env:SYS1_DECISION_TEST_DIR="D:/models/Decision-1.0-Eos-0.8B"
cargo test -p sys1-decision --test decision -- --ignored --nocapture
```

### Clef

```powershell
$env:SYS1_CLEF_TEST_DIR="D:/models/clef-flash"
$env:SYS1_CLEF_DTYPE="bf16"                          # the 9B checkpoint needs BF16 and about 19 GB RAM
cargo test --release -p sys1-clef --test clef matches_the_recorded_triage_decisions \
  -- --ignored --nocapture --test-threads=1

cargo test --release -p sys1-clef --test clef matches_upstream_bf16_probabilities \
  -- --ignored --nocapture --test-threads=1
```

The token row is byte-identical to upstream `encode_record`; probabilities stay
within 0.033 of the authors' `joint_schema_model.py` in BF16.

### Umbrella engine

```powershell
$env:SYS1_LAYA_TEST_ROOT="C:/models/laya-multilingual"
cargo test -p sys1 --test engine -- --ignored --nocapture

# Dispatches every configured family through the engine
$env:SYS1_NLI_TEST_DIR="D:/models/nli-modernbert-base"
$env:SYS1_GLICLASS_TEST_DIR="D:/models/gliclass-edge"
$env:SYS1_QWEN3GUARD_TEST_DIR="D:/models/Qwen3Guard-Gen-0.6B"
$env:SYS1_VON_TEST_DIR="D:/models/von"
$env:SYS1_DECIMA_TEST_DIR="D:/models/decima-base"
$env:SYS1_KEV_TEST_DIR="D:/models/kev-0.8b"
$env:SYS1_DECISION_TEST_DIR="D:/models/Decision-1.0-Eos-0.8B"
cargo test -p sys1 --test families -- --ignored --nocapture
```

`tests/naming.rs` always runs and enforces that no crate except `sys1-laya`
defines Laya-prefixed items.

## Environment variables

| Variable | Used by | Meaning |
|---|---|---|
| `SYS1_LAYA_TEST_ROOT` | `sys1-laya`, `sys1` | Laya repository root or one checkpoint directory |
| `SYS1_LAYA_GOLDEN_DIR` | `sys1-laya` | Official golden answer vectors |
| `SYS1_LAYA_ROUTING_GOLDEN` | `sys1-laya` | Official routing probe JSON |
| `SYS1_LAYA_ONNX_DIR` | `sys1-laya` | ONNX export directory for candle/ONNX parity |
| `SYS1_NLI_TEST_DIR` | `sys1-nli`, `sys1` | ModernBERT NLI checkpoint |
| `SYS1_NLI_DEBERTA_TEST_DIR` | `sys1-nli` | DeBERTa-v3 NLI checkpoint |
| `SYS1_GLICLASS_TEST_DIR` | `sys1-gliclass`, `sys1` | GLiClass edge checkpoint |
| `SYS1_QWEN3_TEST_DIR` | `sys1-qwen` | Dense Qwen3 checkpoint |
| `SYS1_QWEN35_TEST_DIR` | `sys1-qwen` | Qwen3.5 checkpoint |
| `SYS1_QWEN3GUARD_TEST_DIR` | `sys1-qwen3guard`, `sys1` | Qwen3Guard-Gen checkpoint |
| `SYS1_VON_TEST_DIR` | `sys1-von`, `sys1` | Von checkpoint |
| `SYS1_DECIMA_TEST_DIR` | `sys1-decima`, `sys1` | Decima base repository |
| `SYS1_DECIMA_SMALL_TEST_DIR` | `sys1-decima` | Decima small (BERT encoder) |
| `SYS1_DECIMA_GUARD_SMALL_TEST_DIR` | `sys1-decima` | Decima v2 guard-small |
| `SYS1_KEV_TEST_DIR` | `sys1-kev`, `sys1` | Kev 0.8B adapter directory |
| `SYS1_KEV_BASE_DIR` | `sys1-kev` | Qwen3.5 base for the 0.8B test |
| `SYS1_KEV4B_TEST_DIR` | `sys1-kev` | Kev 4B directory containing `base/` |
| `SYS1_KEV9B_TEST_DIR` | `sys1-kev` | Kev 9B directory containing `base/` |
| `SYS1_DECISION_TEST_DIR` | `sys1-decision`, `sys1` | Decision 1.0 checkpoint |
| `SYS1_CLEF_TEST_DIR` | `sys1-clef` | Clef flash checkpoint |
| `SYS1_CLEF_DTYPE` | `sys1-clef` | `bf16` for the 9B checkpoint (default `f32`) |
| `SYS1_LONG_DEVICE` / `SYS1_LONG_DTYPE` | `sys1-laya` example | Long-context benchmark device/dtype |
| `SYS1_LONG_REPEATS` | `sys1-laya` example | Long-context benchmark repetitions |
| `SYS1_BATCH_DEVICE` / `SYS1_BATCH_DTYPE` | `sys1-laya` example | Batch benchmark device/dtype |
| `SYS1_REQUIRE_ORT` | `sys1-ort`, `sys1-laya` | Fail instead of skipping when ONNX Runtime is absent |
| `ORT_DYLIB_PATH` | `sys1-ort`, `sys1-laya` | ONNX Runtime shared library |
| `SYS1_LIVE_HUB` | `sys1-hub` | Run live Hugging Face/ModelScope manifest tests |
| `HF_ENDPOINT` | `sys1-hub` | Hugging Face mirror for live tests |
| `HF_TOKEN` / `MODELSCOPE_API_TOKEN` / `SYS1_HUB_TOKEN` | `sys1-hub` | Credentials for private/live tests |

## CI

`.github/workflows/ci.yml` runs on every push to `main` (and pull requests):

| Job | Runs |
|---|---|
| `Test (ubuntu-latest)` | `cargo fmt --all -- --check`, `cargo clippy --workspace --all-targets -- -D warnings`, `cargo test --workspace` |
| `Test (windows-latest)` | same |
| `Test (macos-latest)` | same |
| `Python tools` | `uv sync --all-extras`, Ruff check/format on `python/tools`, `uv run sys1-generate-scoring-stub`, then `git diff --exit-code` |

The hosted jobs do not download model weights, so the ignored parity tests are
not part of CI.

`.github/workflows/gpu.yml` is manual or path-triggered and runs on a
self-hosted Linux runner labeled `linux`, `gpu`, `cuda`:

- `cargo test --release -p sys1-laya --features cuda --test long_context -- --ignored`;
- builds `sys1-cli` with `cuda` and runs `examples/loadgen` against a served
  Laya checkpoint.

It requires the repository variable `SYS1_LAYA_TEST_ROOT` pointing at a cached
multilingual checkpoint. The TensorRT execution provider still needs a runner
with TensorRT installed; see the roadmap in the root README.

## Python tooling and benchmarks

```bash
uv sync --all-extras
uv run --package sys1-tools --group dev ruff check python/tools
uv run --package sys1-tools --group dev ruff format --check python/tools
uv run sys1-generate-scoring-stub
git diff --exit-code
```

`sys1-bench` is a performance tool, not a correctness test. It starts the
release CLI, waits for `/health`, and records latency/throughput; see
[benchmarks.md](benchmarks.md). ONNX checks:

```bash
uv run --extra onnx sys1-export-onnx /path/to/laya /path/to/laya-onnx
uv run --extra check sys1-check-onnx /path/to/laya-onnx /path/to/golden.json
```

## Adding a parity test

1. Put the test in the family crate's `tests/` directory.
2. Mark it `#[ignore = "requires SYS1_<FAMILY>_TEST_DIR ..."]` and read the
   checkpoint path from an environment variable.
3. Run it with `--ignored --nocapture`; use `--test-threads=1` when the model
   needs many gigabytes.
4. Record golden values from an independent implementation (upstream Python or
   the author's code), not from the Rust implementation itself.
5. Update the table in this document and the family README.
6. If the test is cheap and checkpoint-free, leave it enabled so hosted CI
   covers it.

## Troubleshooting

- **ONNX tests skip**: set `ORT_DYLIB_PATH` to the runtime and
  `SYS1_REQUIRE_ORT=1` to turn a missing runtime into a failure.
- **CUDA build fails on Windows**: pass the MSVC compiler to NVCC with
  `$env:NVCC_CCBIN="C:/Program Files/Microsoft Visual Studio/.../cl.exe"`.
- **Live hub tests time out**: set `HF_ENDPOINT=https://hf-mirror.com`, or keep
  them disabled; loopback hub tests always run.
- **Large-model OOM**: Kev 9B and Clef 9B need about 19 GB of RAM in BF16; a
  12 GB CUDA card cannot hold them. Run those tests on CPU with
  `--release --test-threads=1`.
- **PowerShell environment syntax**: use `$env:NAME="value"`; it applies only to
  the current shell.
