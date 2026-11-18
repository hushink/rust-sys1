# sys1-laya

[简体中文](README.zh-CN.md)

Clean-room Laya decision-model implementation in candle, with an optional ONNX
Runtime backend. The ModernBERT encoder is shared with NLI, GLiClass, Von and
Decima through
`sys1-modernbert`.

## Checkpoints

- `english` — ModernBERT-large, 421M, 512 tokens
- `multilingual` — mmBERT-base, 322M, 1024 tokens
- `typed-decisions` — ModernBERT-large, 421M, 1024 tokens

A checkpoint directory contains:

```text
model.safetensors
encoder/config.json
rl_agent_config.json
tokenizer/tokenizer.json
```

## ONNX Runtime backend

Enable the `onnx` feature to run a Laya checkpoint exported to ONNX:

```bash
cargo build -p sys1-laya --features onnx
```

An ONNX model directory contains:

```text
model.onnx                 # or the file named by onnx.json
rl_agent_config.json
tokenizer/tokenizer.json   # or tokenizer.json
onnx.json                  # optional
act_head.safetensors       # optional act-head weights
```

`LayaOnnxModel::load_dir` shares tokenization, sequence layout, calibration,
and answer formatting with `LayaModel`; only the graph forward pass differs.
`LayaFactory` loads manifests with `backend = "onnx"` and `format = "onnx"`,
and `is_onnx_directory` / `is_model_directory` detect such directories.

### Graph contract

| Input | Type | Shape |
| --- | --- | --- |
| `input_ids` | int64 | `[batch, seq]` |
| `attention_mask` | int64 | `[batch, seq]` |
| `type_ids` | int64 | `[batch]` |
| `markers` | int64 | `[n]` |

`markers` holds flattened `row * seq + position` indexes into `input_ids`.
The graph applies the encoder, type embedding, decision head, marker gather,
and scorer, then returns one float32 logit per marker as `logits` with shape
`[n]`. Batch and sequence axes must be dynamic.

It may also return `pooled` with shape `[batch, hidden]`: token 0 after the
decision-head layers. When `act_head.safetensors` is present (or `act_head` is
configured), `LayaOnnxModel` requests `pooled` and computes
`action.act_probability` with the same act-head math as the candle backend.

`onnx.json` overrides defaults:

```json
{
  "model": "model.onnx",
  "max_batch": 8,
  "intra_threads": 4,
  "inter_threads": 1,
  "input_ids": "input_ids",
  "attention_mask": "attention_mask",
  "type_ids": "type_ids",
  "markers": "markers",
  "output": "logits",
  "pooled_output": "pooled",
  "act_head": "act_head.safetensors"
}
```

`max_batch` bounds unique items per ONNX run; larger requests are chunked.
`pooled_output` and `act_head` are optional; without an act-head sidecar the
backend returns an empty `actions` map.
ONNX Runtime is loaded dynamically at session creation. Set `ORT_DYLIB_PATH`
to the `onnxruntime` shared library, or put it on the platform loader path.
Builds never download runtime binaries. The binding targets ONNX Runtime API
17 and has been verified against 1.17.

Execution-provider features: `ort-cuda`, `ort-tensorrt`, `ort-directml`, and
`ort-coreml` (each implies `onnx`). TensorRT requires a local TensorRT SDK.
No official Laya ONNX artifact is published yet. The repository includes a
pure NumPy/ONNX exporter (no PyTorch) that implements the contract above:

```bash
uv run --extra onnx sys1-export-onnx \
  /path/to/laya-multilingual /path/to/laya-onnx

uv run --extra check sys1-check-onnx \
  /path/to/laya-onnx /path/to/golden/multilingual/case_dict_state.json
```

The exporter writes F32 weights (~1.2 GiB for multilingual, ~1.6 GiB for the
large checkpoints), copies the tokenizer and `rl_agent_config.json`, writes
`onnx.json`, and copies `act_head.safetensors` when the checkpoint has an act
head. Exported logits were checked against the official golden vectors for all
three checkpoints: max `|ONNX - golden|` was 1.7e-6 to 2.5e-6.

The Rust end-to-end parity test compares candle and ONNX answers, including
`action.act_probability` when both backends have an act head:

```powershell
$env:SYS1_LAYA_TEST_ROOT="C:/.../multilingual"
$env:SYS1_LAYA_ONNX_DIR="D:/models/laya-onnx"
$env:SYS1_LAYA_GOLDEN_DIR="D:/path/to/laya/laya-rust/tests/golden"
cargo test -p sys1-laya --features onnx --test onnx_parity -- --ignored --nocapture
```

For the multi-gigabyte exports, use a current ONNX Runtime build. On Windows,
ONNX Runtime 1.17 was unstable during session creation for the 1.2 GiB
multilingual graph; 1.32 loaded and ran it reliably. Set `ORT_DYLIB_PATH` to the
newer library. The synthetic graph tests still cover batching, padding, marker
indexing, and factory wiring.

## Use as a library

```toml
[dependencies]
sys1-laya = { git = "https://github.com/hushink/rust-sys1" }
sys1-core = { git = "https://github.com/hushink/rust-sys1" }
sys1-protocol = { git = "https://github.com/hushink/rust-sys1" }
# Or from a checkout:
# sys1-laya = { path = "../rust-sys1/crates/sys1-laya" }
```

Add `features = ["onnx"]` for the ONNX Runtime backend, or `cuda` / `cudnn` /
`metal` / `accelerate` for the matching candle backend. `sys1` also re-exports
`LayaModel`, `LayaOptions`, and `LayaOnnxModel` if you prefer the umbrella
crate.

## Example

```rust
use std::path::Path;
use sys1_core::{DecisionModel, Questions};
use sys1_laya::{ConfidenceMode, LayaModel, LayaOptions};
use sys1_protocol::State;

let model = LayaModel::load_dir(
    Path::new("/path/to/laya-multilingual"),
    LayaOptions {
        confidence_mode: ConfidenceMode::LayaNative,
        ..LayaOptions::default()
    },
)?;

let state = State::text("I was charged twice. Please refund me today.");
let questions = Questions::new()
    .score("urgency", ["Can wait", "This week", "Today"])
    .noul("billing", "Is this about billing?");

let output = model.predict(&state, &questions)?;
let urgency = output.answers.score("urgency")?;
```

## Options

- `device`: candle `Device` (CPU by default).
- `dtype`: `F32` by default; `F16` / `BF16` supported.
- `max_len`: override the checkpoint's context budget.
- `confidence_mode`: `TypeSafe` (default) or `LayaNative`.
- `local_window`: exact sliding-window attention for local layers on CUDA
  (default `true`, applied at ≥512 tokens); set `false` for eager attention.
- `max_attention_elements`: cap for `batch * heads * seq_len²` in one
  global-attention batch (default 64M elements); larger batches are split.
- `router` / `lang_guess`: see [Router](#router).
- Checkpoints that carry an `act_head.*` head also predict
  `action.act_probability`, exposed as `output.action(id)` and
  `output.actions`. The ONNX backend enables the same head when the export has
  an `act_head.safetensors` sidecar; older exports return an empty map.

`TypeSafe` confidence is `(K * p_max - 1) / (K - 1)`; `LayaNative` is
`1 - H(p) / ln(K)`. `noul` answers never carry confidence, matching the official shape.

## Long-context scanning

`DecisionModel::predict_long` tokenizes the state once, splits it into
overlapping token windows, scores them in shared forward passes, and aggregates
per question. `LayaModel`, `LayaOnnxModel`, and `LayaRouter` implement it; the
official wire API is unchanged.

```rust
use sys1_core::LongContextOptions;

let output = model.predict_long(
    &State::text(document),
    &questions,
    LongContextOptions {
        window: Some(512),   // default: max(64, max_len - head_max_len - 8)
        stride: Some(256),   // default: half the effective window
        batch_size: None,    // windows per forward pass
    },
)?;
let deciding = output.window("urgency").expect("window attribution");
assert_eq!(deciding.count, output.scored_windows);
```

- `noul` takes the largest P(true) across windows.
- `choice` / `score` take the answer from the single most-confident window, so
  a localized signal is not out-voted by neutral windows.
- Each answer's window records `index`, `token_start`, `token_end`, and
  `count`; `scored_windows` maps to the official `usage.windows`.
- A state that fits one window takes the ordinary `predict` path
  (`scored_windows = 1`, no per-answer window).
- `usage.input_tokens` sums every scored window, including overlap. The
  returned confidence belongs to the deciding window, not the whole document.

The English checkpoint was checked against official `predict_long` on the same
state, questions, window, and stride: answers, `action.act_probability`, window
attribution, and `usage.input_tokens` match.

## Abstention

`predict_with` applies the official `min_confidence` gate after prediction:

```rust
use sys1_core::{MinConfidence, PredictionOptions};

let output = model.predict_with(
    &state,
    &questions,
    &PredictionOptions {
        min_confidence: Some(MinConfidence::scalar(0.9)?),
    },
)?;
let gate = output.abstention("urgency").expect("gate ran");
```

- `answer_confidence` is the largest answer probability (`noul` is its P(true));
  if probabilities are absent it falls back to the reported confidence.
- A scalar applies to every answer. `MinConfidence::buckets` accepts
  `choice:2` / `choice:3-5` / `choice:6-10` / `choice:11+`, the same `score` and
  `noul` keys, and `default`; an unnamed bucket falls back to `default`, then
  to `0.0` (gate nothing).
- `AbstentionState` is `passed`, `abstained`, or `unevaluated` (the answer had
  no usable confidence). `low_confidence` is true only for `abstained`.
- The answer payload and the official wire response are unchanged. A threshold
  of `0.0` still runs the gate, so every evaluated answer reports `passed`.
- `predict_long` takes no `min_confidence`, matching the official API; the
  window loop has no single confidence to gate on.

## Router

`LayaRouter::new(root, options)` expects the Laya repository root with
`english/`, `multilingual/`, and `typed-decisions/` subdirectories. It ports the
official router's script detection, Latin stopword profiles, acronym/shouted
rules, mixed-segment handling, and Python-compatible reason strings. Selection
precedence is:

1. an explicit `model` hint (`RouteHints` / `route_with`);
2. an explicit `task` hint;
3. a typed-decisions workflow, only when `auto_task_detection` is enabled;
4. an explicit `lang` hint;
5. the per-call or router-wide `lang_guess`;
6. language analysis: `english` for English text, otherwise the `default`
   variant (`multilingual` unless configured).

`LayaRouterOptions { default, auto_task_detection }` mirrors the official
defaults (`multilingual`, `false`); `LayaOptions::lang_guess` sets the
router-wide hint. The official routing golden replays all 159 probe cases with
exact checkpoint, workflow, reason, and detection fields.

`LayaRouter::predict_batch` groups requests by selected checkpoint, so
`laya-latest` also benefits from server batching without mixing checkpoints.

## Factory

`LayaFactory` maps model names to checkpoint directories or router roots for
`ModelRegistry`. `register` / `register_router` take `&self`, so a running
server can add models; `unregister` drops a mapping. Registering an existing
name keeps the first registration — unregister first to replace it.

## Checkpoint helpers

Checkpoint layout knowledge lives here so composition layers do not have to
define Laya-prefixed APIs:

- `is_checkpoint_directory` — candle safetensors checkpoint layout
- `is_onnx_directory` / `is_model_directory` — ONNX export layout, or either backend
- `has_checkpoint_variants`
- `variant_directories` — the three catalog names and paths of a repository root
- `local_checkpoint_manifest` — manifest plus file map for a bare checkpoint
  directory without `model.json`
- `register_checkpoint_bundle` — registers a single checkpoint or a router
  bundle with the factory and returns the manifests to add to a registry

## Parity

Golden tests compare `input_ids`, `markers`, logits, answers, and
`action.act_probability` against the official Laya vectors. The three
checkpoints match `case_dict_state` within 1e-3. The routing test replays the
official probe file and requires exact model/workflow/reason/detection output.

```bash
$env:SYS1_LAYA_TEST_ROOT="C:/.../snapshots/<sha>"
$env:SYS1_LAYA_GOLDEN_DIR="D:/path/to/laya/laya-rust/tests/golden"
$env:SYS1_LAYA_ROUTING_GOLDEN="D:/path/to/laya/laya-rust/tests/golden/routing/route_probe.json"
cargo test --offline -p sys1-laya -- --ignored --nocapture
cargo test --offline -p sys1-laya --test routing_golden -- --nocapture
```

## Performance

On an i5-12400F + RTX 3060 with `laya-multilingual` F32 and distinct states,
a single noul request is ~85 ms on CPU vs ~17 ms on CUDA; a three-question
request is ~200 ms vs ~21 ms. CPU latency grows roughly linearly with
choice/score option count; CUDA stays near 15–18 ms up to 10 options.

CPU Laya opts out of server-side cross-request batching because independent
CPU forwards already use the thread pool; it saturates at ~24–26 req/s.
GPU batching raises 8-concurrent throughput to ~164 req/s and 32-concurrent
to ~209 req/s in the PowerShell bench. `--dtype f16` reaches ~248 req/s at
32 concurrent there, and the Rust `loadgen` example saturates at ~345 req/s
(F16, `--batch-max 32`), with probabilities within ~0.0005 in spot checks. See the root README for full
sweeps and reproduction commands.

For long states, local layers use exact sliding-window attention on CUDA at
≥512 tokens. A 932-token noul request drops from ~194 to ~141 ms on F16
(1.38x) and from ~185 to ~142 ms on F32 (1.30x); F32 answers match eager
exactly and F16 differs by at most ~`6e-4`. Four concurrent 932-token F16
requests improve from ~724 to ~433 ms. `long_context_bench` reproduces the
model-level table; `LayaOptions::local_window = false` (or CLI
`--no-local-window`) selects eager attention.

## Limitations

- Parallel option layout (`option_layout = "parallel"` checkpoints) is not implemented.
- Box questions are unsupported by Laya and return `ModelError::Unsupported`.
- Model weights are not bundled; check their license before redistribution.

## Features

`onnx` enables ONNX Runtime and `LayaOnnxModel`. `cuda`, `cudnn`, `metal`, and
`accelerate` forward to candle. `ort-cuda`, `ort-tensorrt`, `ort-directml`, and
`ort-coreml` enable ONNX Runtime execution providers.

FlashAttention is not enabled: a prototype was ~2.6x faster on a 932-token
state but changed the decision outputs, and the upstream kernel build failed
with this CUDA 13.4/Windows toolchain. The eager attention path remains the
reference implementation.
