# sys1-laya

[简体中文](README.zh-CN.md)

Clean-room Laya decision-model implementation in candle.

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

`TypeSafe` confidence is `(K * p_max - 1) / (K - 1)`; `LayaNative` is
`1 - H(p) / ln(K)`. `noul` answers never carry confidence, matching the official shape.

## Router

`LayaRouter::new(root, options)` expects the Laya repository root with
`multilingual/` and `typed-decisions/` subdirectories. It selects:

1. `typed-decisions` when the question-id set matches one of the four workflows;
2. `english` for mostly-ASCII text with English function words;
3. `multilingual` otherwise.

The router is heuristic in v1; full official routing golden parity is pending.
`LayaRouter::predict_batch` groups requests by selected checkpoint, so
`laya-latest` also benefits from server batching without mixing checkpoints.

## Parity

Golden tests compare `input_ids`, `markers`, logits, and answers against the
official Laya vectors. The three checkpoints match `case_dict_state` within 1e-3.

```bash
$env:SYS1_LAYA_TEST_ROOT="C:/.../snapshots/<sha>"
$env:SYS1_LAYA_GOLDEN_DIR="D:/path/to/laya/laya-rust/tests/golden"
cargo test --offline -p sys1-laya -- --ignored --nocapture
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

- Long-context scanning, act head, abstention, and parallel option layout are not implemented.
- Box questions are unsupported by Laya and return `ModelError::Unsupported`.
- Model weights are not bundled; check their license before redistribution.

## Features

`cuda`, `cudnn`, `metal`, `accelerate` forward to candle.

FlashAttention is not enabled: a prototype was ~2.6x faster on a 932-token
state but changed the decision outputs, and the upstream kernel build failed
with this CUDA 13.4/Windows toolchain. The eager attention path remains the
reference implementation.
