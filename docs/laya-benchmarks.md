# Laya benchmarks

[简体中文](laya-benchmarks.zh-CN.md)

Laya-specific latency, throughput, batching, and long-context measurements.
For the NLI, GLiClass, Von, Decima, Qwen3Guard, Kev, Decision, and Clef
families see [benchmarks.en.md](benchmarks.en.md).

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

## CPU vs CUDA (F32, distinct states)

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

## Concurrency sweep (noul, F32, distinct states)

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

## Rust load generator (F16, closed-loop and open-loop)

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

## F16 high-throughput mode

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

## Long-context attention (CUDA)

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

## Option-count sweep (F32, sequential)

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

## cuDNN

The cuDNN build links on this machine (`--features cudnn`, with the cuDNN `lib`
directory on `LIB`). With distinct states, 32-concurrent throughput is ~211 vs
~209 req/s for plain CUDA—within run-to-run noise. Laya's ModernBERT has no
convolution ops, so cuDNN adds no meaningful speedup; the CUDA build is the
relevant accelerator. cuDNN would only matter for future models with
convolutional layers.

## Reproduce

```powershell
# CPU: Laya automatically opts out of cross-request batching
$env:SYS1_BENCH_MODEL="C:/path/to/laya-multilingual"
uv run sys1-bench --device cpu --dtype f32 --unique-states --option-sweep --concurrency-sweep

# CUDA: build with the Visual Studio environment first
cargo build --release --offline -p sys1-cli --features cuda --target-dir target-cuda
uv run sys1-bench --exe target-cuda/release/sys1.exe `
  --device cuda:0 --dtype f32 --unique-states --concurrency-sweep

# High-throughput F16, or optional TF32 for F32
uv run sys1-bench --exe target-cuda/release/sys1.exe `
  --device cuda:0 --dtype f16 --unique-states --concurrency-sweep
uv run sys1-bench --exe target-cuda/release/sys1.exe `
  --device cuda:0 --dtype f32 --tf32 --unique-states

# Unbatched baseline for comparison
uv run sys1-bench --exe target-cuda/release/sys1.exe `
  --device cuda:0 --dtype f32 --unique-states --no-batch
```
