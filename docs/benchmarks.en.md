# Model family benchmarks (2026-11-18)

[简体中文](benchmarks.md)

HTTP benchmarks for NLI, GLiClass, Von, Decima (base / small), Qwen3Guard,
Kev and Decision on CPU and CUDA. Requests go through `sys1 serve`'s
`/v1/systemone`; the tool is `uv run sys1-bench`.

## Environment

| Item | Value |
|---|---|
| CPU | Intel64 Family 6 Model 151 (12 logical cores) |
| RAM | 31.8 GiB |
| GPU | NVIDIA GeForce RTX 3060 12 GiB |
| OS | Windows 11 Pro for Workstations |
| Build | `cargo build --release -p sys1-cli`; CUDA adds `--features cuda` |
| dtype | F32 on both devices (see "F16 and cuDNN") |
| Tool | `sys1-bench` from `python/tools`, `--confidence typesafe` |

Reproduce:

```bash
cargo build --release -p sys1-cli
cargo build --release -p sys1-cli --features cuda

uv run sys1-bench \
  --exe target/release/sys1.exe \
  --model-dir target/models/decima-small-pytorch \
  --model-name decima-small --device cpu --dtype f32 \
  --family generic --option-sweep --concurrency-sweep --json
```

Point `--exe` at `target-cuda/release/sys1.exe` and use `--device cuda` for
CUDA. Qwen3Guard needs `--family qwen3guard`; other models use
`--family generic`. `--no-concurrent` skips the fixed x4/x8 scenarios; it suits
Qwen3.5 hybrid models such as Kev/Decision.

## Single request and concurrency

`POST noul` carries one noul question; `batch` is noul+score+choice. `x8` is
eight concurrent clients with five requests each; RPS is wall-clock throughput
for the whole scenario.

| Model | Device | noul p50 (ms) | batch p50 (ms) | x8 p50 (ms) | x8 RPS |
|---|---:|---:|---:|---:|---:|
| NLI (ModernBERT) | CPU | 54.6 | 159.6 | 162.7 | 45.8 |
| NLI (ModernBERT) | CUDA | 18.6 | 23.0 | 225.2 | 33.7 |
| NLI (DeBERTa-v3-large) | CPU | 382.2 | 1532.2 | 1590.2 | 4.9 |
| NLI (DeBERTa-v3-large) | CUDA | 30.2 | 78.7 | 240.5 | 31.8 |
| GLiClass | CPU | 15.1 | 42.4 | 27.7 | 213.1 |
| GLiClass | CUDA | 18.7 | 12.4 | 116.8 | 62.1 |
| Von | CPU | 276.2 | 491.5 | 1684.4 | 3.9 |
| Von | CUDA | 32.6 | 39.3 | 263.3 | 28.9 |
| Decima-small | CPU | 53.1 | 179.4 | 114.5 | 62.7 |
| Decima-small | CUDA | 23.4 | 63.4 | 195.1 | 38.7 |
| Decima-base | CPU | 102.8 | 358.5 | 341.7 | 22.0 |
| Decima-base | CUDA | 36.6 | 122.0 | 444.0 | 17.5 |
| Qwen3Guard 0.6B | CPU | 1677.2 | 3164.8 | 7530.5 | 1.05 |
| Qwen3Guard 0.6B | CUDA | 95.3 | 179.1 | 659.8 | 11.7 |
| Kev 0.8B | CPU | 585.0 | 1627.6 | — | — |
| Kev 0.8B | CUDA | 54.2 | 148.6 | 573.4 | 13.4 |
| Kev 4B (BF16) | CUDA | 83.2 | 248.6 | 779.8 | 9.9 |
| Decision 0.8B | CPU | 1124.3 | 3287.9 | 4937.8 | 1.56 |
| Decision 0.8B | CUDA | 76.7 | 215.0 | 665.4 | 11.7 |

The Kev/Decision CUDA numbers include the depthwise-convolution optimization
(see below). Before it they were 4379 ms / 4404 ms, and the x8 scenario could
not finish in reasonable time.

## Option-count sweep

Three requests per option count (two for Decima-base); `choice` runs from 2 to
40 options and `score` from 2 to 10 levels. Cells are p50 (ms).

| Model | Device | choice 2 | choice 5 | choice 10 | choice 40 | score 2 | score 5 | score 10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Decima-small | CPU | 79.9 | 68.9 | 91.9 | 247.9 | 44.6 | 53.2 | 91.9 |
| Decima-small | CUDA | 28.9 | 28.0 | 28.9 | 36.2 | 23.9 | 21.7 | 20.6 |
| Decima-base | CPU | 139.6 | 139.2 | 233.1 | 900.7 | 100.1 | 136.6 | 177.9 |
| Decima-base | CUDA | 37.6 | 36.8 | 38.7 | 94.7 | 34.6 | 35.7 | 34.7 |
| Von | CPU | 162.9 | 168.9 | 207.0 | 521.9 | 134.5 | 165.0 | 206.3 |
| Von | CUDA | 24.6 | 25.4 | 26.2 | 42.2 | 26.0 | 24.9 | 25.3 |

Decima groups option rows by token length: equal-length options share one
encoder and scorer forward, while different lengths stay separate so padding
cannot change the encoder's local attention. Forty options still take 36 ms on
Decima-small (CUDA): the optimization reduces option encodings from 40 to a few
groups.

## F16 and cuDNN

CUDA builds use `--features cuda` and `--features cudnn`, both with dtype F16.
`noul` is single-request p50 (ms); `x8` is throughput (RPS) for eight
concurrent clients and 40 requests.

| Model | CUDA noul | CUDA x8 | cuDNN noul | cuDNN x8 |
|---|---:|---:|---:|---:|
| NLI (ModernBERT) | 17.5 | 36.6 | 18.4 | 36.8 |
| NLI (DeBERTa-v3-large) | 22.8 | 28.3 | — | — |
| GLiClass | 11.2 | 72.3 | 16.4 | 72.0 |
| Von | 25.0 | 27.9 | 26.2 | 27.8 |
| Decima-small | 18.1 | 39.8 | 20.4 | 39.7 |
| Decima-base | 39.4 | 16.9 | 34.7 | 17.2 |
| Qwen3Guard | 77.6 | 16.6 | 77.1 | 16.6 |
| Kev 0.8B | 54.3 | 13.1 | 51.9 | 13.2 |
| Decision 0.8B | 63.6 | 11.7 | 61.2 | 11.7 |

cuDNN gives Decima-base and Qwen3.5 decoders about 3–8% on single requests, but
slows small encoders such as GLiClass/Von/Decima-small by 2%–45%; x8 throughput
is essentially unchanged. The default CUDA build is therefore the right
choice; enable cuDNN only for large models or specific GEMM shapes. Building
cuDNN needs the directory holding `cudnn.lib` on `LIB`
(`D:\Applications\cudnn-windows\lib\x64` here) and its DLL directory on `PATH`.

## TF32

`--tf32` (or `[server] tf32 = true`) only affects F32 GEMMs on CUDA; F16/BF16
paths are unchanged. RTX 3060, F32, single request, no concurrency:

| Model | Scenario | F32 | F32 + TF32 |
|---|---|---:|---:|
| DeBERTa-v3 NLI | noul p50 | 33.4 ms | 32.6 ms |
| DeBERTa-v3 NLI | noul+score+choice p50 | 79.7 ms | 70.6 ms |
| Decima guard-small | noul p50 | 36.1 ms | 31.9 ms |
| Decima guard-small | noul+score+choice p50 | 105.5 ms | 120.3 ms |

A large encoder's multi-question request gains about 10%; the small encoder
stays within run-to-run noise. TF32 has a smaller mantissa than F32; keep
numerically sensitive comparisons on F32/F16 or BF16.

## Cross-request batching

`sys1 serve` merges concurrent requests for the same model into one forward by
default. CPU comparison:

| Model | Mode | noul p50 | x8 p50 | x8 RPS |
|---|---|---:|---:|---:|
| Von | merged batching | 276.2 | 1684.4 | 3.9 |
| Von | `--no-batch` | 270.6 | 1131.4 | 6.9 |
| Decima-small | merged batching | 53.1 | 114.5 | 62.7 |
| Decima-small | `--no-batch` | 44.6 | 92.6 | 77.8 |

For these small encoders, merging eight requests on CPU does not raise
throughput: the batch lengthens each forward and the server must run the batch
serially, while without merging the eight requests use multiple cores in
parallel. Decoders such as Qwen3Guard are different: batching plus CUDA raises
throughput sharply (1.05 RPS on CPU → 11.7 RPS on CUDA).

## Conclusions and follow-ups

- **Encoders**: single-request CUDA is usually 1.5–8× faster (GLiClass is the
  exception: 15.1 ms CPU, 18.7 ms CUDA); at eight concurrent clients,
  NLI/GLiClass/Decima-small achieve higher CPU throughput, while Von and
  Decima-base are still better on CUDA. Measure per model and concurrency
  before deploying.
- **Qwen3Guard (dense Qwen3)**: the largest GPU gain, 1.68 s → 95 ms per
  request.
- **Kev / Decision (Qwen3.5 hybrid)**: before optimization, CUDA took 4.4 s per
  request, 3–5× slower than CPU. Layer profiling showed the generic `conv1d` in
  the 18 DeltaNet layers cost about 254 ms each, 98% of the forward pass.
  Replacing it with four shifted multiply-adds brought each layer to 0.36 ms.
  Kev CUDA 4.38 s → **54 ms** (about 80×), Decision 4.40 s → **77 ms** (about
  57×), and CPU 785/1464 ms → 585/1124 ms. The Qwen3.5 family is now 10–15×
  faster on CUDA than CPU.
- **F16**: the CUDA F16 path exposed dtype and contiguity bugs in
  NLI/Von/GLiClass, now fixed (masks cast to the compute dtype; contiguous
  encoder/scorer inputs). See the F16 matrix above; encoders and Qwen3Guard
  beat F32, and Kev/Decision are another ~20% faster. cuDNN has no reliable
  gain.
- **Clef 9B**: about 18.8 GB of weights, beyond 12 GiB of VRAM; CPU F32 needs
  about 36 GB of RAM. See "Clef 9B verification" for the BF16 CPU results.

## DeBERTa-v3 NLI verification

`MoritzLaurer/deberta-v3-large-zeroshot-v2.0@cf44676c` (870 MB F16) matches the
Hugging Face SHA256 for `model.safetensors`. The new disentangled-attention
encoder agrees with transformers F32 within 1e-4 on choice/score/noul (tone
0.859564/0.140436, urgency 1.539309, noul 0.998042). CPU single request
382 ms; CUDA F16 23 ms, 28 RPS at eight concurrent clients.

## Decima v2 (guard-small) verification

`amyrmahdy/decima-guard-small@ac110248` uses the decima v2 layout: mmBERT-small
(384 hidden, 22 layers, 6 heads), `choice_fit: true`, `question_in_state: true`,
`max_state_tokens: 2048`, `max_choice_tokens: 64`, temperature 0.754586. The
checkpoint is a 562 MB encoder plus an 18 MB head.

Against the author's `decima@1b07634e` (v2.3.0, through Ollaya's independent
reference wrapper), F32:

| Question | Upstream log-probs | Rust |
|---|---|---|
| team choice | [−0.003964, −5.532571] | [−0.003963, −5.532570] |
| urgent noul | 0.414360 (P(yes)) | 0.414360 |
| anger score | [0.429440, 0.357271, 0.213289] | [0.429440, 0.357271, 0.213289] |
| long choice (triggers `choice_fit`) | billing 0.920987 | 0.920987 |

When the question exceeds 64 tokens, both sides render
`<question-word prefix> … <whole option>` (the option stays whole and the
question is shortened at word boundaries by binary search), matching upstream
`render.choice_texts`.

```bash
$env:SYS1_DECIMA_GUARD_SMALL_TEST_DIR="C:/path/to/decima-guard-small"
cargo test -p sys1-decima --test decima predicts_with_the_guard_small_checkpoint -- --ignored
```

## Kev 4B verification

`jaredpalmer/kev-4b` (124 MB adapter) and `Qwen/Qwen3.5-4B-Base` (two shards,
9.3 GB) match their Hugging Face SHA256 values. CPU F32 agrees with an
independent PyTorch LoRA-merge + pointer-head reference within 1e-6 (billing
0.900975, refund 0.853919, urgency 0.953128); BF16 on a 12 GB CUDA card is
within 0.003, 83 ms per request. The merged 4B BF16 weights fit in 12 GB.

## Kev 9B verification

`jaredpalmer/kev-9b@2629c06a` (173 MB adapter, 8.4 MB `head.pt`) and
`Qwen/Qwen3.5-9B-Base@68c46c4b` (four shards, about 19 GB) match their Hugging
Face SHA256 values. `head.pt` records temperature **2.297397** and base
revision `68c46c4b…`; the adapter has 248 LoRA pairs at merge scale 2.0.

The BF16 backbone does not fit 12 GiB of VRAM, so it runs BF16 on CPU like
Clef. Against an independent PyTorch reference (BF16 backbone, manual LoRA
merge, pointer head):

| Question | PyTorch BF16 | Rust BF16 |
|---|---|---|
| team | billing 0.976657 / tech 0.023343 | 0.978158 / 0.021842 |
| refund | 0.920695 | 0.924340 |
| urgency | [0.272992, 0.349913, 0.377095] | [0.267890, 0.349262, 0.382848] |
| urgency score | 1.104103 | 1.114958 |

Probabilities differ by ≤0.006 and every decision agrees; the spread is BF16
accumulation order. `predict` takes about **47 s** for three questions, with a
peak working set near 19–20 GB after loading.

```bash
$env:SYS1_KEV9B_TEST_DIR="C:/path/to/kev-9b"   # contains base/
cargo test --release -p sys1-kev --test kev predicts_with_the_9b_checkpoint \
  -- --ignored --nocapture --test-threads=1
```

## Clef 9B verification

The four BF16 shards of `Cloudflare/clef-flash` total about 18.8 GB and match
their Hugging Face LFS SHA256 values. The backbone does not fit 12 GiB of VRAM,
so verification uses CPU + BF16:

```bash
$env:SYS1_CLEF_TEST_DIR="C:/Users/hushink/AppData/Local/Temp/sys1-clef"
$env:SYS1_CLEF_DTYPE="bf16"
cargo test --release -p sys1-clef --test clef matches_the_recorded_triage_decisions \
  -- --ignored --nocapture --test-threads=1
```

Results:

- Loading takes about 20 s; the peak working set is 18–19 GB.
- One triage request (five questions) takes **52.1 s** in `predict` (CPU BF16,
  remeasured).
- Through `sys1 serve`'s HTTP `/v1/systemone`: 18.3 s for a single noul
  request and 26.0 s for noul+score+choice (short state; this proves the
  serving and batching paths).
- Decisions match Ollaya's recorded `clef-flash-fp16` integration corpus:
  `team=billing`, `refund/urgent/churn=true`, and `upset`'s argmax is the
  fourth level (index 3).
- This verification exposed and fixed two bugs only a real checkpoint triggers:
  a Linear over a 4096-wide vector must be promoted to 2-D first, and
  `residual_gate` needs a rank-0 tensor for sigmoid.
- **Logit-level parity**: against the authors' `joint_schema_model.py` (BF16,
  CPU), the Rust token row is byte-identical to upstream `encode_record` (179
  tokens for the team request). The largest probability difference is ≤0.033
  (BF16 accumulation order between candle and PyTorch), and every argmax
  agrees: team billing 0.9565 → 0.9874, triage refund 0.7873 → 0.7994, and the
  fourth `upset` level 0.7144 → 0.6867. Reproduce:

  ```bash
  cargo test --release -p sys1-clef --test clef matches_upstream_bf16_probabilities \
    -- --ignored --nocapture --test-threads=1
  ```

GPU deployment needs at least 24 GB of VRAM (about 18 GB of BF16 weights and a
peak near 22.8 GB for a 4096-token request); an RTX 3060 12 GB is not enough.
