# Laya 压测

[English](laya-benchmarks.md)

Laya 的延迟、吞吐、batching 与长上下文测量。NLI、GLiClass、Von、Decima、
Qwen3Guard、Kev、Decision 与 Clef 的数据见
[benchmarks.md](benchmarks.md)。

2026-11-17，单机 release 构建。并发场景为每个请求发送不同的 state，避免
Laya 的相同请求去重（dedup）抬高数字。

GPU 默认启用跨请求 batching（`--batch-max 32 --batch-wait-ms 2`）。CPU 上的
Laya 通过 `DecisionModel::supports_cross_request_batching` 自动退出 batching：
CPU 单次前向已经使用线程池，把独立请求串行塞进一个 batch 反而降低吞吐。GPU
单请求只付 0.5 ms 的 yield grace period。

环境：

- CPU：Intel i5-12400F（6 核 / 12 线程），32 GiB RAM
- GPU：NVIDIA RTX 3060 12 GiB，驱动 617.42 / CUDA 13.4
- 模型：`laya-multilingual`（mmBERT-base，322M）
- 客户端：PowerShell `HttpClient` 与 Rust `loadgen` 示例
- 服务：GPU 每批 `spawn_blocking`，CPU 每请求 `spawn_blocking`

## CPU vs CUDA（F32，不同 state）

CPU 为无 batching；CUDA 使用默认 `--batch-max 32`。

| 场景 | CPU mean | CPU p95 | CPU req/s | CUDA mean | CUDA p95 | CUDA req/s |
|---|---:|---:|---:|---:|---:|---:|
| `GET /health` | 0.29 ms | 0.47 ms | 2534 | 0.31 ms | 0.54 ms | 2254 |
| `GET /v1/models` | 0.23 ms | 0.35 ms | 3879 | 0.28 ms | 0.43 ms | 3263 |
| `POST /v1/systemone`（noul） | 84.62 ms | 99.89 ms | 11.81 | 17.39 ms | 21.12 ms | 57.32 |
| `POST /v1/systemone`（noul+score+choice） | 199.72 ms | 214.56 ms | 5.01 | 21.26 ms | 24.06 ms | 46.96 |
| `POST /v1/systemone`（noul），4 并发 | 171.29 ms | 181.46 ms | 22.23 | 24.70 ms | 27.46 ms | 125.53 |
| `POST /v1/systemone`（noul），8 并发 | 316.50 ms | 348.79 ms | 23.92 | 35.06 ms | 56.99 ms | 163.96 |

CUDA F32 把单问题请求从 ~85 ms 降到 ~17 ms（~5x），三问题从 ~200 ms 降到
~21 ms（~9.4x）。不同 state 下，GPU batching 把 8 并发吞吐从无 batching 的
~36 req/s 提升到 ~164 req/s。并发场景运行间波动约 5–10%。

## 并发扫描（noul，F32，不同 state）

| 并发数 | CPU mean | CPU p95 | CPU req/s | CUDA mean | CUDA p95 | CUDA req/s |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 87.70 ms | 90.63 ms | 11.22 | 17.09 ms | 18.03 ms | 53.77 |
| 2 | 112.13 ms | 121.24 ms | 17.55 | 30.97 ms | 37.94 ms | 56.81 |
| 4 | 171.16 ms | 180.03 ms | 22.38 | 26.59 ms | 48.91 ms | 101.03 |
| 8 | 312.40 ms | 345.15 ms | 24.25 | 34.95 ms | 59.60 ms | 152.54 |
| 16 | 597.03 ms | 705.60 ms | 24.87 | 53.39 ms | 98.63 ms | 182.77 |
| 32 | 1108.97 ms | 1486.42 ms | 25.82 | 88.49 ms | 152.53 ms | 209.25 |

CPU 在 ~24–26 req/s 饱和；CUDA 在 32 并发达到 ~209 req/s。PowerShell 的闭环
worker 把 runspace 启动时间计入 wall time，会低估稳态吞吐。下面的 Rust
loadgen 没有该客户端开销，可测到更高并发和开环吞吐。

## Rust loadgen（F16，闭环与开环）

```powershell
cargo build --release --offline -p sys1-cli --example loadgen
target/release/examples/loadgen.exe --url http://127.0.0.1:8080 `
  --concurrency 32 --requests 256
target/release/examples/loadgen.exe --url http://127.0.0.1:8080 `
  --rate 300 --concurrency 128 --requests 1500
```

| 模式 | 并发 / 速率 | Mean | p95 | p99 | req/s | dropped |
|---|---:|---:|---:|---:|---:|---:|
| closed-loop | x8 | 33.46 ms | 39.60 ms | 86.06 ms | 238.34 | 0 |
| closed-loop | x32 | 101.39 ms | 134.38 ms | 135.72 ms | 314.00 | 0 |
| closed-loop | x64 | 184.85 ms | 215.08 ms | 215.79 ms | 334.97 | 0 |
| open-loop | 200 rps | 53.02 ms | 98.55 ms | 121.03 ms | 198.85 | 0 |
| open-loop | 300 rps | 138.76 ms | 348.37 ms | 385.37 ms | 294.58 | 12 |
| open-loop | 400 rps | 295.40 ms | 360.90 ms | 366.94 ms | 343.75 | 184 |

F16、`--batch-max 32`、不同 state。服务端闭环饱和约 ~345 req/s；
`--batch-max 64` + 256 in-flight 可到 ~359 req/s。开环 400 rps 超过 128
in-flight 上限，故有 dropped offer，但 HTTP errors 为 0。F32 约 250 req/s
（闭环 x64 249.7；开环 250 offered 243.9）。模型级 F16 batch32 上限约
396 req/s，HTTP、tokenization、batching 与 JSON 约占 10–15%。

## F16 高吞吐模式

`--dtype f16` 权重仍以 F32 加载，但 encoder 与 head 用 F16 计算。这是目前
GPU 上最大的剩余优化项：

| 场景 | F32 mean | F32 req/s | F16 mean | F16 req/s |
|---|---:|---:|---:|---:|
| noul，1 并发 | 17.39 ms | 57.32 | 19.59 ms | 50.91 |
| noul，4 并发 | 24.70 ms | 125.53 | 21.47 ms | 141.90 |
| noul，8 并发 | 35.06 ms | 163.96 | 33.28 ms | 183.55 |
| noul，16 并发 | 53.39 ms | 182.77 | 40.73 ms | 219.27 |
| noul，32 并发 | 88.49 ms | 209.25 | 70.47 ms | 248.47 |

三问题抽查中，F16 的概率变化最大约 0.0005（`choice` 0.6343、
`noul` 0.9712→0.9711、`score` 1.6681→1.6676）。需要最大数值保真时仍用 F32；
Ampere+ GPU 上追求吞吐可用 F16。BF16 速度接近但抽查精度略差。`--tf32` 为
F32 开启 TF32；32 并发约 217 vs 209 req/s（波动范围内），输出差异约 1e-4。

Laya 的 batch 预处理会对完全相同的请求去重。例如 8 个相同的 F32 请求在 CUDA
上约 21 ms / ~240 req/s；这是特定负载的缓存式收益，不代表通用 batching。

## 长上下文 attention（CUDA）

state ≥512 token 时，local 层默认使用精确的滑动窗口 attention：每个 query 只
收集 local mask 允许的 `±64` 个 key，按 128 query 分块。数学上等价于 eager；
F32 输出完全一致，F16 因舍入最多差约 `6e-4`。`--no-local-window` 可退回 eager；
CPU 始终使用 eager。

| noul tokens | Eager | Window | 加速 | 最大答案差 |
|---:|---:|---:|---:|---:|
| 266 | 27.00 ms | 27.35 ms | 0.99x（走 eager） | 0 |
| 518 | 87.20 ms | 75.18 ms | 1.16x | 6e-4 |
| 932 | 194.33 ms | 141.10 ms | 1.38x | 6e-4 |

932 token F32 为 184.92 → 142.06 ms（1.30x），答案差为 0。4 个并发的
932-token F16 请求：eager ~724 ms / 5.5 req/s，window ~433 ms / 8.3 req/s。

```powershell
$env:SYS1_LAYA_TEST_ROOT="C:/path/to/laya-multilingual"
$env:SYS1_LONG_DEVICE="cuda"
$env:SYS1_LONG_DTYPE="f16"
cargo run --release --offline -p sys1-laya --features cuda `
  --example long_context_bench

# HTTP 并发：生成长 state 后交给 loadgen
$state = ("I was charged twice for invoice 12345 and need a refund today. " * 50)
target/release/examples/loadgen.exe --url http://127.0.0.1:8080 `
  --concurrency 4 --requests 8 --state $state
```

FlashAttention 原型把 932 token 降到 ~69 ms（2.6x），但答案与 eager/F32 明显
不一致（noul 0.67 vs 0.98），因此仍未启用。超长 F32 state 建议把
`--batch-max` 降到 8–16 以限制 eager attention 显存。

## 选项数量扫描（F32，串行）

| 问题类型 | 选项数 | CPU mean | CUDA mean |
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

所有运行都返回了预期的标签数量，概率和为 1.00。CPU 延迟大致随选项数线性增长
（decision head 要处理更多 option marker）；CUDA 在 10 个选项以内基本保持
15–18 ms，20/40 个选项分别增加约 8/21 ms。

## cuDNN

本机可以链接 cuDNN（`--features cudnn`，并把 cuDNN 的 `lib` 目录加入 `LIB`）。
不同 state 下 32 并发约 211 vs 普通 CUDA 209 req/s，在运行波动范围内。Laya 的
ModernBERT 没有卷积算子，cuDNN 不会带来实质加速；实际相关的加速器是 CUDA。
只有未来模型包含卷积层时 cuDNN 才有意义。

## 复现

```powershell
# CPU：Laya 自动退出跨请求 batching
$env:SYS1_BENCH_MODEL="C:/path/to/laya-multilingual"
uv run sys1-bench --device cpu --dtype f32 --unique-states --option-sweep --concurrency-sweep

# CUDA：先在有 Visual Studio 环境的终端构建
cargo build --release --offline -p sys1-cli --features cuda --target-dir target-cuda
uv run sys1-bench --exe target-cuda/release/sys1.exe `
  --device cuda:0 --dtype f32 --unique-states --concurrency-sweep

# 高吞吐 F16，或 F32 + TF32
uv run sys1-bench --exe target-cuda/release/sys1.exe `
  --device cuda:0 --dtype f16 --unique-states --concurrency-sweep
uv run sys1-bench --exe target-cuda/release/sys1.exe `
  --device cuda:0 --dtype f32 --tf32 --unique-states

# 关闭 batching 的基线
uv run sys1-bench --exe target-cuda/release/sys1.exe `
  --device cuda:0 --dtype f32 --unique-states --no-batch
```
