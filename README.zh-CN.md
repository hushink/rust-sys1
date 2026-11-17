# rust-sys1

[English](README.md)

类型安全的 System One 进程内推理库，以及 TypeSafe-compatible HTTP 服务。

`rust-sys1` 在本地运行开放决策模型，并严格实现官方
`POST /v1/systemone` / `GET /v1/models` 契约。v1 内置 Laya
（`english`、`multilingual`、`typed-decisions`），`laya-latest` 提供懒加载路由。

## 特性

- 官方 API 严格兼容：`POST /v1/systemone`、`GET /v1/models`；错误体使用官方 `detail` 形状。
- 进程内推理：typed `Questions` / `Answers`，不经过 HTTP 与序列化。
- candle 版 Laya：ModernBERT、decision head、校准、batch、两种 confidence。
- 模型源：本地路径、Hugging Face、ModelScope、静态 HTTP manifest 中心。
- 缓存：sha256、HTTP Range 断点续传、离线模式、并发拉取锁。
- 单进程服务：注册表、懒加载、LRU、keep_alive、`spawn_blocking` 推理。
- 运行时模型管理：无需重启即可加载、卸载、重载、添加和移除模型。
- 加速器 feature 骨架：CPU / CUDA / cuDNN / Metal / Accelerate。

## Workspace

| Crate | 职责 |
|---|---|
| [`sys1-protocol`](crates/sys1-protocol/README.zh-CN.md) | 官方 wire types、校验、typed questions/answers |
| [`sys1-core`](crates/sys1-core/README.zh-CN.md) | `DecisionModel`、注册表、manifest、进程内 API、设备选择 |
| [`sys1-hub`](crates/sys1-hub/README.zh-CN.md) | 模型源与缓存（Local / HF / ModelScope / HTTP） |
| [`sys1-laya`](crates/sys1-laya/README.zh-CN.md) | clean-room Laya 实现与 Router |
| [`sys1-server`](crates/sys1-server/README.zh-CN.md) | axum `/v1/*` 与 `/health` |
| [`sys1-cli`](crates/sys1-cli/README.zh-CN.md) | `sys1` 二进制：serve/pull/list/show/rm/run/doctor |
| [`sys1`](crates/sys1/README.zh-CN.md) | umbrella 进程内 API |

每个 crate 都提供英文 `README.md` 与中文 `README.zh-CN.md`。

## 快速开始

### 构建

```bash
cargo build --release -p sys1-cli
```

默认 CPU。CUDA / cuDNN / Metal：

```bash
cargo build --release -p sys1-cli --features cuda
cargo build --release -p sys1-cli --features cudnn
cargo build --release -p sys1-cli --features metal   # macOS
```

### Laya 目录结构

```text
model.safetensors
encoder/config.json
rl_agent_config.json
tokenizer/tokenizer.json
```

### 启动服务

```bash
# 单个 checkpoint
sys1 serve --model laya-multilingual=local:/path/to/laya-multilingual

# Laya 仓库根目录：自动注册 laya-english / laya-multilingual /
# laya-typed-decisions，并启用 laya-latest Router
sys1 serve --model laya-latest=local:/path/to/convaiinnovations/laya
```

默认绑定 `127.0.0.1:8080`。可选 Bearer：

```bash
sys1 serve --bind 0.0.0.0:8080 --api-key secret \
  --model laya-latest=local:/path/to/laya
```

默认启用跨请求 batching（`--batch-max 32 --batch-wait-ms 2`）。用 `--no-batch`
关闭，或调大 `--batch-max`（例如 64）以提升 GPU 高并发吞吐。CUDA 上
`--dtype f16` 是最高吞吐模式；`--tf32` 是可选的 F32 加速。CUDA 上 state ≥512
token 默认启用精确 local-window attention；`--no-local-window` 退回 eager。

### 运行时模型管理

CLI 在 `/v1` 之外暴露运维 API。默认开启，与 `/v1/*` 使用同一个 Bearer key；
可用 `--no-admin` 或 `[server] admin = false` 关闭。

| 方法 | 路径 | 含义 |
|---|---|---|
| `GET` | `/api/models` | 模型目录、别名、常驻状态与 LRU 限制 |
| `POST` | `/api/models/{name}/load` | 立即加载到内存 |
| `POST` | `/api/models/{name}/unload` | 释放常驻实例（幂等） |
| `POST` | `/api/models/{name}/reload` | 卸载后重新加载 |
| `POST` | `/api/models` | 解析 `ModelRef` 并加入运行时目录 |
| `DELETE` | `/api/models/{name}` | 卸载并从运行时目录移除 |

```bash
curl http://127.0.0.1:8080/api/models
curl -X POST http://127.0.0.1:8080/api/models/laya-latest/load
curl -X POST http://127.0.0.1:8080/api/models \
  -H 'Content-Type: application/json' \
  -d '{"reference":"hf:convaiinnovations/laya-multilingual"}'
curl -X POST http://127.0.0.1:8080/api/models/laya-multilingual/unload
curl -X DELETE http://127.0.0.1:8080/api/models/laya-multilingual
```

`POST /api/models` 接受与 `sys1 pull` 相同的引用，也接受 `sys1.toml` 中
`[models]` 的名称。运行时添加/移除的条目不会持久化：重启后以配置中的
`[models]` 为准。已经持有模型句柄的请求会正常完成，后续请求会重新触发懒加载。

### 配置

部署相关配置都可以放到 `sys1.toml`，代码里只保留内置默认值。优先级：
**CLI > 环境变量 > `sys1.toml` > 内置默认值**。使用 `--config <file>` 或
`SYS1_CONFIG`；当前目录存在 `sys1.toml` 时自动加载。示例见
[`sys1.toml.example`](sys1.toml.example)。

```toml
[hub]
cache_dir = "D:/models/sys1-cache"  # 模型下载/缓存目录
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

[models]
laya-latest = "local:D:/models/laya"       # 本地模型目录
# laya-multilingual = "hf:convaiinnovations/laya-multilingual"
```

配置文件中的相对路径按配置文件所在目录解析。`[models]` 中的名称可用于
`serve`、`run`、`pull`、`show`、`rm`。凭据建议通过 `HF_TOKEN`、
`MODELSCOPE_API_TOKEN`、`SYS1_HUB_TOKEN` 环境变量提供。

### 调用

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

### 进程内

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

## 兼容性说明

- 只实现官方端点；`/v1/decisions` 有意不实现。
- 校验失败 `422 {"detail": [...]}`；未知模型 `400 {"detail": "..."}`。
- `box` 问题与图片 `assets` 会被解析，但 Laya 返回 `422 unsupported`；缺少
  `TypeSafe-Danger-Zone-Images: true` 返回 `400`。
- 请求体上限 32 MiB；支持 gzip；当前构建对 zstd 返回 `415`。
- 超长 state 返回 `400 max_tokens_exceeded`，不会静默截断。
- `/v1/*` 默认 TypeSafe/ollaya 归一化 confidence；`--laya-confidence laya` 切换为
  Laya 原生熵置信度。
- `/health` 是非官方运维端点，位于 `/v1` 之外。

## 压测

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

### CPU vs CUDA（F32，不同 state）

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

### 并发扫描（noul，F32，不同 state）

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

### Rust loadgen（F16，闭环与开环）

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

### F16 高吞吐模式

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

### 长上下文 attention（CUDA）

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

### 选项数量扫描（F32，串行）

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

### cuDNN

本机可以链接 cuDNN（`--features cudnn`，并把 cuDNN 的 `lib` 目录加入 `LIB`）。
不同 state 下 32 并发约 211 vs 普通 CUDA 209 req/s，在运行波动范围内。Laya 的
ModernBERT 没有卷积算子，cuDNN 不会带来实质加速；实际相关的加速器是 CUDA。
只有未来模型包含卷积层时 cuDNN 才有意义。

### 复现

```powershell
# CPU：Laya 自动退出跨请求 batching
$env:SYS1_BENCH_MODEL="C:/path/to/laya-multilingual"
pwsh ./scripts/bench.ps1 -Device cpu -Dtype f32 -UniqueStates -OptionSweep -ConcurrencySweep

# CUDA：先在有 Visual Studio 环境的终端构建
cargo build --release --offline -p sys1-cli --features cuda --target-dir target-cuda
pwsh ./scripts/bench.ps1 -Exe target-cuda/release/sys1.exe `
  -Device cuda:0 -Dtype f32 -UniqueStates -ConcurrencySweep

# 高吞吐 F16，或 F32 + TF32
pwsh ./scripts/bench.ps1 -Exe target-cuda/release/sys1.exe `
  -Device cuda:0 -Dtype f16 -UniqueStates -ConcurrencySweep
pwsh ./scripts/bench.ps1 -Exe target-cuda/release/sys1.exe `
  -Device cuda:0 -Dtype f32 -Tf32 -UniqueStates

# 关闭 batching 的基线
pwsh ./scripts/bench.ps1 -Exe target-cuda/release/sys1.exe `
  -Device cuda:0 -Dtype f32 -UniqueStates -NoBatch
```

## 测试

```bash
cargo test --offline --workspace
```

Laya 官方 golden 测试默认忽略，需要本地权重与官方 golden vectors：

```powershell
$env:SYS1_LAYA_TEST_ROOT="C:/.../snapshots/<sha>"
$env:SYS1_LAYA_GOLDEN_DIR="D:/path/to/laya/laya-rust/tests/golden"
cargo test -p sys1-laya -- --ignored --nocapture

# 长上下文 local-window 对齐（CUDA，本地 checkpoint）
cargo test --release -p sys1-laya --features cuda --test long_context -- --ignored --nocapture
```

## 已知缺口 / Roadmap

- zstd 请求压缩、`sys1.toml` 配置优先级。
- ONNX Runtime / TensorRT Execution Provider（v1.1）。candle 无法直接承载 TensorRT；
  ORT TensorRT EP 是计划路径。
- long context 扫描、act head、abstention、parallel option layout（v1.1）。
- 长上下文 FlashAttention：用 hdim64-only candle-flash-attn + CUDA 13.3 的原型
  把 932 token 延迟从 ~181 ms 降到 ~69 ms，但答案与 eager/F32 不一致；上游完整
  构建在 CUDA 13.4/Windows 上还遇到 `cicc` access violation。数值对齐前不合并。
- CPU int8 量化 / MKL。用 candle `Q8_0` 对代表性 2304×768×1024 层做 GEMM 探测只比
  F32 快约 6%，`Q4_0` 更慢，因此未集成；MKL crate 不在离线缓存中。
- `laya-latest` Router 目前是启发式；完整官方 routing golden 对齐待补。
- HF / ModelScope 在线下载集成测试；GPU runtime CI。
- Laya 权重再分发前需确认许可证；仓库不包含权重。

## 文档

- [设计](docs/design.md)
- [ADR](docs/adr/)
- [术语表](GLOSSARY.md)

## License

Apache-2.0。Laya 架构与行为参考官方 Laya 仓库与 `sys1` 0.0.3（均为 Apache-2.0）；
本仓库不包含模型权重。
