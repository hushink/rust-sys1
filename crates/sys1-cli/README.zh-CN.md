# sys1-cli

[English](README.md)

`sys1` 二进制：模型 hub、单次推理与 TypeSafe-compatible 服务。

## 命令

| 命令 | 作用 |
|---|---|
| `sys1 serve` | 启动 HTTP 服务 |
| `sys1 pull <ref>` | 下载/物化模型（进度输出到 stderr） |
| `sys1 list` | 列出缓存模型 |
| `sys1 show <ref>` | 打印解析后的 manifest |
| `sys1 rm <ref>` | 删除缓存模型 |
| `sys1 run <ref> --input request.json` | 单次推理 |
| `sys1 doctor` | 打印构建/设备/缓存诊断 |

## 模型引用

- `local:/path/to/model` 或已存在的路径
- `hf:convaiinnovations/laya@main#multilingual`
- `modelscope:owner/name@master`
- `https://models.example.com/laya/model.json`

没有 `model.json` 的本地目录可以直接使用，模型族由目录内容自动识别：

```bash
sys1 run local:/path/to/laya-multilingual --input request.json
sys1 run local:/path/to/nli-modernbert-base --input request.json
sys1 run local:/path/to/gliclass-edge --input request.json
```

支持的 candle 模型族：

- **Laya**（`sys1-laya`）：`english`、`multilingual`、`typed-decisions`。
- **NLI**（`sys1-nli`）：`MoritzLaurer/ModernBERT-{base,large}-zeroshot-v2.0`。
- **GLiClass**（`sys1-gliclass`）：`knowledgator/gliclass-instruct-edge-v1.0`。
- **Qwen3Guard**（`sys1-qwen3guard`）：`Qwen/Qwen3Guard-Gen-0.6B`（预设问题）。
- **Von**（`sys1-von`）：`wfzyx/von`。
- **Decima**（`sys1-decima`）：`amyrmahdy/decima-base`。
- **Kev**（`sys1-kev`）：`jaredpalmer/kev-0.8b` 加对应 Qwen3.5 base。
- **Decision**（`sys1-decision`）：`vllm-sr/Decision-1.0-Eos-0.8B`。
- **Clef**（`sys1-clef`）：`Cloudflare/clef-flash`（纯文本，9B）。

每个模型族都可以从本地目录或 `hf:` 引用启动：

```bash
sys1 serve --model nli=hf:MoritzLaurer/ModernBERT-base-zeroshot-v2.0
sys1 serve --model gliclass=hf:knowledgator/gliclass-instruct-edge-v1.0
sys1 serve --model qwen3guard=hf:Qwen/Qwen3Guard-Gen-0.6B
sys1 serve --model von=hf:wfzyx/von
sys1 serve --model decima=hf:amyrmahdy/decima-base
sys1 serve --model decision=hf:vllm-sr/Decision-1.0-Eos-0.8B
sys1 serve --model clef=hf:Cloudflare/clef-flash
```

## ONNX 模型

CLI 默认启用 `onnx` feature。包含 `model.onnx`（或由 `onnx.json` 指定其他文件）、
`rl_agent_config.json`、tokenizer，以及可选 `act_head.safetensors` 的目录会被识别
为 Laya ONNX 导出：

```bash
sys1 serve --model laya-onnx=local:/path/to/laya-onnx
sys1 run local:/path/to/laya-onnx --input request.json
```

ONNX Runtime 在加载模型时动态加载。设置 `ORT_DYLIB_PATH` 指向 `onnxruntime`
共享库，或将其安装到平台加载路径。多 GB 导出建议使用较新的 ONNX Runtime。
计算图输入/输出契约、可选 `onnx.json` 字段与纯 NumPy 导出器见
[`sys1-laya`](../sys1-laya/README.zh-CN.md#onnx-runtime-后端)。带 act-head
sidecar 的导出同样会生成 `action.act_probability`，与 candle 后端一致。
CLI 的 `pull`/`show` 也接受 ONNX manifest（`backend = "onnx"`、`format = "onnx"`）。

## 启动服务

```bash
# 单个 checkpoint
sys1 serve --model laya-multilingual=local:/path/to/laya-multilingual

# 仓库根目录：自动注册三个变体 + 懒加载 laya-latest Router
sys1 serve --model laya-latest=local:/path/to/convaiinnovations/laya
```

常用参数：

```text
--bind 127.0.0.1:8080
--api-key <key>            # 或 SYS1_API_KEY
--device auto|cpu|cuda:N|metal
--dtype auto|f32|f16|bf16   # CUDA 高并发下 f16 更快
--tf32                     # 为 F32 CUDA GEMM 启用 TF32
--no-local-window          # local 层退回 eager attention
--max-len <tokens>
--laya-confidence typesafe|laya
--laya-router-default english|multilingual|typed-decisions  # 路由兜底 checkpoint
--laya-auto-task-detection # 启用 typed-decisions 工作流匹配
--laya-lang-guess <code>   # 路由级语言提示，如 en、ro
--batch-max 32             # 每次前向的最大请求数
--batch-wait-ms 2          # 收集 batch 的等待窗口
--max-attention-elements 67108864  # 按 attention 显存切分长 batch
--no-batch                 # 关闭跨请求 batching
--no-admin                 # 关闭 /api/models 管理端点
--metrics                  # 强制开启 /metrics 与 /api/metrics
--no-metrics               # 关闭 /metrics 与 /api/metrics
--preload <name>           # 启动前加载，可重复
--log-level <filter>       # 或 SYS1_LOG_LEVEL / RUST_LOG
--log-format text|json     # 或 SYS1_LOG_FORMAT
--log-file <file>          # 或 SYS1_LOG_FILE；追加到文件而非 stdout
--cors-origin <origin>     # 允许来源，可重复，覆盖配置
--no-cors                  # 关闭 CORS 头
--offline
--cache <dir>              # 或 SYS1_CACHE
--max-cache-bytes <bytes>  # 或 SYS1_MAX_CACHE_BYTES；0 = 不限制
```

## 模型管理 API

`sys1 serve` 在 `/v1` 之外提供运维 API，用于控制常驻状态和运行时目录。默认开启，
与 `/v1/*` 使用同一个 Bearer key；可用 `--no-admin` 或 `[server] admin = false`
关闭。

| 方法 | 路径 | 含义 |
|---|---|---|
| `GET` | `/api/models` | 目录、别名、常驻状态、LRU 限制 |
| `POST` | `/api/models/{name}/load` | 立即加载到内存 |
| `POST` | `/api/models/{name}/unload` | 释放常驻实例（幂等） |
| `POST` | `/api/models/{name}/reload` | 卸载后重新加载 |
| `POST` | `/api/models` | 解析引用并加入运行时目录 |
| `DELETE` | `/api/models/{name}` | 卸载并从运行时目录移除 |
| `GET` | `/api/cache` | 列出缓存模型、key、大小与文件 |
| `DELETE` | `/api/cache/{key}` | 从磁盘删除一个缓存条目 |
| `POST` | `/api/config/reload` | 重读 `sys1.toml` 并应用安全的模型变更 |

```bash
curl http://127.0.0.1:8080/api/models
curl -X POST http://127.0.0.1:8080/api/models/laya-latest/load
curl -X POST http://127.0.0.1:8080/api/models \
  -H 'Content-Type: application/json' \
  -d '{"reference":"hf:convaiinnovations/laya-multilingual"}'
curl -X DELETE http://127.0.0.1:8080/api/models/laya-multilingual

# 磁盘缓存：先列出 key，再按 key 删除；常驻模型或进行中的下载需先结束，否则返回 409。
curl http://127.0.0.1:8080/api/cache
curl -X DELETE http://127.0.0.1:8080/api/cache/<key>

# 修改 sys1.toml 的 [models] 后：
curl -X POST http://127.0.0.1:8080/api/config/reload
```

`GET /api/cache` 返回 `cache_dir`、`limit_bytes`、`used_bytes`、`free_bytes`、
`reclaimable_bytes`、`model_count`、`resident_count` 与 `models` 数组。每个模型
包含 `key`、`name`、`directory`、实际 `size_bytes`、`last_used_unix`、
`evictable` 与 `files`。

配置 `[hub] max_cache_bytes` 后，每次 pull 会先按 LRU 淘汰非驻留缓存条目，
直到新文件能够放入配额。本进程内的常驻模型与进行中的下载受保护；
`reclaimable_bytes` 表示当前可回收的空间。模型本身大于整个配额，或剩余空间被
常驻条目占满时，返回 `cache quota exceeded`。缓存命中也会执行配额检查。
`sys1 rm` 是显式删除，无法感知其他进程中的常驻模型；请先停止该进程或卸载模型。

```bash
# 删除一个缓存模型并查看当前用量
sys1 rm laya-multilingual
sys1 list
```

添加引用时会通过 hub 物化模型（已缓存则直接使用），并立即对
`/v1/systemone` 可见。加 `"load": true` 会在返回前加载主模型；加载失败会回滚
注册。`GET /api/models` 会标记每个条目是 `configured` 还是 `runtime`。
运行时添加/移除不会持久化；重启后以 `sys1.toml` 中的 `[models]` 为准。
已经持有模型句柄的请求会正常完成。

`--preload NAME`（或 `[server] preload`）在监听端口前加载指定模型，避免首个
请求承担冷加载；预加载失败会带模型名终止启动。

CLI 默认开启 `/metrics`（Prometheus）与 `/api/metrics`（JSON）；`--no-metrics`
关闭，`--metrics` 强制开启。配置 API key 时使用与 `/v1/*` 相同的 Bearer key。
JSON 包含请求/错误/延迟、batch 大小与耗时、模型加载/卸载、按模型请求数以及
当前常驻模型。预加载模型在 HTTP 指标启动前已常驻，因此不会增加模型加载计数。

`Ctrl-C` / `SIGTERM` 触发优雅关闭：停止接收新连接、等待在途请求结束再退出。
CORS 默认 permissive；用 `--cors-origin`（可重复）、`--no-cors` 或
`[server] cors_origins` 限制或关闭。

## 配置热加载

`POST /api/config/reload` 重读 `--config` 指定（或自动加载）的 `sys1.toml`，
仅应用 `[models]` 新增/移除：

- 新增模型经 hub 解析并注册为 `configured`。
- 移除模型从运行时目录卸载（磁盘缓存保留）。
- 修改 `[hub]`、`[server]` 或已有模型的引用返回 `409 restart required`；
  TOML 或模型解析失败返回 `400`。两种情况都不修改目录。
- 仅当 admin 开启且启动时加载了配置文件时，该端点才存在。

## 配置

`--config <file>` 或 `SYS1_CONFIG` 加载 TOML 配置；当前目录存在 `sys1.toml`
时自动加载。优先级：CLI > 环境变量 > 配置文件 > 内置默认值。

```toml
[hub]
cache_dir = "D:/models/sys1-cache"
max_cache_bytes = 34359738368   # 省略或 0 = 不限制

[server]
device = "cuda:0"
dtype = "f16"
batch_max = 32
max_attention_elements = 67108864
metrics = true
laya_router_default = "multilingual"   # english | multilingual | typed-decisions
laya_auto_task_detection = false       # 启用工作流匹配
# laya_lang_guess = "en"
preload = ["laya-latest"]
log_level = "info"
log_format = "text"          # text | json
# log_file = "logs/sys1.log"
# cors_origins = ["https://app.example"]  # [] 关闭 CORS；省略为 permissive

[models]
laya-latest = "local:D:/models/laya"
laya-multilingual = "hf:convaiinnovations/laya-multilingual"
```

完整示例见 [`sys1.toml.example`](../../sys1.toml.example)。相对 local 路径按
配置文件所在目录解析；`[models]` 中的名称也可用于 `pull`、`show`、`rm`、`run`。

## 日志与追踪

CLI 负责安装 `tracing` subscriber；各库只发事件，嵌入方可自行选择 subscriber。
CLI 支持 text/JSON、追加文件与过滤器：

```toml
[server]
log_level = "info"           # 也支持 sys1=debug,tower_http=info
log_format = "json"
log_file = "logs/sys1.log"   # 相对配置文件解析
```

优先级：CLI 参数 / `SYS1_LOG_LEVEL` / `SYS1_LOG_FORMAT` / `SYS1_LOG_FILE` >
过滤器使用 `RUST_LOG` > `[server]` > 默认。JSON 日志包含服务端 telemetry 事件
的 `request_id` 与 `trace_id` 字段。

`/v1/systemone` 接受 W3C `traceparent`，并在响应中返回新的 `traceparent`
（trace id 不变，server span id 重新生成）。所有响应仍带
`x-typesafe-request-id`；两个 id 会同时出现在日志与 admin 模型/缓存审计事件中。

## 压测工具

```bash
cargo build --release --offline -p sys1-cli --example loadgen
target/release/examples/loadgen --url http://127.0.0.1:8080 \
  --concurrency 64 --requests 512
target/release/examples/loadgen --url http://127.0.0.1:8080 \
  --rate 300 --concurrency 128 --requests 1500
```

闭环模式保持 `--concurrency` 个请求在途；`--rate` 是开环目标速率，拿不到
in-flight permit 的请求计入 `dropped`。

## Features

```bash
cargo build --release -p sys1-cli --features cuda
cargo build --release -p sys1-cli --features cudnn
cargo build --release -p sys1-cli --features metal
cargo build --release -p sys1-cli --features ort-cuda
cargo build --release -p sys1-cli --features ort-tensorrt
```

`onnx` 属于默认 features。若二进制不需要 ONNX Runtime，可用
`--no-default-features` 关闭。`ort-cuda`、`ort-tensorrt`、`ort-directml`、
`ort-coreml` 启用 ONNX Runtime 执行提供器，需要对应的 SDK/运行时。
`sys1 doctor` 输出编译 feature 与解析后的设备。

## 请求文件

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

# 扫描超过 checkpoint 上下文窗口的长文档
sys1 run local:/path/to/laya-multilingual --input long.json \
  --long --window 512 --stride 256

# 置信度低于阈值时标记 abstention
sys1 run local:/path/to/laya-multilingual --input request.json \
  --min-confidence 0.8
```

启用 `--long` 后，每个答案会增加 `window` 对象（`index`、`token_start`、
`token_end`、`count`），`usage.windows` 报告实际打分的窗口数。能放进一个窗口的
state 会走普通单次调用。`--stride` 不能超过有效窗口；`--long-batch` 限制每次
前向的窗口数。

checkpoint 或 ONNX 导出含 act head 时，每个答案会增加
`"action": {"act_probability": ...}`。启用 `--min-confidence` 后，每个答案还会
增加 `abstention`（`passed`、`abstained` 或 `unevaluated`）与
`abstention_threshold`；只有 abstained 的答案会出现 `low_confidence: true`。
Gate 比较的是答案最大概率，而不是报告中的 confidence；该选项不支持与 `--long`
同时使用。typed API 还支持按 bucket 设置阈值映射。

## 压测

```powershell
$env:SYS1_BENCH_MODEL="C:/path/to/laya-multilingual"
uv run sys1-bench
```

结果见根 README。
