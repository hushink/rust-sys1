# sys1-cli

[English](README.md)

`sys1` 二进制：模型 hub、单次推理与 TypeSafe-compatible 服务。

## 命令

| 命令 | 作用 |
|---|---|
| `sys1 serve` | 启动 HTTP 服务 |
| `sys1 pull <ref>` | 下载/物化模型 |
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

没有 `model.json` 的本地 Laya 目录可以直接使用：

```bash
sys1 run local:/path/to/laya-multilingual --input request.json
```

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
--batch-max 32             # 每次前向的最大请求数
--batch-wait-ms 2          # 收集 batch 的等待窗口
--max-attention-elements 67108864  # 按 attention 显存切分长 batch
--no-batch                 # 关闭跨请求 batching
--no-admin                 # 关闭 /api/models 管理端点
--metrics                  # 强制开启 /metrics 与 /api/metrics
--no-metrics               # 关闭 /metrics 与 /api/metrics
--preload <name>           # 启动前加载，可重复
--offline
--cache <dir>              # 或 SYS1_CACHE
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

```bash
curl http://127.0.0.1:8080/api/models
curl -X POST http://127.0.0.1:8080/api/models/laya-latest/load
curl -X POST http://127.0.0.1:8080/api/models \
  -H 'Content-Type: application/json' \
  -d '{"reference":"hf:convaiinnovations/laya-multilingual"}'
curl -X DELETE http://127.0.0.1:8080/api/models/laya-multilingual

# 磁盘缓存：先列出 key，再按 key 删除；常驻模型需先 unload，否则返回 409。
curl http://127.0.0.1:8080/api/cache
curl -X DELETE http://127.0.0.1:8080/api/cache/<key>
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

## 配置

`--config <file>` 或 `SYS1_CONFIG` 加载 TOML 配置；当前目录存在 `sys1.toml`
时自动加载。优先级：CLI > 环境变量 > 配置文件 > 内置默认值。

```toml
[hub]
cache_dir = "D:/models/sys1-cache"

[server]
device = "cuda:0"
dtype = "f16"
batch_max = 32
max_attention_elements = 67108864
metrics = true
preload = ["laya-latest"]

[models]
laya-latest = "local:D:/models/laya"
laya-multilingual = "hf:convaiinnovations/laya-multilingual"
```

完整示例见 [`sys1.toml.example`](../../sys1.toml.example)。相对 local 路径按
配置文件所在目录解析；`[models]` 中的名称也可用于 `pull`、`show`、`rm`、`run`。

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
```

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
```

## 压测

```powershell
$env:SYS1_BENCH_MODEL="C:/path/to/laya-multilingual"
pwsh ./scripts/bench.ps1
```

结果见根 README。
