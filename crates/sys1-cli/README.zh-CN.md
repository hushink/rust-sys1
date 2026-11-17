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
--no-batch                 # 关闭跨请求 batching
--offline
--cache <dir>              # 或 SYS1_CACHE
```

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
