# sys1-tools

[English](README.md)

rust-sys1 的 Python 工具集，使用 [uv](https://docs.astral.sh/uv/) 管理。
包含 ONNX 导出/校验、合成 ORT fixture 生成，以及原先位于
`scripts/bench.ps1` 的 HTTP 压测工具。

## 安装

在仓库根目录执行：

```bash
uv sync --all-extras
uv run sys1-bench --help
```

`uv` 会创建 `.venv` 与 `uv.lock`；需要 Python 3.10 或更新版本。每个工具都是独立
console script，在 `[project.scripts]` 中配置：

| Script | 用途 |
|---|---|
| `sys1-export-onnx` | 导出 checkpoint 为 ONNX |
| `sys1-check-onnx` | 与 golden 数据校验 logits 与 pooled 输出 |
| `sys1-generate-scoring-stub` | 重新生成合成 ORT fixture |
| `sys1-bench` | 运行 HTTP 压测 |

可选依赖按用途分组，只跑压测时不需要安装 ONNX：

| Extra | 内容 |
|---|---|
| `onnx` | `numpy`、`onnx`、`safetensors`，用于导出与 fixture 生成 |
| `check` | `numpy`、`onnxruntime`，用于 golden logits 校验 |
| `all` | 以上全部 |

## 命令

### 导出 Laya checkpoint 为 ONNX

```bash
uv run --extra onnx sys1-export-onnx \
  /path/to/laya-multilingual /path/to/laya-onnx
```

导出目录包含 `model.onnx`（带 `logits` 与 `pooled` 输出）、`onnx.json`、
`rl_agent_config.json`、tokenizer，以及 checkpoint 含 act head 时的
`act_head.safetensors`。`--skip-assets` 只写 `model.onnx`。

### 与官方 golden logits 校验

```bash
uv run --extra check sys1-check-onnx \
  /path/to/laya-onnx /path/to/golden/multilingual/case_dict_state.json
```

命令会校验 `pooled` 形状与 logits 最大误差 `|onnx - golden|`；
`--tolerance` 默认 `1e-4`。

### 重新生成合成 ORT fixture

```bash
uv run --extra onnx sys1-generate-scoring-stub
```

不传 `--output` 时写入仓库中的
`crates/sys1-ort/tests/fixtures/scoring_stub.onnx`。

### 运行 HTTP 压测

先构建 release CLI：

```bash
cargo build --release --offline -p sys1-cli
```

然后指定本地 checkpoint：

```bash
uv run sys1-bench --model-dir /path/to/laya-multilingual
uv run sys1-bench --model-dir /path/to/decima-small --model-name decima-small
uv run sys1-bench --model-dir ... --family qwen3guard --model-name guard
uv run sys1-bench --model-dir ... --concurrency-sweep --option-sweep
uv run sys1-bench --model-dir ... --json
```

工具会在 `127.0.0.1:18082` 启动 `sys1 serve`，等待 `/health`，运行与原
PowerShell 脚本相同的场景，打印 markdown 友好表格，并在退出时停止服务。
`--family generic` 发送 noul/score/choice 问题；`--family qwen3guard` 发送
Qwen3Guard 固定的 safety、unsafe 与 category 问题（任意选项会被拒绝，因此跳过
option sweep）。参数与原脚本对应：`--exe`、`--model-name`、`--device`、
`--dtype`、`--confidence`、`--warmup`、`--sequential`、
`--batch-sequential`、`--option-requests`、`--batch-max`、`--batch-wait-ms`、
`--no-batch`、`--no-concurrent`（跳过固定 x4/x8 场景，适合慢速 decoder）、
`--tf32`、`--unique-states`、`--json`。`SYS1_BENCH_MODEL` 与
`SYS1_BENCH_EXE` 提供相同默认值。

## 开发

```bash
uv run --package sys1-tools --group dev ruff check python/tools
uv run --package sys1-tools --group dev ruff format --check python/tools
```
