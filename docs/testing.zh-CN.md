# rust-sys1 测试指南

[English](testing.md)

本文是测试体系的统一说明：默认运行哪些测试、如何运行真实 checkpoint 的
parity 测试、需要哪些环境变量，以及 CI 执行什么。根目录 README 只保留简短
摘要和本文链接。

## 快速开始

```bash
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
```

默认测试不需要网络、模型权重或 GPU。hub 测试会启动本地回环 HTTP 服务，CLI
测试创建临时缓存。需要外部资源的少数测试标记为 `#[ignore]`，需要显式运行。

依赖已缓存后可离线运行：

```bash
cargo test --offline --workspace
```

## 测试分层

| 层 | 位置 | 覆盖内容 |
|---|---|---|
| Wire 契约 | `crates/sys1-protocol/tests/wire.rs` | 官方请求/响应示例、校验、限制、往返 |
| Trait 与注册表 | `crates/sys1-core/tests/core.rs` | `DecisionModel`、registry LRU/keep-alive、解析、abstention、设备选择；确定性 JSON 属性测试 |
| Hub | `crates/sys1-hub/tests/{hub,http,hf,modelscope}.rs` | 缓存配额/LRU、静态 HTTP 中心、Range 续传、截断重试、HF 分页/LFS、ModelScope 列表、进度回调 |
| Hub 属性 | `crates/sys1-hub/tests/properties.rs` | cache key 文件系统安全且稳定；repo 引用往返 |
| Laya 单元 | `crates/sys1-laya/tests/*.rs` | 分词、路由、校准、合成图 ONNX 接线 |
| Family 单元 | 各 family crate | 配置解析、行布局、问题校验、答案格式化 |
| Server | `crates/sys1-server/tests/*.rs` | `/v1/*`、`/api/*`、指标、优雅关闭、CORS |
| CLI | `crates/sys1-cli/tests/config.rs` | 配置优先级、模型解析、preload、日志、缓存 |
| Umbrella engine | `crates/sys1/tests/{engine,families,naming}.rs` | engine 注册/推理、family 分发、命名策略 |
| Python 工具 | `python/tools` | Ruff lint/format、fixture 生成、ONNX 导出/校验、benchmark CLI |

## 真实 checkpoint parity 测试

这些测试加载本地 checkpoint，并与上游 PyTorch、`transformers`、作者代码或
记录决策对比。默认忽略。设置环境变量后，用 `--ignored --nocapture` 运行指定
测试；多 GB 模型建议加 `--test-threads=1`。

### Laya

```powershell
$env:SYS1_LAYA_TEST_ROOT="C:/models/laya"            # 仓库根目录（english、multilingual、typed-decisions）
$env:SYS1_LAYA_GOLDEN_DIR="D:/laya/laya-rust/tests/golden"
$env:SYS1_LAYA_ROUTING_GOLDEN="D:/laya/laya-rust/tests/golden/routing/route_probe.json"
$env:SYS1_LAYA_ONNX_DIR="D:/models/laya-onnx"

cargo test -p sys1-laya --test parity matches_official_golden_answers -- --ignored --nocapture
cargo test -p sys1-laya --test routing_golden matches_the_official_routing_golden -- --nocapture
cargo test -p sys1-laya --test predict_long scans_a_long_state_and_attributes_every_answer -- --ignored --nocapture
cargo test -p sys1-laya --test abstention matches_the_official_scalar_and_bucket_gate -- --ignored --nocapture

# 仅 CUDA：精确 local-window 校验
cargo test --release -p sys1-laya --features cuda --test long_context \
  exact_local_window_matches_eager_on_long_state -- --ignored --nocapture

# 官方 golden case 上的 candle vs ONNX
cargo test -p sys1-laya --features onnx --test onnx_parity \
  candle_and_onnx_models_produce_the_same_answers -- --ignored --nocapture
```

`routing_golden` 未标记 ignore；未设置 `SYS1_LAYA_ROUTING_GOLDEN` 时会打印
跳过信息。导出器 logits 与官方 golden 向量最大误差 2.5e-6。

### NLI（ModernBERT 与 DeBERTa-v3）

```powershell
$env:SYS1_NLI_TEST_DIR="D:/models/nli-modernbert-base"
cargo test -p sys1-nli --test nli predicts_all_question_types -- --ignored --nocapture

$env:SYS1_NLI_DEBERTA_TEST_DIR="D:/models/deberta-v3-large-zeroshot-v2.0"
cargo test -p sys1-nli --test nli predicts_with_the_deberta_checkpoint -- --ignored --nocapture
```

两者与 `transformers` F32 结果的误差均在 1e-4 内。

### GLiClass

```powershell
$env:SYS1_GLICLASS_TEST_DIR="D:/models/gliclass-edge"
cargo test -p sys1-gliclass --test gliclass -- --ignored --nocapture
```

choice、score、noul 概率与 `gliclass==0.1.20` 的误差在 1e-4 内。

### Qwen 骨干

```powershell
$env:SYS1_QWEN3_TEST_DIR="D:/models/Qwen3Guard-Gen-0.6B"
cargo test -p sys1-qwen --test qwen3_forward -- --ignored --nocapture

$env:SYS1_QWEN35_TEST_DIR="D:/models/qwen3.5-0.8b-base"
cargo test -p sys1-qwen --test qwen35_forward -- --ignored --nocapture
```

### Qwen3Guard

```powershell
$env:SYS1_QWEN3GUARD_TEST_DIR="D:/models/Qwen3Guard-Gen-0.6B"
cargo test -p sys1-qwen3guard --test qwen3guard -- --ignored --nocapture
```

将固定预设 `safety` / `unsafe` / `category` 与 `transformers` 参考对比。

### Von

```powershell
$env:SYS1_VON_TEST_DIR="D:/models/von"
cargo test -p sys1-von --test von -- --ignored --nocapture
```

choice、noul、score 概率与 `von-sdk==1.1.1` 一致。

### Decima

每个 checkpoint 对应一个测试，使用不同变量：

```powershell
$env:SYS1_DECIMA_TEST_DIR="D:/models/decima-base"
cargo test -p sys1-decima --test decima predicts_all_question_types -- --ignored --nocapture

$env:SYS1_DECIMA_SMALL_TEST_DIR="D:/models/decima-small"
cargo test -p sys1-decima --test decima predicts_with_the_bert_encoder -- --ignored --nocapture

$env:SYS1_DECIMA_GUARD_SMALL_TEST_DIR="D:/models/decima-guard-small"
cargo test -p sys1-decima --test decima predicts_with_the_guard_small_checkpoint -- --ignored --nocapture
```

v2 guard-small 测试覆盖 `choice_fit` 选项渲染。log-probs 与作者
`decima@1b07634e` 的误差在 1e-6 内。

### Kev

```powershell
$env:SYS1_KEV_TEST_DIR="D:/models/kev-0.8b"
$env:SYS1_KEV_BASE_DIR="D:/models/qwen3.5-0.8b-base"
cargo test -p sys1-kev --test kev predicts_all_question_types -- --ignored --nocapture

$env:SYS1_KEV4B_TEST_DIR="D:/models/kev-4b"          # 含 base/
cargo test --release -p sys1-kev --test kev predicts_with_the_4b_checkpoint -- --ignored --nocapture

$env:SYS1_KEV9B_TEST_DIR="D:/models/kev-9b"          # 含 base/；约 19 GB 内存
cargo test --release -p sys1-kev --test kev predicts_with_the_9b_checkpoint \
  -- --ignored --nocapture --test-threads=1
```

4B 的 F32 路径与独立 PyTorch 参考在 1e-6 内一致；12 GB CUDA 卡上的 BF16
误差在 0.003 内。9B CPU BF16 的误差在 0.006 内。

### Decision

```powershell
$env:SYS1_DECISION_TEST_DIR="D:/models/Decision-1.0-Eos-0.8B"
cargo test -p sys1-decision --test decision -- --ignored --nocapture
```

### Clef

```powershell
$env:SYS1_CLEF_TEST_DIR="D:/models/clef-flash"
$env:SYS1_CLEF_DTYPE="bf16"                          # 9B checkpoint 需要 BF16 与约 19 GB 内存
cargo test --release -p sys1-clef --test clef matches_the_recorded_triage_decisions \
  -- --ignored --nocapture --test-threads=1

cargo test --release -p sys1-clef --test clef matches_upstream_bf16_probabilities \
  -- --ignored --nocapture --test-threads=1
```

输入 token 行与上游 `encode_record` 完全一致；BF16 概率与作者
`joint_schema_model.py` 的最大偏差为 0.033。

### Umbrella engine

```powershell
$env:SYS1_LAYA_TEST_ROOT="C:/models/laya-multilingual"
cargo test -p sys1 --test engine -- --ignored --nocapture

# 通过 engine 分发所有已配置 family
$env:SYS1_NLI_TEST_DIR="D:/models/nli-modernbert-base"
$env:SYS1_GLICLASS_TEST_DIR="D:/models/gliclass-edge"
$env:SYS1_QWEN3GUARD_TEST_DIR="D:/models/Qwen3Guard-Gen-0.6B"
$env:SYS1_VON_TEST_DIR="D:/models/von"
$env:SYS1_DECIMA_TEST_DIR="D:/models/decima-base"
$env:SYS1_KEV_TEST_DIR="D:/models/kev-0.8b"
$env:SYS1_DECISION_TEST_DIR="D:/models/Decision-1.0-Eos-0.8B"
cargo test -p sys1 --test families -- --ignored --nocapture
```

`tests/naming.rs` 始终运行，确保除 `sys1-laya` 外的 crate 不定义 Laya 前缀的
类型或函数。

## 环境变量

| 变量 | 使用方 | 含义 |
|---|---|---|
| `SYS1_LAYA_TEST_ROOT` | `sys1-laya`、`sys1` | Laya 仓库根目录或单个 checkpoint |
| `SYS1_LAYA_GOLDEN_DIR` | `sys1-laya` | 官方 golden 答案向量 |
| `SYS1_LAYA_ROUTING_GOLDEN` | `sys1-laya` | 官方 routing probe JSON |
| `SYS1_LAYA_ONNX_DIR` | `sys1-laya` | 用于 candle/ONNX parity 的 ONNX 导出目录 |
| `SYS1_NLI_TEST_DIR` | `sys1-nli`、`sys1` | ModernBERT NLI checkpoint |
| `SYS1_NLI_DEBERTA_TEST_DIR` | `sys1-nli` | DeBERTa-v3 NLI checkpoint |
| `SYS1_GLICLASS_TEST_DIR` | `sys1-gliclass`、`sys1` | GLiClass edge checkpoint |
| `SYS1_QWEN3_TEST_DIR` | `sys1-qwen` | dense Qwen3 checkpoint |
| `SYS1_QWEN35_TEST_DIR` | `sys1-qwen` | Qwen3.5 checkpoint |
| `SYS1_QWEN3GUARD_TEST_DIR` | `sys1-qwen3guard`、`sys1` | Qwen3Guard-Gen checkpoint |
| `SYS1_VON_TEST_DIR` | `sys1-von`、`sys1` | Von checkpoint |
| `SYS1_DECIMA_TEST_DIR` | `sys1-decima`、`sys1` | Decima base 仓库 |
| `SYS1_DECIMA_SMALL_TEST_DIR` | `sys1-decima` | Decima small（BERT 编码器） |
| `SYS1_DECIMA_GUARD_SMALL_TEST_DIR` | `sys1-decima` | Decima v2 guard-small |
| `SYS1_KEV_TEST_DIR` | `sys1-kev`、`sys1` | Kev 0.8B adapter 目录 |
| `SYS1_KEV_BASE_DIR` | `sys1-kev` | 0.8B 测试使用的 Qwen3.5 base |
| `SYS1_KEV4B_TEST_DIR` | `sys1-kev` | 含 `base/` 的 Kev 4B 目录 |
| `SYS1_KEV9B_TEST_DIR` | `sys1-kev` | 含 `base/` 的 Kev 9B 目录 |
| `SYS1_DECISION_TEST_DIR` | `sys1-decision`、`sys1` | Decision 1.0 checkpoint |
| `SYS1_CLEF_TEST_DIR` | `sys1-clef` | Clef flash checkpoint |
| `SYS1_CLEF_DTYPE` | `sys1-clef` | 9B 使用 `bf16`（默认 `f32`） |
| `SYS1_LONG_DEVICE` / `SYS1_LONG_DTYPE` | `sys1-laya` 示例 | 长上下文 benchmark 的设备/dtype |
| `SYS1_LONG_REPEATS` | `sys1-laya` 示例 | 长上下文 benchmark 重复次数 |
| `SYS1_BATCH_DEVICE` / `SYS1_BATCH_DTYPE` | `sys1-laya` 示例 | batch benchmark 的设备/dtype |
| `SYS1_REQUIRE_ORT` | `sys1-ort`、`sys1-laya` | ONNX Runtime 缺失时失败而非跳过 |
| `ORT_DYLIB_PATH` | `sys1-ort`、`sys1-laya` | ONNX Runtime 动态库 |
| `SYS1_LIVE_HUB` | `sys1-hub` | 运行真实 Hugging Face/ModelScope manifest 测试 |
| `HF_ENDPOINT` | `sys1-hub` | live 测试使用的 Hugging Face 镜像 |
| `HF_TOKEN` / `MODELSCOPE_API_TOKEN` / `SYS1_HUB_TOKEN` | `sys1-hub` | 私有/在线测试凭据 |

## CI

`.github/workflows/ci.yml` 在每次推送到 `main`（以及 PR）时运行：

| Job | 内容 |
|---|---|
| `Test (ubuntu-latest)` | `cargo fmt --all -- --check`、`cargo clippy --workspace --all-targets -- -D warnings`、`cargo test --workspace` |
| `Test (windows-latest)` | 同上 |
| `Test (macos-latest)` | 同上 |
| `Python tools` | `uv sync --all-extras`、对 `python/tools` 执行 Ruff check/format、`uv run sys1-generate-scoring-stub`，然后 `git diff --exit-code` |

托管 job 不下载模型权重，因此忽略的真实 checkpoint parity 测试不在 CI 中。

`.github/workflows/gpu.yml` 手动触发或按路径触发，运行在带 `linux`、`gpu`、
`cuda` 标签的自托管 runner 上：

- `cargo test --release -p sys1-laya --features cuda --test long_context -- --ignored`；
- 以 `cuda` feature 构建 `sys1-cli`，启动 Laya 服务并用 `examples/loadgen`
  做冒烟压测。

需要仓库变量 `SYS1_LAYA_TEST_ROOT` 指向缓存的 multilingual checkpoint。
TensorRT 执行提供器还需要安装 TensorRT 的 runner；见根 README 的 roadmap。

## Python 工具与 benchmark

```bash
uv sync --all-extras
uv run --package sys1-tools --group dev ruff check python/tools
uv run --package sys1-tools --group dev ruff format --check python/tools
uv run sys1-generate-scoring-stub
git diff --exit-code
```

`sys1-bench` 是性能工具，不是正确性测试。它会启动 release CLI、等待
`/health`，并记录延迟/吞吐；详见 [benchmarks.zh-CN.md](benchmarks.md)。ONNX
校验：

```bash
uv run --extra onnx sys1-export-onnx /path/to/laya /path/to/laya-onnx
uv run --extra check sys1-check-onnx /path/to/laya-onnx /path/to/golden.json
```

## 新增 parity 测试

1. 放在 family crate 的 `tests/` 下。
2. 标记 `#[ignore = "requires SYS1_<FAMILY>_TEST_DIR ..."]`，从环境变量读取
   checkpoint 路径。
3. 用 `--ignored --nocapture` 运行；模型占用数 GB 时加 `--test-threads=1`。
4. golden 值必须来自独立实现（上游 Python 或作者代码），不能用 Rust 实现
   自身生成。
5. 更新本文表格与 family README。
6. 如果不依赖 checkpoint 且运行很快，就不要加 ignore，让托管 CI 覆盖。

## 排查

- **ONNX 测试被跳过**：设置 `ORT_DYLIB_PATH`，并设 `SYS1_REQUIRE_ORT=1`
  把缺失运行时变成失败。
- **Windows CUDA 构建失败**：通过
  `$env:NVCC_CCBIN="C:/Program Files/Microsoft Visual Studio/.../cl.exe"`
  把 MSVC 编译器传给 NVCC。
- **live hub 测试超时**：设置 `HF_ENDPOINT=https://hf-mirror.com`，或保持关闭；
  本地回环 hub 测试始终运行。
- **大模型 OOM**：Kev 9B 与 Clef 9B 在 BF16 下需要约 19 GB 内存，12 GB
  CUDA 卡放不下；在 CPU 上使用 `--release --test-threads=1` 运行。
- **PowerShell 环境变量语法**：使用 `$env:NAME="value"`，只对当前 shell 生效。
