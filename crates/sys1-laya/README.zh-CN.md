# sys1-laya

[English](README.md)

candle 版 clean-room Laya 决策模型实现，并提供可选 ONNX Runtime 后端。

## Checkpoints

- `english` — ModernBERT-large，421M，512 tokens
- `multilingual` — mmBERT-base，322M，1024 tokens
- `typed-decisions` — ModernBERT-large，421M，1024 tokens

一个 checkpoint 目录包含：

```text
model.safetensors
encoder/config.json
rl_agent_config.json
tokenizer/tokenizer.json
```

## ONNX Runtime 后端

启用 `onnx` feature 即可运行导出为 ONNX 的 Laya checkpoint：

```bash
cargo build -p sys1-laya --features onnx
```

ONNX 模型目录包含：

```text
model.onnx                 # 或 onnx.json 指定的文件
rl_agent_config.json
tokenizer/tokenizer.json   # 或 tokenizer.json
onnx.json                  # 可选
```

`LayaOnnxModel::load_dir` 与 `LayaModel` 共用 tokenization、序列布局、校准和
答案格式化，仅前向图不同。`LayaFactory` 会加载 `backend = "onnx"`、
`format = "onnx"` 的 manifest；`is_onnx_directory` / `is_model_directory`
用于识别这类目录。

### 计算图契约

| 输入 | 类型 | 形状 |
| --- | --- | --- |
| `input_ids` | int64 | `[batch, seq]` |
| `attention_mask` | int64 | `[batch, seq]` |
| `type_ids` | int64 | `[batch]` |
| `markers` | int64 | `[n]` |

`markers` 是 `input_ids` 的扁平索引 `row * seq + position`。计算图需依次完成
encoder、type embedding、decision head、marker gather 与 scorer，并输出形状为
`[n]` 的 float32 `logits`，每个 marker 一个 logit。batch 与 seq 轴必须是动态的。

`onnx.json` 可覆盖默认值：

```json
{
  "model": "model.onnx",
  "max_batch": 8,
  "intra_threads": 4,
  "inter_threads": 1,
  "input_ids": "input_ids",
  "attention_mask": "attention_mask",
  "type_ids": "type_ids",
  "markers": "markers",
  "output": "logits"
}
```

`max_batch` 限制每次 ONNX run 的 unique item 数，超出会切分。ONNX Runtime 在创建
session 时动态加载：设置 `ORT_DYLIB_PATH` 指向 `onnxruntime` 共享库，或将库放入
平台加载路径。构建期不会下载运行时二进制。绑定以 ONNX Runtime API 17 为目标，
已在 1.17 上验证。

执行提供器 feature：`ort-cuda`、`ort-tensorrt`、`ort-directml`、`ort-coreml`
（均隐含 `onnx`）。TensorRT 需要本机 TensorRT SDK。官方尚未发布 Laya ONNX
产物；仓库提供纯 NumPy/ONNX 导出器（不依赖 PyTorch）：

```bash
uv run --with onnx --with safetensors --with numpy --python 3.13 \
  ../../scripts/export_laya_onnx.py /path/to/laya-multilingual /path/to/laya-onnx

uv run --with onnxruntime --with numpy --python 3.13 \
  ../../scripts/check_laya_onnx.py /path/to/laya-onnx \
  /path/to/golden/multilingual/case_dict_state.json
```

导出器写出 F32 权重（multilingual 约 1.2 GiB，大模型约 1.6 GiB），复制
tokenizer 与 `rl_agent_config.json`，并生成 `onnx.json`。三个 checkpoint 的
导出 logits 均与官方 golden 对比：最大 `|ONNX - golden|` 为 1.7e-6 到 2.5e-6。

Rust 端到端一致性测试会比较 candle 与 ONNX 的答案：

```powershell
$env:SYS1_LAYA_TEST_ROOT="C:/.../multilingual"
$env:SYS1_LAYA_ONNX_DIR="D:/models/laya-onnx"
$env:SYS1_LAYA_GOLDEN_DIR="D:/path/to/laya/laya-rust/tests/golden"
cargo test -p sys1-laya --features onnx --test onnx_parity -- --ignored --nocapture
```

多 GB 导出请使用较新的 ONNX Runtime。Windows 上 ONNX Runtime 1.17 在加载
1.2 GiB multilingual 计算图时 session 创建不稳定；1.32 可稳定加载运行。
将 `ORT_DYLIB_PATH` 指向较新的库即可。合成计算图测试仍覆盖 batching、padding、
marker 索引与 factory 接线。

## 示例

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

## 选项

- `device`：candle `Device`（默认 CPU）。
- `dtype`：默认 `F32`；支持 `F16` / `BF16`。
- `max_len`：覆盖 checkpoint 的上下文预算。
- `confidence_mode`：`TypeSafe`（默认）或 `LayaNative`。
- `local_window`：CUDA 上 local 层的精确滑动窗口 attention（默认 `true`，
  ≥512 token 时启用）；设为 `false` 使用 eager attention。
- `max_attention_elements`：单次 global-attention batch 的
  `batch * heads * seq_len²` 上限（默认 64M 元素），超出会切分。
- `router` / `lang_guess`：见 [Router](#router)。
- checkpoint 若包含 `act_head.*`，还会预测 `action.act_probability`，通过
  `output.action(id)` 与 `output.actions` 读取。ONNX 导出暂未包含该 head，
  因此该后端的 `actions` 为空。

`TypeSafe` confidence 为 `(K * p_max - 1) / (K - 1)`；`LayaNative` 为
`1 - H(p) / ln(K)`。`noul` 按官方形状不携带 confidence。

## 长上下文扫描

`DecisionModel::predict_long` 只对 state tokenize 一次，切成重叠的 token 窗口，
用共享前向打分，再按问题聚合。`LayaModel`、`LayaOnnxModel` 与 `LayaRouter`
均实现该方法；官方 wire API 保持不变。

```rust
use sys1_core::LongContextOptions;

let output = model.predict_long(
    &State::text(document),
    &questions,
    LongContextOptions {
        window: Some(512),   // 默认 max(64, max_len - head_max_len - 8)
        stride: Some(256),   // 默认有效窗口的一半
        batch_size: None,    // 每次前向的窗口数
    },
)?;
let deciding = output.window("urgency").expect("窗口归因");
assert_eq!(deciding.count, output.scored_windows);
```

- `noul` 取各窗口 P(true) 的最大值。
- `choice` / `score` 取最自信窗口的完整答案，避免中性窗口多数票淹没局部信号。
- 每个答案的窗口记录 `index`、`token_start`、`token_end` 与 `count`；
  `scored_windows` 对应官方 `usage.windows`。
- state 能放进一个窗口时直接走普通 `predict`（`scored_windows = 1`，无逐答案窗口）。
- `usage.input_tokens` 汇总所有窗口（含重叠部分）。返回的置信度属于决定窗口，
  不是整篇文档的校准值。

英文 checkpoint 已在相同 state、问题、window 与 stride 下与官方
`predict_long` 对比：答案、`action.act_probability`、窗口归因与
`usage.input_tokens` 一致。

## Router

`LayaRouter::new(root, options)` 需要 Laya 仓库根目录（含 `english/`、
`multilingual/` 与 `typed-decisions/` 子目录）。实现移植了官方 router 的文字
系统检测、拉丁语停用词画像、缩写/全大写规则、混合片段处理，以及与 Python
`repr` 一致的 reason 字符串。选择优先级：

1. 显式 `model` hint（`RouteHints` / `route_with`）；
2. 显式 `task` hint；
3. 命中 typed-decisions 工作流（仅当 `auto_task_detection` 开启）；
4. 显式 `lang` hint；
5. 单次调用或 router 级 `lang_guess`；
6. 语言分析：英文 → `english`，其余 → `default` 变体（默认 `multilingual`）。

`LayaRouterOptions { default, auto_task_detection }` 对齐官方默认值
（`multilingual`、`false`）；`LayaOptions::lang_guess` 设置 router 级提示。
官方 routing golden 的全部 159 个用例均可精确复现 checkpoint、workflow、
reason 与 detection 字段。

`LayaRouter::predict_batch` 会按选中的 checkpoint 分组，因此 `laya-latest`
也能受益于服务端 batching，而不会把不同 checkpoint 混进同一批。

## Factory

`LayaFactory` 为 `ModelRegistry` 维护「模型名 → checkpoint 目录或 router 根目录」
的映射。`register` / `register_router` 接收 `&self`，因此服务运行期间也能添加模型；
`unregister` 删除映射。重复注册同名项时保留第一次的注册，替换前需先 `unregister`。

## Checkpoint 辅助

checkpoint 布局知识集中在 sys1-laya，组合层无需再定义 Laya 前缀 API：

- `is_checkpoint_directory` — candle safetensors checkpoint 布局
- `is_onnx_directory` / `is_model_directory` — ONNX 导出布局，或任一后端
- `has_checkpoint_variants`
- `variant_directories` — 仓库根目录下的三个目录名与路径
- `local_checkpoint_manifest` — 为没有 `model.json` 的裸 checkpoint 目录生成
  manifest 与文件映射
- `register_checkpoint_bundle` — 注册单个 checkpoint 或 router bundle，返回应加入
  注册表的 manifest

## 数值对齐

Golden 测试对比官方 Laya vectors 的 `input_ids`、`markers`、logits、答案与
`action.act_probability`。三个 checkpoint 的 `case_dict_state` 均在 1e-3 内一致。
routing 测试回放官方 probe 文件，要求 model/workflow/reason/detection 完全一致。

```bash
$env:SYS1_LAYA_TEST_ROOT="C:/.../snapshots/<sha>"
$env:SYS1_LAYA_GOLDEN_DIR="D:/path/to/laya/laya-rust/tests/golden"
$env:SYS1_LAYA_ROUTING_GOLDEN="D:/path/to/laya/laya-rust/tests/golden/routing/route_probe.json"
cargo test --offline -p sys1-laya -- --ignored --nocapture
cargo test --offline -p sys1-laya --test routing_golden -- --nocapture
```

## 性能

在 i5-12400F + RTX 3060 上，`laya-multilingual` F32 且并发请求使用不同 state 时，
单 noul 请求 CPU ~85 ms、CUDA ~17 ms；三问题请求 ~200 ms vs ~21 ms。CPU 延迟
大致随 choice/score 选项数线性增长；CUDA 在 10 个选项以内基本保持 15–18 ms。

CPU 上的 Laya 会退出服务端跨请求 batching，因为独立 CPU 前向已经使用线程池；
CPU 吞吐在 ~24–26 req/s 饱和。PowerShell 压测下 GPU batching 把 8 并发提升到
~164 req/s、32 并发 ~209 req/s；`--dtype f16` 在 32 并发达到 ~248 req/s。
Rust `loadgen` 示例在 F16 + `--batch-max 32` 下饱和约 ~345 req/s，抽查概率
差异约 0.0005。完整扫描与复现命令见根 README。

长 state 下，CUDA local 层在 ≥512 token 时使用精确滑动窗口 attention：932 token
单 noul 请求在 F16 从 ~194 ms 降到 ~141 ms（1.38x），F32 从 ~185 ms 降到
~142 ms（1.30x）；F32 答案与 eager 完全一致，F16 最大差异约 `6e-4`。4 个并发
932-token F16 请求从 ~724 ms 降到 ~433 ms。`long_context_bench` 可复现模型级
数据；`LayaOptions::local_window = false`（或 CLI `--no-local-window`）可退回
eager。

## 限制

- 未实现 long context 扫描、act head、abstention、parallel option layout。
- Laya 不支持 box questions，返回 `ModelError::Unsupported`。
- 不内置模型权重；再分发前需确认权重许可证。

## Features

`onnx` 启用 ONNX Runtime 与 `LayaOnnxModel`；`cuda`、`cudnn`、`metal`、
`accelerate` 转发到 candle；`ort-cuda`、`ort-tensorrt`、`ort-directml`、
`ort-coreml` 启用 ONNX Runtime 执行提供器。

FlashAttention 未启用：932 token 原型快约 2.6x，但决策输出发生变化；上游
kernel 在当前 CUDA 13.4/Windows 工具链上也构建失败。eager attention 仍是
参考实现。
