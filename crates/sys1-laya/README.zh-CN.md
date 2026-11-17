# sys1-laya

[English](README.md)

candle 版 clean-room Laya 决策模型实现。

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

`TypeSafe` confidence 为 `(K * p_max - 1) / (K - 1)`；`LayaNative` 为
`1 - H(p) / ln(K)`。`noul` 按官方形状不携带 confidence。

## Router

`LayaRouter::new(root, options)` 需要 Laya 仓库根目录（含 `multilingual/` 与
`typed-decisions/` 子目录）。选择顺序：

1. 问题 ID 集合命中四种 typed-decisions workflow 之一 → `typed-decisions`；
2. 以 ASCII 为主且包含英文功能词 → `english`；
3. 其余 → `multilingual`。

v1 的 Router 是启发式；完整官方 routing golden 对齐待补。
`LayaRouter::predict_batch` 会按选中的 checkpoint 分组，因此 `laya-latest`
也能受益于服务端 batching，而不会把不同 checkpoint 混进同一批。

## Factory

`LayaFactory` 为 `ModelRegistry` 维护「模型名 → checkpoint 目录或 router 根目录」
的映射。`register` / `register_router` 接收 `&self`，因此服务运行期间也能添加模型；
`unregister` 删除映射。重复注册同名项时保留第一次的注册，替换前需先 `unregister`。

## Checkpoint 辅助

checkpoint 布局知识集中在 sys1-laya，组合层无需再定义 Laya 前缀 API：

- `is_checkpoint_directory` / `has_checkpoint_variants`
- `variant_directories` — 仓库根目录下的三个目录名与路径
- `local_checkpoint_manifest` — 为没有 `model.json` 的裸 checkpoint 目录生成
  manifest 与文件映射
- `register_checkpoint_bundle` — 注册单个 checkpoint 或 router bundle，返回应加入
  注册表的 manifest

## 数值对齐

Golden 测试对比官方 Laya vectors 的 `input_ids`、`markers`、logits 与答案。
三个 checkpoint 的 `case_dict_state` 均在 1e-3 内一致。

```bash
$env:SYS1_LAYA_TEST_ROOT="C:/.../snapshots/<sha>"
$env:SYS1_LAYA_GOLDEN_DIR="D:/path/to/laya/laya-rust/tests/golden"
cargo test --offline -p sys1-laya -- --ignored --nocapture
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

`cuda`、`cudnn`、`metal`、`accelerate` 转发到 candle。

FlashAttention 未启用：932 token 原型快约 2.6x，但决策输出发生变化；上游
kernel 在当前 CUDA 13.4/Windows 工具链上也构建失败。eager attention 仍是
参考实现。
