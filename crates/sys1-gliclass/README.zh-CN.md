# sys1-gliclass

[English](README.md)

基于 candle ModernBERT 的 GLiClass uni-encoder 决策模型
（`gliclass-uni-v1`）。

label 与文本共享一条序列：每个 label 前缀 `<<LABEL>>`，prompt 与 state 跟在
`<<SEP>>` 之后；label span 做 mean pooling，MLP 将每个 label 与首 token
（`[CLS]`）表示打分。choice/score 对 label logits 做 softmax；带双描述的
`noul` 用两个 label 行，没有描述的 `noul` 用单 label sigmoid。

## Checkpoint

`knowledgator/gliclass-instruct-edge-v1.0`（Apache-2.0）：32.7M 参数的
Ettin/ModernBERT 编码器，`model.safetensors` 约 131 MB，是 GLiClass 家族中
小且快的成员，CPU 即可运行。

目录需要 `config.json`、`model.safetensors` 与 `tokenizer.json`。

## 作为库使用

```toml
[dependencies]
sys1-gliclass = { git = "https://github.com/hushink/rust-sys1" }
sys1-core = { git = "https://github.com/hushink/rust-sys1" }
sys1-protocol = { git = "https://github.com/hushink/rust-sys1" }
# 或从本地 checkout 引入：
# sys1-gliclass = { path = "../rust-sys1/crates/sys1-gliclass" }
```

## 用法

```rust
use sys1_core::{DecisionModel, Questions};
use sys1_gliclass::{GliclassModel, GliclassOptions};
use sys1_protocol::State;

let model = GliclassModel::load_dir(
    Path::new("/path/to/gliclass-edge"),
    GliclassOptions::default(),
)?;
let output = model.predict(
    &State::text("I was charged twice and need a refund."),
    &Questions::new().score("urgency", ["Can wait", "This week", "Today"]),
)?;
```

CLI / 服务：

```powershell
sys1 run local:/path/to/gliclass-edge --input request.json
sys1 serve --model gliclass=local:/path/to/gliclass-edge
sys1 serve --model gliclass=hf:knowledgator/gliclass-instruct-edge-v1.0
```

## 布局

- 行文本：`<<LABEL>>label...<<SEP>>prompt + state`。
- choice label：`label` 或 `label: description`。
- score label：`level i: description`。
- 带双描述的 `noul`：`[false description, true description]`，prompt 为
  instructions；否则单 label = instructions，prompt 为空。
- 用户文本中的结构 marker 替换为空格。
- 使用 checkpoint tokenizer，右侧截断到 1024 token；label 被截断的问题会拒绝。

## 选项

- `device` / `dtype`：candle 设备与权重 dtype（默认 F32）。
- `max_len`：label + prompt + state 的总预算，默认 1024。
- `max_batch`：每次前向的行数，默认 32。

## 数值对齐

```powershell
$env:SYS1_GLICLASS_TEST_DIR="D:/models/gliclass-edge"
cargo test -p sys1-gliclass --test gliclass -- --ignored --nocapture
```

同一句子上与上游 `gliclass==0.1.20` 模型对比，choice、score 与 noul 概率均在
1e-4 内一致。

## 限制

- 只实现 ModernBERT（edge）uni-encoder 变体；DeBERTa-v3-large 与 v4 calibrator
  未实现。
- 面向英文的 checkpoint；单行预算 1024 token。
- 不支持 box 问题。
