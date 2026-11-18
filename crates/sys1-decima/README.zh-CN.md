# sys1-decima

[English](README.md)

基于 candle 的 Decima（`decima-late-interaction-v1`/`-v2`）。ModernBERT 编码器
（`decima-base` 为 mmBERT）或 BERT 编码器（`decima-small` 为
`multilingual-e5-small`）分别编码 state 和每个选项；late-interaction scorer
让每个选项读取 state token 后得到一个原始分数。选项之间不互相注意。编码器架构
由 `encoder/config.json` 的 `model_type = "bert"` 自动选择；BERT 默认使用
`[CLS]`/`[SEP]` 0/2，配置可覆盖。

## 作为库使用

```toml
[dependencies]
sys1-decima = { git = "https://github.com/hushink/rust-sys1" }
sys1-core = { git = "https://github.com/hushink/rust-sys1" }
sys1-protocol = { git = "https://github.com/hushink/rust-sys1" }
# 或从本地 checkout 引入：
# sys1-decima = { path = "../rust-sys1/crates/sys1-decima" }
```

```rust
use std::path::Path;
use sys1_core::{DecisionModel, Questions};
use sys1_decima::{DecimaModel, DecimaOptions};
use sys1_protocol::State;

let model = DecimaModel::load_dir(Path::new("/models/decima-base"), DecimaOptions::default())?;
let output = model.predict(
    &State::text("I was charged twice and want a refund."),
    &Questions::new()
        .choice("team", [("billing", "Charges and refunds"), ("tech", "Bugs")])
        .score("urgency", ["low", "medium", "high"]),
)?;
let team = output.answers.choice("team")?;
let urgency = output.answers.score("urgency")?;
```

## 行

```text
state 行   [cls] tok(state_prefix + normalize(text + "\n" + state))[..max_state_tokens] [sep]
option 行  [cls] tok(option_prefix + normalize(text + " " + option))[..max_choice_tokens] [sep]
```

- `text` 是问题 instructions；缺失/null 时用 question id。带 true/false 描述的
  noul 会追加 `\nTrue if: …\nFalse if: …`。
- choice 选项为 `label: description`（描述为空时仅 label）；score 为等级文本；
  noul 为 `yes`/`no`。
- `normalize` 为 NFC 后接 Python `str.strip()`。
- state 行超过预算时报 `STATE_TRUNCATED`；option 行按上游行为静默截断。
- `choice_fit`/`option_fit` checkpoint 在完整文本放不下时，把问题适配进选项窗口；
  `DecimaOptions::choice_fit`（CLI：`[families.decima] choice_fit`）可覆盖
  `decima.json` 中的值。

choice/noul 概率为 `softmax(scores / T)`；score 使用 Decima 的 cumulative-link
ordinal head。token 长度相同的选项行会合并为一批编码与打分；不同长度保持分行，
避免 padding 改变编码器的局部注意力。

## 文件

接受仓库根目录或其 `pytorch/` 目录：

```text
decima.json
encoder/config.json
encoder/model.safetensors
encoder/tokenizer.json
head.safetensors
```

## 测试

```bash
$env:SYS1_DECIMA_TEST_DIR="C:/models/decima-base"
cargo test --offline -p sys1-decima --test decima -- --ignored --nocapture
```

测试用独立的 PyTorch 实现（按文档构建图）对比 choice/noul/score 答案。
