# sys1-nli

[English](README.md)

基于 candle ModernBERT 的 zero-shot NLI 决策模型（`nli-pairs-v1`）。

每个选项转成一条 hypothesis，编码一行
`[CLS] state [SEP] hypothesis [SEP]`，其 entailment logit 作为该选项分数。
choice/score 对选项 logits 做 softmax；带双描述的 `noul` 是两条 hypothesis 的
softmax，没有描述的 `noul` 用单行 `[not_entailment, entailment]`（sigmoid）。

## Checkpoint

- `MoritzLaurer/ModernBERT-base-zeroshot-v2.0`（150M，约 299 MB safetensors）。
- `MoritzLaurer/ModernBERT-large-zeroshot-v2.0`（395M，约 792 MB）。

两者均为英文、Apache-2.0，训练时使用 512 token 的 premise/hypothesis 对。
目录需要 `config.json`、`model.safetensors` 与 `tokenizer.json`。

## 用法

```rust
use sys1_core::{DecisionModel, Questions};
use sys1_nli::{NliModel, NliOptions};
use sys1_protocol::State;

let model = NliModel::load_dir(Path::new("/path/to/nli-modernbert-base"), NliOptions::default())?;
let output = model.predict(
    &State::text("I was charged twice and need a refund."),
    &Questions::new().choice("tone", [("angry", "Upset"), ("calm", "Neutral")]),
)?;
```

CLI / 服务：

```powershell
sys1 run local:/path/to/nli-modernbert-base --input request.json
sys1 serve --model nli=local:/path/to/nli-modernbert-base
sys1 serve --model nli=hf:MoritzLaurer/ModernBERT-base-zeroshot-v2.0
```

模型族通过 `config.json` 自动识别（`model_type = "modernbert"` 且含
`entailment` 标签），因此 `local:` / `hf:` 路径都能用于 pull/show/rm。

## 选项

- `device` / `dtype`：candle 设备与权重 dtype（默认 F32）。
- `max_len`：pair 预算，默认 512，并受编码器上限校验。
- `max_batch`：每次前向的行数，默认 32。

## 数值对齐

`tests/nli.rs` 是对上游 `transformers` 结果的 opt-in 对齐测试：

```powershell
$env:SYS1_NLI_TEST_DIR="D:/models/nli-modernbert-base"
cargo test -p sys1-nli --test nli -- --ignored --nocapture
```

测试句子的 choice 概率、score 概率与 `noul` 均在 1e-4 内与上游
`AutoModelForSequenceClassification` 前向一致。

## 限制

- 英文 checkpoint；其他语言需要多语言 NLI 变体。
- 512 token 是训练上限；更大的 `max_len` 仅在架构允许时接受，不做质量承诺。
- 不支持 box 问题。
