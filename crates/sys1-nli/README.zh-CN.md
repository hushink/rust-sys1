# sys1-nli

[English](README.md)

基于 candle ModernBERT 或 DeBERTa-v3 的 zero-shot NLI 决策模型
（`nli-pairs-v1`）。

每个选项转成一条 hypothesis，编码一行
`[CLS] state [SEP] hypothesis [SEP]`，其 entailment logit 作为该选项分数。
choice/score 对选项 logits 做 softmax；带双描述的 `noul` 是两条 hypothesis 的
softmax，没有描述的 `noul` 用单行 `[not_entailment, entailment]`（sigmoid）。

DeBERTa-v3 使用 disentangled attention：content-to-content、
content-to-position 与 position-to-content 三部分分数，共享一张按 LayerNorm
归一化的相对位置表（256 个 bucket）。其 zeroshot-v2.0 checkpoint 使用
`The answer to "{instructions}" is: …` 模板；ModernBERT 保持
`{description}` / `This text is about {label}.`。

## Checkpoint

- `MoritzLaurer/ModernBERT-base-zeroshot-v2.0`（150M，约 299 MB safetensors）。
- `MoritzLaurer/ModernBERT-large-zeroshot-v2.0`（395M，约 792 MB）。
- `MoritzLaurer/deberta-v3-large-zeroshot-v2.0`（435M，约 870 MB F16；
  上游系列中准确率最高）。

均为英文、宽松许可证，训练时使用 512 token 的 premise/hypothesis 对。
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
sys1 serve --model nli=hf:MoritzLaurer/deberta-v3-large-zeroshot-v2.0
```

模型族通过 `config.json` 自动识别（`model_type` 为 `modernbert` 或
`deberta-v2`，且含 `entailment` 标签），因此 `local:` / `hf:` 路径都能用于
pull/show/rm。

## 选项

- `device` / `dtype`：candle 设备与权重 dtype（默认 F32）。
- `max_len`：pair 预算，默认 512，并受编码器上限校验。
- `max_batch`：每次前向的行数，默认 32。

## 数值对齐

`tests/nli.rs` 是对上游 `transformers` 结果的 opt-in 对齐测试：

```powershell
$env:SYS1_NLI_TEST_DIR="D:/models/nli-modernbert-base"
cargo test -p sys1-nli --test nli -- --ignored --nocapture

$env:SYS1_NLI_DEBERTA_TEST_DIR="D:/models/deberta-v3-large-zeroshot-v2.0"
cargo test -p sys1-nli --test nli predicts_with_the_deberta_checkpoint \
  -- --ignored --nocapture
```

测试句子的 choice 概率、score 概率与 `noul` 在两种编码器上均于 1e-4 内与
上游 `AutoModelForSequenceClassification` 前向一致。

## 限制

- 英文 checkpoint；其他语言需要多语言 NLI 变体。
- 512 token 是训练上限；更大的 `max_len` 仅在架构允许时接受，不做质量承诺。
- 不支持 box 问题。
