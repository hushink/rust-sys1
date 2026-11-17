# sys1-nli

[简体中文](README.zh-CN.md)

Zero-shot NLI decision model (`nli-pairs-v1`) on candle ModernBERT.

Each option becomes a hypothesis; one `[CLS] state [SEP] hypothesis [SEP]` row
is encoded per option, and the entailment logit becomes that option's score.
Choice/score answers use a softmax over option logits; `noul` with both
descriptions is a two-hypothesis softmax, and a `noul` without descriptions is
one row read as `[not_entailment, entailment]` (sigmoid).

## Checkpoints

- `MoritzLaurer/ModernBERT-base-zeroshot-v2.0` (150M, ~299 MB safetensors).
- `MoritzLaurer/ModernBERT-large-zeroshot-v2.0` (395M, ~792 MB).

Both are English, Apache-2.0, and trained for 512-token premise/hypothesis
pairs. The directory needs `config.json`, `model.safetensors`, and
`tokenizer.json`.

## Usage

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

CLI / server:

```powershell
sys1 run local:/path/to/nli-modernbert-base --input request.json
sys1 serve --model nli=local:/path/to/nli-modernbert-base
sys1 serve --model nli=hf:MoritzLaurer/ModernBERT-base-zeroshot-v2.0
```

The family is detected from `config.json` (`model_type = "modernbert"` plus an
`entailment` label), so the same `local:`/`hf:` paths work with pull/show/rm.

## Options

- `device` / `dtype`: candle device and weight dtype (F32 default).
- `max_len`: pair budget, default 512; validated against the encoder.
- `max_batch`: rows per forward pass, default 32.

## Parity

`tests/nli.rs` is an opt-in parity test against the upstream
`transformers` result:

```powershell
$env:SYS1_NLI_TEST_DIR="D:/models/nli-modernbert-base"
cargo test -p sys1-nli --test nli -- --ignored --nocapture
```

For the shared sentence in the test, choice probabilities, score
probabilities, and `noul` match the upstream `AutoModelForSequenceClassification`
forward within 1e-4.

## Limits

- English checkpoints; other languages need the multilingual NLI variants.
- The 512-token pair budget is the training limit; larger `max_len` values are
  accepted only when the architecture allows them, without a quality claim.
- Box questions are unsupported.
