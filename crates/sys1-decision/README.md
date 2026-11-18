# sys1-decision

[简体中文](README.zh-CN.md)

Decision 1.0 (`decision-endpoint-v1`) implemented with candle. A fully
fine-tuned Qwen3.5 text backbone scores each option endpoint against a global
query through `decision_head.safetensors`.

## Use as a library

```toml
[dependencies]
sys1-decision = { git = "https://github.com/hushink/rust-sys1" }
sys1-core = { git = "https://github.com/hushink/rust-sys1" }
sys1-protocol = { git = "https://github.com/hushink/rust-sys1" }
# Or from a checkout:
# sys1-decision = { path = "../rust-sys1/crates/sys1-decision" }
```

```rust
use std::path::Path;
use sys1_core::{DecisionModel as _, Questions};
use sys1_decision::{DecisionModel as DecisionOne, DecisionOptions};
use sys1_protocol::State;

let model = DecisionOne::load_dir(
    Path::new("/models/Decision-1.0-Eos-0.8B"),
    DecisionOptions::default(),
)?;
let output = model.predict(
    &State::text("I was charged twice and want a refund."),
    &Questions::new()
        .choice("team", [("billing", "Charges and refunds"), ("tech", "Bugs")])
        .noul("refund", "The customer asks for a refund."),
)?;
let team = output.answers.choice("team")?;
```

## Layout

Each question is one causal row:

```text
Context:\n<payload(state)>\n\nTask type: <type>\nQuestion:\n<payload(instructions)>\nOptions:
  \n<option>\n{"key":…,"description":…}\n</option>   (one segment per option)
  \n\nSelect the single option best supported by the context and instructions.\nDecision:
```

`payload` uses a string verbatim and otherwise canonical JSON (sorted keys, no
spaces). Each segment is tokenized separately. The head reads the last token of
each option segment and the row's last token.

- Choice criteria are objects with 2–255 options.
- Score levels are 2–10.
- Noul options are `[false, true]`; missing sides use the author's defaults.
- Rows over 16384 tokens are rejected, never truncated.

Probabilities are `softmax(logits / T)` with the checkpoint temperature from
`config.json`.

## Files

```text
config.json
decision_config.json
decision_head.safetensors
tokenizer.json
backbone/config.json
backbone/*.safetensors
```

## Test

```bash
$env:SYS1_DECISION_TEST_DIR="C:/models/Decision-1.0-Eos-0.8B"
cargo test --offline -p sys1-decision --test decision -- --ignored --nocapture
```
