# sys1-von

[简体中文](README.zh-CN.md)

Von 1.1 (`von-option-marker-v1`) as a candle decision model. Von is a
ModernBERT-large encoder with an option-marker scorer: every option gets a
`[MASK]` token in one sequence and is scored from that token's hidden state.

## Use as a library

```toml
[dependencies]
sys1-von = { git = "https://github.com/hushink/rust-sys1" }
sys1-core = { git = "https://github.com/hushink/rust-sys1" }
sys1-protocol = { git = "https://github.com/hushink/rust-sys1" }
# Or from a checkout:
# sys1-von = { path = "../rust-sys1/crates/sys1-von" }
```

```rust
use std::path::Path;
use sys1_core::{DecisionModel, Questions};
use sys1_protocol::State;
use sys1_von::{VonModel, VonOptions};

let model = VonModel::load_dir(Path::new("/models/von"), VonOptions::default())?;
let output = model.predict(
    &State::text("I was charged twice and want a refund."),
    &Questions::new()
        .choice("team", [("billing", "Charges and refunds"), ("tech", "Bugs")])
        .noul("refund", "The customer asks for a refund."),
)?;
let team = output.answers.choice("team")?;
let refund = output.answers.noul("refund")?;
```

## Layout

```text
[CLS] <instructions> <state> [SEP] [MASK] <description 0> [MASK] <description 1> ... [SEP]
```

- The state is rendered with Python `str()`: a string verbatim, an object as
  `key: value` lines, an array as Python `repr`.
- A choice option is its description, or the label when the description is
  empty. Score levels use Von's `what Examples:` rendering.
- A noul question reads options in `[true, false]` order and returns
  `[false, true]` logits. A zero-shot noul adds an empty-state row and subtracts
  `0.7 · (n_true − n_false)` from the true side.
- Rows longer than 8192 tokens cut the state at a character boundary; the
  instructions and options alone must fit.

Calibration uses Von's input-conditioned temperature map
(`von-entropy-length-v1`), clamped to `[0.3, 12]`.

## Files

A checkpoint directory contains `config.json`, `tokenizer.json` and
`option_marker.pt` (the upstream PyTorch zip is read in place through candle's
restricted pickle reader). The ModernBERT config's inherited 2k rotary limit is
grown to the requested `max_len`.

## Test

```bash
$env:SYS1_VON_TEST_DIR="C:/models/von"
cargo test --offline -p sys1-von --test von -- --ignored --nocapture
```

The test compares choice, noul and score probabilities with `von-sdk==1.1.1`.
