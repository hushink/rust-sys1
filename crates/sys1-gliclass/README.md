# sys1-gliclass

[简体中文](README.zh-CN.md)

GLiClass uni-encoder decision model (`gliclass-uni-v1`) on candle ModernBERT.

Labels and the text share one sequence: every label is prefixed with
`<<LABEL>>`, the prompt and state follow a `<<SEP>>`, label spans are
mean-pooled, and an MLP scores each label against the first-token (`[CLS]`)
representation. Choice/score use a softmax over labels; a `noul` with both
descriptions uses two label rows, while a `noul` without descriptions is a
one-label sigmoid.

## Checkpoint

`knowledgator/gliclass-instruct-edge-v1.0` (Apache-2.0): a 32.7M-parameter
Ettin/ModernBERT encoder, ~131 MB `model.safetensors`. It is the small/fast
member of the GLiClass family and runs comfortably on CPU.

The directory needs `config.json`, `model.safetensors`, and `tokenizer.json`.

## Use as a library

```toml
[dependencies]
sys1-gliclass = { git = "https://github.com/hushink/rust-sys1" }
sys1-core = { git = "https://github.com/hushink/rust-sys1" }
sys1-protocol = { git = "https://github.com/hushink/rust-sys1" }
# Or from a checkout:
# sys1-gliclass = { path = "../rust-sys1/crates/sys1-gliclass" }
```

## Usage

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

CLI / server:

```powershell
sys1 run local:/path/to/gliclass-edge --input request.json
sys1 serve --model gliclass=local:/path/to/gliclass-edge
sys1 serve --model gliclass=hf:knowledgator/gliclass-instruct-edge-v1.0
```

## Layout

- Row text: `<<LABEL>>label...<<SEP>>prompt + state`.
- Choice labels: `label` or `label: description`.
- Score labels: `level i: description`.
- Noul with both descriptions: `[false description, true description]` and the
  instructions as prompt; otherwise one label = instructions, empty prompt.
- Structural markers in user text are replaced with spaces.
- Tokenization uses the checkpoint tokenizer with right truncation to 1024
  tokens; a question whose labels are cut off is rejected.

## Options

- `device` / `dtype`: candle device and weight dtype (F32 default).
- `max_len`: total label+prompt+state budget, default 1024.
- `max_batch`: rows per forward pass, default 32.

## Parity

```powershell
$env:SYS1_GLICLASS_TEST_DIR="D:/models/gliclass-edge"
cargo test -p sys1-gliclass --test gliclass -- --ignored --nocapture
```

Against the upstream `gliclass==0.1.20` model on the same sentence, choice,
score, and noul probabilities match within 1e-4.

## Limits

- Only the ModernBERT (edge) uni-encoder variant is implemented; the
  DeBERTa-v3-large variant and the v4 calibrator are not.
- English-oriented checkpoint; 1024-token row budget.
- Box questions are unsupported.
