# sys1-clef

[简体中文](README.zh-CN.md)

Clef (`clef-joint-v1`) implemented with candle. A Qwen3.5 backbone reads the
state and schema, and Clef's joint schema head routes evidence to every question
and scores every option together in one forward pass.

## Use as a library

```toml
[dependencies]
sys1-clef = { git = "https://github.com/hushink/rust-sys1" }
sys1-core = { git = "https://github.com/hushink/rust-sys1" }
sys1-protocol = { git = "https://github.com/hushink/rust-sys1" }
# Or from a checkout:
# sys1-clef = { path = "../rust-sys1/crates/sys1-clef" }
```

```rust
use std::path::Path;
use sys1_clef::{ClefModel, ClefOptions};
use sys1_core::{DecisionModel, Questions};
use sys1_protocol::State;

let model = ClefModel::load_dir(Path::new("/models/clef-flash"), ClefOptions::default())?;
let output = model.predict(
    &State::text("I was charged twice and want a refund."),
    &Questions::new()
        .choice("team", [("billing", "Charges and refunds"), ("tech", "Bugs")])
        .noul("refund", "The customer asks for a refund."),
)?;
let team = output.answers.choice("team")?;
```

## Sequence

```text
ids = tok(prefix) ++ tok(render(state))[..budget] ++ schema ++ tok(suffix)

prefix = "<|im_start|>system\n…<|im_end|>\n<|im_start|>user\nSTATE:\n"
schema = "\n\nSCHEMA FIELDS:\n"
         per question: "\nFIELD i\nID: …\nTYPE: …\nINSTRUCTION: " + instructions
         + "\nALLOWED OPTIONS:\n"
         + per option: "OPTION j: " + canonical({"option_id": …, "description": …}) + "\n"
         + "END FIELD\n"
suffix = "\n<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\nJOINT SCHEMA DECISIONS:"
```

- Strings are used verbatim; other values use canonical JSON (sorted keys, no
  spaces).
- Noul options are `[true, false]`; choice labels are sorted by code point;
  score levels keep their order.
- The default text limit is 4096 tokens. The state is cut to fit; a schema that
  cannot fit is rejected.
- The head reads span means, the untied LM head for lexical vectors, two
  evidence-routing layers and four decoder layers, then scores each option
  against its field.

Probabilities are a plain softmax of the head's logits.

## Files

```text
config.json
tokenizer.json
joint_head_config.json
joint_head.safetensors
model-*.safetensors          # Qwen3.5 shards (vision tensors are ignored)
```

The 9B `clef-flash` needs roughly 18 GB in BF16. Use `dtype = "bf16"` and a GPU
for practical latency and memory; on a 12 GB card the backbone does not fit, so
the tests below use BF16 on the CPU (about 30 GB of RAM is required).

## Test

```bash
$env:SYS1_CLEF_TEST_DIR="C:/models/clef-flash"
cargo test --release --offline -p sys1-clef --test clef -- --ignored --nocapture
```

Set `SYS1_CLEF_DTYPE=bf16` for the 9B checkpoint (the default test dtype is
F32). `matches_the_recorded_triage_decisions` compares the choice/noul/score
decisions with Ollaya's recorded `clef-flash` integration corpus.
