# sys1-clef

[简体中文](README.zh-CN.md)

Clef (`clef-joint-v1`) implemented with candle. A Qwen3.5 backbone reads the
state and schema, and Clef's joint schema head routes evidence to every question
and scores every option together in one forward pass.

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
for practical latency and memory.

## Test

```bash
$env:SYS1_CLEF_TEST_DIR="C:/models/clef-flash"
cargo test --offline -p sys1-clef --test clef -- --ignored --nocapture
```
