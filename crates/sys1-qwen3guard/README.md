# sys1-qwen3guard

[简体中文](README.zh-CN.md)

Qwen3Guard-Gen as a fixed-preset safety decision model, implemented with
candle. The model's policy, labels and output format live in its chat template,
so it answers only its own preset questions:

| id | type | options |
|---|---|---|
| `safety` | choice | `safe`, `controversial`, `unsafe` |
| `unsafe` | noul | P(Unsafe) |
| `unsafe_strict` | noul | P(Unsafe or Controversial) |
| `category` | choice | `none`, `violent`, `non_violent_illegal`, `sexual`, `pii`, `suicide_self_harm`, `unethical`, `politically_sensitive`, `copyright`, `jailbreak` |

Use `Qwen3GuardModel::preset_questions()` to obtain the exact question
definitions. Any other question is rejected.

## Rows

```text
base = prompt_head + state_text + prompt_tail   # the user branch of the chat template
r0   = tok(base + "Safety:")
r1   = tok(base + "Safety: Unsafe\nCategories:")
```

Both rows are causal; the backbone is read at the last real token and restricted
to the first token of each label. `category` reuses the teacher-forced `r1`
row. The model is a dense Qwen3-0.6B fine-tune; weights are read from
`model.safetensors` and the tokenizer from `tokenizer.json`.

## Test

```bash
$env:SYS1_QWEN3GUARD_TEST_DIR="C:/models/Qwen3Guard-Gen-0.6B"
cargo test --offline -p sys1-qwen3guard --test qwen3guard -- --ignored --nocapture
```

The test checks a safe weather question and a bomb request against the
transformers reference.

## Limits

- Preset questions only; arbitrary questions are a 422.
- Rows longer than 32k tokens are rejected, never truncated.
- Response moderation (`{prompt, response}` with `Refusal:`) is not exposed.
