# sys1-decima

[简体中文](README.zh-CN.md)

Decima (`decima-late-interaction-v1`/`-v2`) implemented with candle. A
ModernBERT encoder (mmBERT for `decima-base`) encodes the state and every option
separately; a late-interaction scorer lets each option read the state tokens
before producing one raw score. Options never attend to each other.

## Rows

```text
state row   [cls] tok(state_prefix + normalize(text + "\n" + state))[..max_state_tokens] [sep]
option row  [cls] tok(option_prefix + normalize(text + " " + option))[..max_choice_tokens] [sep]
```

- `text` is the question's instructions, or the question id when absent/null.
  A noul question with true/false descriptions appends
  `\nTrue if: …\nFalse if: …`.
- Choice options are `label: description` (label alone when empty); score levels
  are their text; noul options are `yes`/`no`.
- `normalize` is NFC followed by Python `str.strip()`.
- A state row over the budget is rejected (`STATE_TRUNCATED`); an option row is
  cut silently, as upstream does.
- `choice_fit`/`option_fit` checkpoints fit the question into the option window
  when the full text does not fit.

Choice and noul probabilities are `softmax(scores / T)`. Score questions use
Decima's cumulative-link ordinal head on the same scores.

## Files

The repository root or its `pytorch/` directory is accepted:

```text
decima.json
encoder/config.json
encoder/model.safetensors
encoder/tokenizer.json
head.safetensors
```

## Test

```bash
$env:SYS1_DECIMA_TEST_DIR="C:/models/decima-base"
cargo test --offline -p sys1-decima --test decima -- --ignored --nocapture
```

The test compares choice/noul/score answers with an independent PyTorch
implementation of the documented graph.
