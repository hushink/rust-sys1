# sys1-kev

[简体中文](README.zh-CN.md)

Kev (`kev-pointer-v1`) implemented with candle. Kev is a Qwen3.5 base with an
optional PEFT LoRA adapter and a two-projection pointer head.

## Directory layout

The KeV repository ships only the adapter, head and tokenizer. Put the Qwen3.5
base beside them or in a `base/` subdirectory:

```text
kev/
  adapter_config.json
  adapter_model.safetensors
  head.pt
  tokenizer.json
  base/                 # or put these files directly in kev/
    config.json
    model.safetensors
```

`KevOptions::base_directory` overrides the location. LoRA is merged into the
base linear layers at load time with `W + α/r · B A`.

## Rows

```text
[state] user(render(state))[:8191]
  [question] user(render(instructions))
  ([option_open] user(option) [option_close])*
  [decide]
```

`user(text)` rewrites every `<|name|>` to `<¦name¦>` so text cannot create
delimiters. Option scores are `k(h[option_close]) · q(h[decide]) / √256`;
softmax over the checkpoint temperature (from `kev.json` when present, otherwise
2.351 for kev-0.8b).

## Test

```bash
$env:SYS1_KEV_TEST_DIR="C:/models/kev-0.8b"
cargo test --offline -p sys1-kev --test kev -- --ignored --nocapture
```
