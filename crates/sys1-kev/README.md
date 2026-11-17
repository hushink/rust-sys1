# sys1-kev

[简体中文](README.zh-CN.md)

Kev (`kev-pointer-v1`) implemented with candle. Kev is a Qwen3.5 base with an
optional PEFT LoRA adapter and a two-projection pointer head.

## Directory layout

The Kev repository ships only the adapter, head and tokenizer. Put the Qwen3.5
base beside them, in a `base/` subdirectory, or let the loader pull the base
declared by the checkpoint:

```text
kev/
  adapter_config.json   # base_model_name_or_path = Qwen/Qwen3.5-0.8B-Base
  adapter_model.safetensors
  head.pt               # records base, base_revision and calibrated temperature
  tokenizer.json
  base/                 # optional; or put these files directly in kev/
    config.json
    model.safetensors
```

`KevOptions::base_directory` (CLI: `[families.kev] base_dir`) overrides the
location. Otherwise `resolve_base_directory` reads `kev.json`, `head.pt`,
`provenance.json`, or `adapter_config.json`, appends the declared revision, and
hands the reference to the caller's pull callback. LoRA is merged into the base
linear layers at load time with `W + α/r · B A`.

## Rows

```text
[state] user(render(state))[:8191]
  [question] user(render(instructions))
  ([option_open] user(option) [option_close])*
  [decide]
```

`user(text)` rewrites every `<|name|>` to `<¦name¦>` so text cannot create
delimiters. Option scores are `k(h[option_close]) · q(h[decide]) / √256`;
softmax uses, in order: `KevOptions::temperature`, `kev.json`, the calibrated
temperature embedded in `head.pt`, and finally 2.351. The 4B/9B checkpoints
therefore use their own fitted temperature without extra configuration.

## Test

```bash
$env:SYS1_KEV_TEST_DIR="C:/models/kev-0.8b"
cargo test --offline -p sys1-kev --test kev -- --ignored --nocapture
```
