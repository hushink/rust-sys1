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
$env:SYS1_KEV_BASE_DIR="C:/models/qwen3.5-0.8b-base"
cargo test --offline -p sys1-kev --test kev -- --ignored --nocapture
```

The 4B checkpoint uses its own test (the base is resolved from `base/`):

```bash
$env:SYS1_KEV4B_TEST_DIR="C:/models/kev-4b"
cargo test --release -p sys1-kev --test kev predicts_with_the_4b_checkpoint \
  -- --ignored --nocapture
```

The 4B numbers match an independent PyTorch LoRA-merge + pointer-head
reference within 1e-6 on the CPU F32 path; BF16 on a 12 GB CUDA card is within
0.003.

The 9B checkpoint needs about 19 GB of RAM, so it runs BF16 on CPU (its
4 shards do not fit a 12 GB card):

```bash
$env:SYS1_KEV9B_TEST_DIR="C:/models/kev-9b"   # contains base/
cargo test --release -p sys1-kev --test kev predicts_with_the_9b_checkpoint \
  -- --ignored --nocapture --test-threads=1
```

BF16 probabilities stay within 0.006 of the independent PyTorch BF16
reference; `head.pt` carries temperature 2.297397.
