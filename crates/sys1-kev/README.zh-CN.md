# sys1-kev

[English](README.md)

基于 candle 的 Kev（`kev-pointer-v1`）。Kev 由 Qwen3.5 base、可选的 PEFT LoRA
adapter 和两投影 pointer head 组成。

## 目录布局

Kev 仓库只提供 adapter、head 和 tokenizer。把 Qwen3.5 base 放在同目录或
`base/` 子目录：

```text
kev/
  adapter_config.json
  adapter_model.safetensors
  head.pt
  tokenizer.json
  base/                 # 或直接放在 kev/ 下
    config.json
    model.safetensors
```

`KevOptions::base_directory` 可覆盖该位置。加载时按 `W + α/r · B A` 将 LoRA
合并进 base 线性层。

## 行

```text
[state] user(render(state))[:8191]
  [question] user(render(instructions))
  ([option_open] user(option) [option_close])*
  [decide]
```

`user(text)` 会把 `<|name|>` 改写为 `<¦name¦>`，避免用户文本产生分隔符。选项分数为
`k(h[option_close]) · q(h[decide]) / √256`；按 checkpoint 温度 softmax（存在
`kev.json` 时读取，否则 kev-0.8b 默认 2.351）。

## 测试

```bash
$env:SYS1_KEV_TEST_DIR="C:/models/kev-0.8b"
cargo test --offline -p sys1-kev --test kev -- --ignored --nocapture
```
