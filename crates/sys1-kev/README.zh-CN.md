# sys1-kev

[English](README.md)

基于 candle 的 Kev（`kev-pointer-v1`）。Kev 由 Qwen3.5 base、可选的 PEFT LoRA
adapter 和两投影 pointer head 组成。

## 目录布局

Kev 仓库只提供 adapter、head 和 tokenizer。把 Qwen3.5 base 放在同目录、
`base/` 子目录，或让加载器按 checkpoint 声明的仓库自动拉取：

```text
kev/
  adapter_config.json   # base_model_name_or_path = Qwen/Qwen3.5-0.8B-Base
  adapter_model.safetensors
  head.pt               # 记录 base、base_revision 与标定温度
  tokenizer.json
  base/                 # 可选；也可直接放在 kev/ 下
    config.json
    model.safetensors
```

`KevOptions::base_directory`（CLI：`[families.kev] base_dir`）可覆盖该位置。
否则 `resolve_base_directory` 依次读取 `kev.json`、`head.pt`、
`provenance.json`、`adapter_config.json`，补上声明的 revision，并把引用交给
调用方的 pull 回调。加载时按 `W + α/r · B A` 将 LoRA 合并进 base 线性层。

## 行

```text
[state] user(render(state))[:8191]
  [question] user(render(instructions))
  ([option_open] user(option) [option_close])*
  [decide]
```

`user(text)` 会把 `<|name|>` 改写为 `<¦name¦>`，避免用户文本产生分隔符。选项分数为
`k(h[option_close]) · q(h[decide]) / √256`；softmax 温度依次取
`KevOptions::temperature`、`kev.json`、`head.pt` 内嵌的标定值，最后才用 2.351。
因此 4B/9B checkpoint 无需额外配置即可使用各自拟合的温度。

## 测试

```bash
$env:SYS1_KEV_TEST_DIR="C:/models/kev-0.8b"
cargo test --offline -p sys1-kev --test kev -- --ignored --nocapture
```
