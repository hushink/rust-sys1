# sys1-kev

[English](README.md)

基于 candle 的 Kev（`kev-pointer-v1`）。Kev 由 Qwen3.5 base、可选的 PEFT LoRA
adapter 和两投影 pointer head 组成。

## 作为库使用

```toml
[dependencies]
sys1-kev = { git = "https://github.com/hushink/rust-sys1" }
sys1-core = { git = "https://github.com/hushink/rust-sys1" }
sys1-protocol = { git = "https://github.com/hushink/rust-sys1" }
# 或从本地 checkout 引入：
# sys1-kev = { path = "../rust-sys1/crates/sys1-kev" }
```

```rust
use std::path::Path;
use sys1_core::{DecisionModel, Questions};
use sys1_kev::{KevModel, KevOptions};
use sys1_protocol::State;

// base 必须与 adapter 相邻或位于 base/；否则设置
// KevOptions { base_directory: Some(path), .. }，或交给 engine 自动拉取。
let model = KevModel::load_dir(Path::new("/models/kev-0.8b"), KevOptions::default())?;
let output = model.predict(
    &State::text("I was charged twice and want a refund."),
    &Questions::new()
        .choice("team", [("billing", "Charges and refunds"), ("tech", "Bugs")])
        .noul("refund", "The customer asks for a refund."),
)?;
let team = output.answers.choice("team")?;
```

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
$env:SYS1_KEV_BASE_DIR="C:/models/qwen3.5-0.8b-base"
cargo test --offline -p sys1-kev --test kev -- --ignored --nocapture
```

4B checkpoint 使用单独测试（base 从 `base/` 解析）：

```bash
$env:SYS1_KEV4B_TEST_DIR="C:/models/kev-4b"
cargo test --release -p sys1-kev --test kev predicts_with_the_4b_checkpoint \
  -- --ignored --nocapture
```

4B 的 CPU F32 输出与独立的 PyTorch LoRA 合并 + pointer head 参考在 1e-6 内
一致；12 GB CUDA 卡上的 BF16 输出误差在 0.003 内。

9B checkpoint 需要约 19 GB 内存，4 个分片放不进 12 GB 显存，因此使用
CPU BF16：

```bash
$env:SYS1_KEV9B_TEST_DIR="C:/models/kev-9b"   # 含 base/
cargo test --release -p sys1-kev --test kev predicts_with_the_9b_checkpoint \
  -- --ignored --nocapture --test-threads=1
```

BF16 概率与独立的 PyTorch BF16 参考相差不超过 0.006；`head.pt` 中的温度为
2.297397。
