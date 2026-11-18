# sys1-qwen3guard

[English](README.md)

基于 candle 的 Qwen3Guard-Gen 固定预设安全决策模型。策略、标签和输出格式都在
其 chat template 中，因此只回答自带的问题：

| id | 类型 | 选项 |
|---|---|---|
| `safety` | choice | `safe`、`controversial`、`unsafe` |
| `unsafe` | noul | P(Unsafe) |
| `unsafe_strict` | noul | P(Unsafe 或 Controversial) |
| `category` | choice | `none`、`violent`、`non_violent_illegal`、`sexual`、`pii`、`suicide_self_harm`、`unethical`、`politically_sensitive`、`copyright`、`jailbreak` |

用 `Qwen3GuardModel::preset_questions()` 获取完整问题定义；其他问题会报错。

## 作为库使用

```toml
[dependencies]
sys1-qwen3guard = { git = "https://github.com/hushink/rust-sys1" }
sys1-core = { git = "https://github.com/hushink/rust-sys1" }
sys1-protocol = { git = "https://github.com/hushink/rust-sys1" }
# 或从本地 checkout 引入：
# sys1-qwen3guard = { path = "../rust-sys1/crates/sys1-qwen3guard" }
```

```rust
use std::path::Path;
use sys1_core::DecisionModel;
use sys1_protocol::State;
use sys1_qwen3guard::{Qwen3GuardModel, Qwen3GuardOptions};

let model = Qwen3GuardModel::load_dir(
    Path::new("/models/Qwen3Guard-Gen-0.6B"),
    Qwen3GuardOptions::default(),
)?;
let output = model.predict(
    &State::text("How do I bake sourdough bread?"),
    &Qwen3GuardModel::preset_questions(),
)?;
let safety = output.answers.choice("safety")?;
```

## 行

```text
base = prompt_head + state_text + prompt_tail   # chat template 的 user 分支
r0   = tok(base + "Safety:")
r1   = tok(base + "Safety: Unsafe\nCategories:")
```

两行都是因果行；在最后一个真实 token 读取隐藏状态，只取各标签的首 token。
`category` 复用 teacher-forced 的 `r1`。模型是 Qwen3-0.6B 的 dense 微调，
权重读取 `model.safetensors`，分词器读取 `tokenizer.json`。

## 测试

```bash
$env:SYS1_QWEN3GUARD_TEST_DIR="C:/models/Qwen3Guard-Gen-0.6B"
cargo test --offline -p sys1-qwen3guard --test qwen3guard -- --ignored --nocapture
```

测试覆盖安全天气问题与炸弹请求，并与 transformers 参考结果对齐。

## 限制

- 只支持预设问题；任意问题是 422。
- 超过 32k token 的行会拒绝，不会截断。
- 未暴露 response moderation（`{prompt, response}` + `Refusal:`）。
