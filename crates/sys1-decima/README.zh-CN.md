# sys1-decima

[English](README.md)

基于 candle 的 Decima（`decima-late-interaction-v1`/`-v2`）。ModernBERT 编码器
（`decima-base` 为 mmBERT）分别编码 state 和每个选项；late-interaction scorer
让每个选项读取 state token 后得到一个原始分数。选项之间不互相注意。

## 行

```text
state 行   [cls] tok(state_prefix + normalize(text + "\n" + state))[..max_state_tokens] [sep]
option 行  [cls] tok(option_prefix + normalize(text + " " + option))[..max_choice_tokens] [sep]
```

- `text` 是问题 instructions；缺失/null 时用 question id。带 true/false 描述的
  noul 会追加 `\nTrue if: …\nFalse if: …`。
- choice 选项为 `label: description`（描述为空时仅 label）；score 为等级文本；
  noul 为 `yes`/`no`。
- `normalize` 为 NFC 后接 Python `str.strip()`。
- state 行超过预算时报 `STATE_TRUNCATED`；option 行按上游行为静默截断。
- `choice_fit`/`option_fit` checkpoint 在完整文本放不下时，把问题适配进选项窗口。

choice/noul 概率为 `softmax(scores / T)`；score 使用 Decima 的 cumulative-link
ordinal head。

## 文件

接受仓库根目录或其 `pytorch/` 目录：

```text
decima.json
encoder/config.json
encoder/model.safetensors
encoder/tokenizer.json
head.safetensors
```

## 测试

```bash
$env:SYS1_DECIMA_TEST_DIR="C:/models/decima-base"
cargo test --offline -p sys1-decima --test decima -- --ignored --nocapture
```

测试用独立的 PyTorch 实现（按文档构建图）对比 choice/noul/score 答案。
