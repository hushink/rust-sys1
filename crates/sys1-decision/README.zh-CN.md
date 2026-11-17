# sys1-decision

[English](README.md)

基于 candle 的 Decision 1.0（`decision-endpoint-v1`）。完全微调的 Qwen3.5 文本
骨干通过 `decision_head.safetensors` 将每个选项的 endpoint 与全局 query 对比打分。

## 布局

每个问题是一行因果序列：

```text
Context:\n<payload(state)>\n\nTask type: <type>\nQuestion:\n<payload(instructions)>\nOptions:
  \n<option>\n{"key":…,"description":…}\n</option>   （每个选项一个 segment）
  \n\nSelect the single option best supported by the context and instructions.\nDecision:
```

`payload` 对字符串原样输出，其余使用 canonical JSON（key 排序、无空格）。每个
segment 单独分词。头部读取每个选项 segment 的最后一个 token 和整行最后一个 token。

- Choice criteria 为对象，2–255 个选项。
- Score 2–10 个等级。
- Noul 选项为 `[false, true]`；缺失的一侧使用作者默认描述。
- 超过 16384 token 的行会拒绝，不截断。

概率为 `softmax(logits / T)`，T 来自 `config.json`。

## 文件

```text
config.json
decision_config.json
decision_head.safetensors
tokenizer.json
backbone/config.json
backbone/*.safetensors
```

## 测试

```bash
$env:SYS1_DECISION_TEST_DIR="C:/models/Decision-1.0-Eos-0.8B"
cargo test --offline -p sys1-decision --test decision -- --ignored --nocapture
```
