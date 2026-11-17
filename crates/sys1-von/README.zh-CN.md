# sys1-von

[English](README.md)

基于 candle 的 Von 1.1（`von-option-marker-v1`）决策模型。Von 是
ModernBERT-large 编码器加 option-marker scorer：每个选项在序列中占一个
`[MASK]`，从该位置的隐藏状态得到分数。

## 布局

```text
[CLS] <instructions> <state> [SEP] [MASK] <description 0> [MASK] <description 1> ... [SEP]
```

- state 用 Python `str()` 渲染：字符串原样，对象为 `key: value` 行，数组为
  Python `repr`。
- choice 选项取描述；描述为空时取标签。score 等级使用 Von 的
  `what Examples:` 格式。
- noul 以 `[true, false]` 顺序读取，返回 `[false, true]` logits。零样本 noul
  增加一行空 state，并从 true 侧减去 `0.7 · (n_true − n_false)`。
- 超过 8192 token 时按字符边界截断 state；instructions 与选项本身必须能装下。

校准使用 Von 的输入条件温度映射（`von-entropy-length-v1`），clamp 到
`[0.3, 12]`。

## 文件

checkpoint 目录包含 `config.json`、`tokenizer.json` 和 `option_marker.pt`
（上游 PyTorch zip 通过 candle 受限 pickle reader 原地读取）。ModernBERT
配置继承的 2k rotary 上限会按请求的 `max_len` 扩展。

## 测试

```bash
$env:SYS1_VON_TEST_DIR="C:/models/von"
cargo test --offline -p sys1-von --test von -- --ignored --nocapture
```

测试将 choice、noul、score 概率与 `von-sdk==1.1.1` 对比。
