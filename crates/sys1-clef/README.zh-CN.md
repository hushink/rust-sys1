# sys1-clef

[English](README.md)

基于 candle 的 Clef（`clef-joint-v1`）。Qwen3.5 骨干读取 state 与 schema，
Clef 的 joint schema head 将状态证据路由到每个问题，并在一次前向中联合打分所有选项。

## 序列

```text
ids = tok(prefix) ++ tok(render(state))[..budget] ++ schema ++ tok(suffix)

prefix = "<|im_start|>system\n…<|im_end|>\n<|im_start|>user\nSTATE:\n"
schema = "\n\nSCHEMA FIELDS:\n"
         每个问题: "\nFIELD i\nID: …\nTYPE: …\nINSTRUCTION: " + instructions
         + "\nALLOWED OPTIONS:\n"
         + 每个选项: "OPTION j: " + canonical({"option_id": …, "description": …}) + "\n"
         + "END FIELD\n"
suffix = "\n<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\nJOINT SCHEMA DECISIONS:"
```

- 字符串原样使用；其他值使用 canonical JSON（key 排序、无空格）。
- Noul 选项为 `[true, false]`；choice 标签按 code point 排序；score 等级保持顺序。
- 默认文本上限 4096 token。state 会裁剪以适配；schema 本身放不下则拒绝。
- 头部读取 span 均值、用 untied LM head 得到 lexical 向量，经过两层 evidence
  routing 和四层 decoder，再将选项与 field 打分。

概率为头部 logits 的普通 softmax。

## 文件

```text
config.json
tokenizer.json
joint_head_config.json
joint_head.safetensors
model-*.safetensors          # Qwen3.5 分片（vision tensor 不使用）
```

9B 的 `clef-flash` BF16 约 18 GB；实际使用建议 `dtype = "bf16"` 并配合 GPU。
12 GB 显存放不下 9B 主干，因此下列测试使用 CPU + BF16（约需 30 GB 内存）。

## 测试

```bash
$env:SYS1_CLEF_TEST_DIR="C:/models/clef-flash"
cargo test --release --offline -p sys1-clef --test clef -- --ignored --nocapture
```

9B checkpoint 需设置 `SYS1_CLEF_DTYPE=bf16`（默认测试 dtype 为 F32）。
`matches_the_recorded_triage_decisions` 用 Ollaya 记录的 `clef-flash`
integration corpus 校验 choice/noul/score 的决策。
