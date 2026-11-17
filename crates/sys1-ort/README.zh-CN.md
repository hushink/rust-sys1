# sys1-ort

[English](README.md)

rust-sys1 的 ONNX Runtime 执行层。它负责 session 生命周期与 System One
token-scoring 契约；模型 family 保留 tokenization、序列布局、校准与答案格式化。

## 用法

```rust
use sys1_ort::{ScoringInputs, ScoringSession, SessionConfig};

let session = ScoringSession::load(SessionConfig::new("model.onnx"))?;
let logits = session.score(ScoringInputs {
    input_ids: &[1, 2, 3, 4],
    attention_mask: &[1, 1, 1, 1],
    type_ids: &[0],
    markers: &[1, 3],
    batch: 1,
    seq: 4,
})?;
assert_eq!(logits.len(), 2);
```

`SessionConfig` 可覆盖所有输入/输出名，并设置 intra-op/inter-op 线程数。
`ScoringInputs::validate` 会在调用运行时前拒绝形状不匹配与超出
`0..batch * seq` 的 marker 索引。

## 契约

| 输入 | 类型 | 形状 |
| --- | --- | --- |
| `input_ids` | int64 | `[batch, seq]` |
| `attention_mask` | int64 | `[batch, seq]` |
| `type_ids` | int64 | `[batch]` |
| `markers` | int64 | `[n]` |

`markers` 是 `[batch * seq]` 的扁平位置。模型在每个 marker 处 gather hidden
state，并输出形状 `[n]` 的 float32 `logits`。batch 与 seq 轴必须是动态的。
`Session::run` 接收 `&mut self`，因此 `ScoringSession` 用 mutex 串行化并发 run；
单次 run 内部由 ONNX Runtime 并行。

## 运行时加载

本 crate 使用 `ort` 的 `load-dynamic` feature 构建，构建期不会下载 ONNX Runtime。
创建 session 时通过 `ORT_DYLIB_PATH` 或平台加载路径查找共享库。每个进程只尝试
加载一次；结果（包括解析后的路径）会被缓存，失败以 `OrtError::Runtime` 返回，
不会 panic。绑定以 API 17 为目标，适用于 ONNX Runtime 1.17 及更新版本。

执行提供器是 Cargo features：`cuda`、`tensorrt`、`directml`、`coreml`。它们
跟随 `ort` crate，需要安装对应的运行时与 SDK。TensorRT 仅能通过该执行提供器路径使用。

## 测试

```bash
SYS1_REQUIRE_ORT=1 cargo test -p sys1-ort
```

`tests/fixtures/scoring_stub.onnx` 是由
`tests/fixtures/generate_scoring_stub.py` 生成的小型合成计算图。它返回每个 marker
处的 token id，用于验证 batching、padding、marker 索引与输出提取。未设置
`SYS1_REQUIRE_ORT=1` 时，若 ONNX Runtime 无法加载则跳过测试。

## 范围

本 crate 不是 `ModelFactory`，不感知 `sys1-protocol` 问题或 Laya；`sys1-laya`
等 family crate 负责把协议请求映射到 scoring 契约并实现 `DecisionModel`。
