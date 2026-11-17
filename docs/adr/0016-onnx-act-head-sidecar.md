# ONNX act head 通过 pooled 输出与权重 sidecar 补齐

ONNX 导出的计算图新增第二个输出 `pooled`：decision head 之后 token 0 的表示，形状 `[batch, hidden]`。导出器同时把 checkpoint 的 `act_head.0/2` 权重写入 `act_head.safetensors`，并在 `onnx.json` 记录 `pooled_output` 与 `act_head`。`LayaOnnxModel` 检测到 sidecar 后请求 pooled 输出，用与 candle 后端相同的 `ActHead`（candle `Linear` + GELU(erf) + softmax）计算 `action.act_probability`；没有 sidecar 的旧导出继续返回空 `actions`。`sys1-ort` 的 `ScoringOutputs` 增加可选 `pooled`，并要求其元素数能被 batch 整除。

**考虑过的替代方案**：把整个 act head 编进主 ONNX 图（需要在图内按可变 marker 数计算 top1/top2/熵与 softmax，批次布局复杂）；单独生成第二个小 ONNX 图并由 Rust 管理两个 session（多一个运行时对象与 I/O 契约）；每次推理输出 act 权重张量（约 2 MB/次的无谓拷贝）；从原 `model.safetensors` 读取 head（ONNX 导出目录不保证带原始权重）；只保留 candle act head（ONNX 后端行为不一致）。

**取舍**：sidecar 只有约 1 MB，避免修改主图的计算逻辑，也不增加第二 session。`LayaOnnxModel` 依赖 candle 仅用于 1 MB 的 head 数学，主前向仍完全由 ONNX Runtime 执行。`pooled` 输出使导出图比原来多一个 `[batch, hidden]` 张量，数据量相对 logits 很小。旧导出无需重新导出即可继续工作，只是 `actions` 为空。

**后果**：英文 checkpoint 上 ONNX 与 candle 的答案、`action.act_probability` 在 2e-3 内一致（`onnx_parity` 覆盖）；candle 侧与官方 golden 的 act 对齐保持 1e-3。需要 actions 的 ONNX 部署必须用包含 pooled 输出与 sidecar 的新导出；`Synthetic` 测试图也新增了 pooled 输出以覆盖 ORT 解包。
