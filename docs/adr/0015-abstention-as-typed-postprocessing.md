# Abstention 作为 typed 后处理结果

`DecisionModel::predict_with` / `predict_batch_with` 接收 `PredictionOptions { min_confidence }`。阈值支持标量或按 bucket 的映射（`choice:2` … `noul:11+`，以及 `default`）。gate 在推理后运行：`answer_confidence` 取答案概率的最大值（`noul` 取其 P(true)，概率缺失时回退到报告的 confidence），低于阈值记为 `abstained`，没有可用 confidence 记为 `unevaluated`，其余为 `passed`。结果写入 `DecisionOutput::abstentions`，每个答案带 `low_confidence` 与解析后的 `abstention_threshold`；答案本身与官方 wire 响应不变。

**考虑过的替代方案**：直接给 `ChoiceAnswer` / `ScoreAnswer` / `NoulAnswer` 增加公开字段（会改变 wire 序列化，且不是所有调用方都需要）；把 `min_confidence` 放进 `LayaOptions` 作为模型级配置（官方是逐调用参数，且同一模型不同调用可设不同策略）；在各后端分别实现 gate（candle/ONNX/未来模型会重复并漂移）；服务端全局配置强制 gate（会改变 `/v1` 响应形状）。

**取舍**：gate 是通用后处理，不依赖模型实现，candle、ONNX 与未来后端共享同一套 bucket/阈值语义；`predict_with` 默认调用 `predict`，`predict_batch_with` 在后端原生 batch 之后逐结果 gate，因此不牺牲 batching。与官方一致，`predict_long` 不接受 `min_confidence`；窗口循环没有单一 confidence 可以 gate。阈值为 `0.0` 时 gate 仍会运行，所有可评估答案报告 `passed`。

**后果**：typed 调用方读取 `output.abstention(id)`；CLI `sys1 run --min-confidence` 把 `abstention`、`abstention_threshold` 与（仅低置信时）`low_confidence: true` 注入答案 JSON。HTTP wire 不增加字段。英文 checkpoint 的标量与 bucket gate 已与官方 `system_one(min_confidence=...)` 对齐。
