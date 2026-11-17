# 长上下文扫描作为 typed 进程内能力

`DecisionModel::predict_long` 接收 `LongContextOptions { window, stride, batch_size }`：state 只 tokenize 一次，按 token 切分重叠窗口（默认窗口为 `max(64, max_len - head_max_len - 8)` 并受问题实际 state 余量限制，默认 stride 为窗口一半），逐窗口打分后聚合。`noul` 取各窗口 P(true) 的最大值；`choice` / `score` 取最大答案概率所在的窗口，避免中性窗口多数票淹没局部信号。每个答案的 `DecisionOutput.windows` 记录决定窗口的 `index`、`token_start`、`token_end` 与 `count`，`DecisionOutput.scored_windows` 对应官方 `usage.windows`。state 能放进一个窗口时直接走普通 `predict`，不产生窗口归因。

**考虑过的替代方案**：继续静默截断（丢失超出 `max_len` 的内容）；按字符切分再 tokenize（窗口边界不受控）；对所有窗口概率求平均（长文档中大量中性窗口会稀释局部证据）；增加 HTTP 端点（官方 wire 没有长上下文接口，会扩大协议面）；等待官方提供长上下文 ONNX 图。

**取舍**：窗口按原始 token id 切片，而不是像官方 Python 那样 decode 回文本再 tokenize；这避免 BPE 边界漂移，窗口长度严格受控。英文 checkpoint 上同一 state、问题、窗口与 stride 的结果与官方 `predict_long` 一致：答案、`action.act_probability`、窗口归因和 `usage.input_tokens` 均在 1e-3 / 精确值内。返回的概率与置信度属于决定窗口，不是整篇文档的校准值；调用方应结合窗口归因判断证据位置。`predict_long` 默认以 `Unsupported` 拒绝未实现该能力的模型，避免退化回截断。

**后果**：`predict_long` 只存在于 typed API 和 `sys1 run --long`，`/v1/systemone` 保持不变。`sys1-laya` 的 candle 与 ONNX 模型共享窗口规划、聚合与归因；`LayaRouter` 先路由整篇 state，再调用选中 checkpoint 的扫描。长扫描的 token 成本是各窗口序列之和（含重叠），`usage.input_tokens` 会高于单窗口调用。
