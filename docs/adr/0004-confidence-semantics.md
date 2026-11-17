# `/v1/*` 默认使用 TypeSafe 归一化 confidence

官方 schema 不规定 `confidence` 的计算公式。`/v1/*` 默认使用与 ollaya 对齐的 `(K·p_max-1)/(K-1)`，以保证客户端阈值行为一致；提供配置切换到 Laya 原生熵置信度 `1 - H(p)/ln(k)`，Laya 原生值通过后续 `/api/*` 或响应头暴露。两者不能在同一响应中混用。

**考虑过的替代方案**：始终输出 Laya 原生熵置信度（与 golden vectors 一致，但与 TypeSafe/ollaya 语义不同）；只输出归一化值、不保留原生值。

**后果**：Laya golden vectors 的 confidence 需要转换，或在 Laya-native 模式下比对；API 文档必须写明当前公式与切换方式。
