# 可替换 telemetry + W3C traceparent；日志由组合层安装

`sys1-server` 不安装全局 logger，只发 `tracing` span/事件，并通过 `Telemetry` trait 暴露结构化事件：`RequestEvent`、`ModelEvent`、`BatchEvent`、`CacheEvent`，均携带 `request_id`/`trace_id`。默认 `NoopTelemetry`；CLI 注入 `TracingTelemetry` 并负责安装 `tracing-subscriber`。嵌入方可通过 `ServerExtensions { telemetry, resolver, cache }` 注入自定义 sink（审计、OpenTelemetry 等），也可自行安装任意 subscriber。每个 HTTP 响应写入 W3C `traceparent`：合法入站 trace id 保留，服务端生成新的 span id 与 sampled 标志；无入站头则生成新 trace，同时保留 `x-typesafe-request-id`。CLI 日志支持 text/JSON、文件追加与过滤器；优先级 CLI / `SYS1_LOG_*` > `RUST_LOG` > `[server] log_*` > 默认。

**考虑过的替代方案**：直接依赖 OpenTelemetry SDK 并硬编码 exporter（依赖重、嵌入方无法选择）；使用 `log` facade + 自定义全局 logger（与现有 tracing span/字段能力不匹配）；只在库内安装 subscriber（嵌入方无法替换，且多次初始化冲突）；只做 metrics 不做审计事件（缺少按请求/模型的追踪语义）。日志格式化放在 CLI 而非 server，使库保持无 I/O、无全局状态。

**后果**：库使用者必须自带 subscriber 或 telemetry sink；CLI 的 JSON 日志需要 `tracing-subscriber/json` feature（仅 CLI 编译期依赖）；trace id 只做传播与关联，不内置 exporter/采样后端；`Telemetry` 同步调用，sink 应快速返回或自行缓冲，避免阻塞推理路径；未来接入 OpenTelemetry 只需实现 `Telemetry`，无需改 server 核心。
