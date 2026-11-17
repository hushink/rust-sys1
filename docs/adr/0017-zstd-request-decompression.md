# zstd 请求解压使用纯 Rust 解码器

`sys1-server` 在读取请求体时支持 `Content-Encoding: gzip` 与 `zstd`，两者都在进程内同步解压。压缩后的请求体先受 `max_body_bytes` 限制；解压时用 `Read::take(max_body_bytes + 1)` 限制输出，超过上限返回 `413`，损坏帧返回 `400`，其他编码返回 `415`。解压实现不依赖 tower-http 的异步解压层，保持现有 `read_body` 的同步错误语义与响应形状。

**考虑过的替代方案**：`zstd` crate（C 实现，解压更快，但需要 `zstd-sys`/`cc` 与 C 工具链，且增加 unsafe 边界）；`tower-http` 的 `RequestDecompressionLayer`（需要异步 body 流与额外 feature，错误会绕过现有 `ApiError` 路径）；继续返回 `415`（TypeSafe 兼容客户端可能发送 zstd 请求）；自研解码器（不现实）。

**取舍**：选择 `ruzstd` 纯 Rust 解码器，关闭默认 feature、只启用 `std`，因此不引入 xxhash 依赖、不需要 C 编译器，也不增加 unsafe。代价是解压吞吐低于 C zstd；请求体本身受 32 MiB 压缩前上限约束，解码开销不是热路径。`ruzstd` 不启用 `hash` feature，因此不校验帧内容校验和；HTTP 层已用 TLS/完整性传输，且解码后的 JSON 仍会经过严格校验。

**后果**：`Cargo.lock` 增加 `ruzstd 0.9.1`。gzip/zstd 都只支持单层 `Content-Encoding`；多编码或未知编码返回 `415`。测试覆盖正常 zstd 帧、损坏帧与 HTTP 全链路，`ruzstd` 的压缩实现仅用于测试数据生成。
