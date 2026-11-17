# 磁盘缓存按字节配额与 LRU 淘汰

`[hub] max_cache_bytes` / `--max-cache-bytes` / `SYS1_MAX_CACHE_BYTES` 设置 `<cache_dir>/models` 的字节上限；`None` 或 `0` 表示不限制。每次远程 pull 在下载前估算缺失文件字节数，按 LRU 淘汰非驻留条目，直到 `used + additional <= limit`。`local:` 源不复制进缓存，不占配额。

**考虑过的替代方案**：只提供 `sys1 rm` 手工清理（无法防止磁盘耗尽）；按条目数而不是字节数限制（模型大小差异太大）；在模型加载时淘汰（会与请求路径耦合，且无法覆盖 `sys1 pull`）；使用 atime（Windows/部分文件系统不可靠）；后台清理线程（增加生命周期与并发复杂度，收益有限）。

**取舍**：LRU 以 `model.json` 的 mtime 为序，pull 成功（含缓存命中）时刷新；它近似“最近物化/使用”，而不是每次推理访问。`Hub` 通过 `CacheResidency` 查询常驻模型，`InferenceEngine` 与 CLI 自动安装基于 `ModelRegistry` 的实现；没有 provider 的裸 `Hub` 把所有条目视为非驻留。进行中的 pull 与被拉取的条目受保护，`remove` / `remove_cached` 在下载期间返回 `CacheBusy`。超出配额且无法通过淘汰解决时返回 `CacheQuotaExceeded`，不会先删除再失败。`GET /api/cache` 与 `sys1 list` 暴露 `limit/used/free/reclaimable`。

**后果**：缓存内容可能在两次 pull 之间被淘汰，需要保留的模型应保持常驻或提高配额。`sys1 rm` 是显式删除，无法感知其他进程的常驻状态，文档要求先停止/卸载其他进程。手动删除与配额淘汰共享“常驻/活动下载受保护”的规则，但服务端删除仍返回 409 以保持现有 API 语义。
