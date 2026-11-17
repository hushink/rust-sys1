# 配置热加载只做 `[models]` 安全增删，其他变更要求重启

`POST /api/config/reload` 通过 `ConfigReloader` trait 注入；CLI 的实现持有配置文件路径、`HubModelResolver`、注册表和上一次的 `FileConfig` 快照。reload 时重新解析 TOML：若 `[hub]`/`[server]` 改变，或已有名称的 ModelRef 改变，返回 `409 restart required` 且不修改任何状态（避免半套配置）；新增模型经 hub 解析、以 `register_configured` 写入（`origin=configured`），移除项从运行时目录卸载并释放 factory 映射，磁盘缓存保留。新增阶段失败会回滚已注册项与 factory 状态，并返回 `400`。reload 与 admin add/remove 共用 server 的 admin lock；CLI 内另有 reload 锁防止并发重读。

**考虑过的替代方案**：全量重建 registry（在途请求、已加载权重与动态 unload 状态会丢失）；允许任意设置热更新（bind、device、dtype、logging、batch 均有启动期状态，需逐个重构）；同引用原地替换（factory 当前“已存在不覆盖”，需先卸载，失败时难以原子回滚）；只做进程重启（运维成本高，模型目录常变）。选择拒绝式安全增量，让失败可预测且不破坏当前服务。

**后果**：修改 `[models]` 可在线完成；修改 hub/server/已有引用必须重启；新增模型解析失败不会留下半注册状态；被移除的模型若有在途请求会自然完成，后续请求得到 unknown model；配置中的模型名和引用映射会同步到 admin add 的名称解析；该能力要求启动时显式加载配置文件，纯默认配置无 reload 端点。
