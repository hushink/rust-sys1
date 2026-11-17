# 运行时模型管理：可变目录 + 可选 resolver + `/api` 管理端点

`ModelRegistry` 的目录改为 `RwLock<Catalog>`，支持 `register` / `unregister`，因此无需重启即可添加、移除、加载、卸载和重载模型。新模型通过 `ModelResolver` 解析外部引用（`hf:`、`local:`、manifest URL），resolver 负责物化并让 factory 能按 manifest 名称加载；`sys1-core` 不依赖 hub。管理端点在 `/v1` 之外（`/api/models`），避免污染官方契约；库内默认关闭（`ServerConfig::enable_admin = false`），`sys1 serve` 默认开启并可用 `--no-admin` / `[server] admin = false` 关闭。运行时增删不持久化，重启后以配置中的 manifest 为准。

**考虑过的替代方案**：只能重启换模型（不可运维）；把模型目录塞进 `/v1`（破坏官方契约）；在 server 内直接依赖 hub/laya 做下载（破坏 `sys1-server` 与模型族解耦）；动态插件 ABI（上一轮已否决）。注册表读路径保留 `RwLock` 而非全局 `Mutex`，避免请求解析与管理员操作互相阻塞。

**后果**：解析每个模型名会克隆 manifest（含文件列表），在高 QPS 下产生小量分配，但相对推理可忽略；`unregister` 与在途请求存在自然竞态——已持有 `Arc<dyn DecisionModel>` 的请求正常完成，后续请求才看到目录变化；动态添加若与现有名称冲突返回 409 且保留原目录，但 resolver 可能已重写 factory 映射，因此 `LayaFactory::register` 采用“已存在则不覆盖”，替换需先 `unregister`；管理端点与 `/v1/*` 共用 Bearer key，未配置 key 时开放，必须保持绑定地址私密；运行时添加的模型不会写回 `sys1.toml`。
