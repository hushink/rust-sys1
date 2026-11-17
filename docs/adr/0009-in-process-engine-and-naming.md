# 高层进程内引擎放 umbrella，Laya 前缀 API 只留在 sys1-laya

`sys1` umbrella 新增 `InferenceEngine`：`EngineOptions` 聚合 hub、注册表与模型选项，Engine 负责「引用 → hub 物化 → manifest → factory 注册 → 注册表 → typed 推理」全链路，提供 `predict`、`systemone`、`predict_batch`、`add` / `add_as`、`load` / `unload` / `remove`、`models` / `model` / `loaded_models`。Engine 实现 `ModelResolver`，可直接用于 `sys1-server` 的 `/api/models` 运行时管理。命名边界同时收紧：`Laya*` 类/函数/方法只在 `sys1-laya` 中定义；其他 crate 使用通用名或导入别名（`HubModelResolver`、`make_model_options`、`CheckpointFactory`、`ModelOptions`），umbrella 根不再扁平导出 `LayaModel` / `LayaOptions` / `LayaFactory` / `LayaRouter`，需要具体实现时走 `sys1::laya`。checkpoint 布局辅助（`is_checkpoint_directory`、`local_checkpoint_manifest`、`register_checkpoint_bundle` 等）也移入 `sys1-laya`，CLI 与 Engine 共用。

**考虑过的替代方案**：把 Engine 放 `sys1-laya`（SDK 会显得 Laya 专属，违背可替换模型族目标）；让 Engine 对 factory/backend 泛型化（v1 只有 Laya，trait 复杂度与测试成本过高，待第二种 family 落地再抽象）；保留 umbrella 的 Laya 扁平导出（容易让用户误以为库绑定 Laya）；在 Engine 与 CLI 各写一份本地 checkpoint 清单（两处漂移）。CLI 继续使用通用命名的组合层 resolver，不把 hub 拉进 `sys1-core`。

**后果**：umbrella 根导出变为通用 API，`LayaModel` 等需从 `sys1::laya` 引用（pre-1.0 的破坏性调整）；新增模型族时可复用 Engine 的命名与生命周期接口，不必再暴露族名；本地裸 checkpoint 的 manifest 生成与 router/variant 注册只有一份实现；Engine 目前绑定 Laya 后端，泛型化留待第二种 backend。
