# 高层进程内引擎放 umbrella，Laya 前缀 API 只留在 sys1-laya

`sys1` umbrella 新增 `InferenceEngine`：`EngineOptions` 聚合 hub、注册表与模型选项，Engine 负责「引用 → hub 物化 → manifest → factory 注册 → 注册表 → typed 推理」全链路，提供 `predict`、`systemone`、`predict_batch`、`add` / `add_as`、`load` / `unload` / `remove`、`models` / `model` / `loaded_models`。Engine 实现 `ModelResolver`，可直接用于 `sys1-server` 的 `/api/models` 运行时管理。命名边界同时收紧：`Laya*` 类/函数/方法只在 `sys1-laya` 中定义；其他 crate 使用通用名或导入别名（`HubModelResolver`、`make_model_options`、`CheckpointFactory`、`ModelOptions`）。umbrella 根继续再导出 `LayaModel` / `LayaOptions` / `LayaFactory` / `LayaRouter`，保持原有直接调用方式；再导出不构成新的 Laya 命名，通用 API 仍以 Engine/ModelOptions 为主。checkpoint 布局辅助（`is_checkpoint_directory`、`local_checkpoint_manifest`、`register_checkpoint_bundle` 等）也移入 `sys1-laya`，CLI 与 Engine 共用。

**考虑过的替代方案**：把 Engine 放 `sys1-laya`（SDK 会显得 Laya 专属，违背可替换模型族目标）；让 Engine 对 factory/backend 泛型化（v1 只有 Laya，trait 复杂度与测试成本过高，待第二种 family 落地再抽象）；移除 umbrella 的 Laya 扁平导出（破坏已有用户代码，且再导出不是重新命名）；在 Engine 与 CLI 各写一份本地 checkpoint 清单（两处漂移）。CLI 继续使用通用命名的组合层 resolver，不把 hub 拉进 `sys1-core`。

**后果**：原有 `use sys1::{LayaModel, LayaOptions}` 调用保持不变，`InferenceEngine` 是纯增量入口；新增模型族时可复用 Engine 的命名与生命周期接口，不必再暴露族名；本地裸 checkpoint 的 manifest 生成与 router/variant 注册只有一份实现；Engine 目前绑定 Laya 后端，泛型化留待第二种 backend。`crates/sys1/tests/naming.rs` 会扫描除 `sys1-laya` 外的所有源码/测试/示例，阻止新增 Laya 前缀定义；`sys1` 根部的兼容再导出不在此限制内。
