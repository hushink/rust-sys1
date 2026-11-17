# Laya 路由与 act head 的官方语义对齐

`LayaRouter` 按官方优先级选择 checkpoint：显式 `model` > 显式 `task` > typed-decisions 工作流（仅 `auto_task_detection`）> 显式 `lang` > `lang_guess` > 语言检测。检测覆盖文字系统、拉丁语停用词画像、缩写/全大写与混合片段，并生成与 Python `repr` 一致的 reason。`LayaRouterOptions::default` 为 `multilingual` + `auto_task_detection = false`，与官方默认一致；CLI 通过 `--laya-router-default`、`--laya-auto-task-detection`、`--laya-lang-guess` 与 `[server]` 同名配置暴露这些设置。

checkpoint 若包含 `act_head.*`，`LayaModel` 在打分后计算 `action.act_probability`，写入 `DecisionOutput.actions`。该字段只存在于进程内 typed API；官方 wire `SystemOneResponse` 不携带它。ONNX 导出暂不包含 act head，因此 `LayaOnnxModel` 的 `actions` 为空。

**考虑过的替代方案**：继续使用启发式路由（与官方行为不一致）；默认开启 typed-decisions 工作流匹配（会改变官方默认决策）；把 actions 塞进 `answers`（破坏 answer 类型与官方 JSON）；为 ONNX 单独维护一套动作头权重（导出产物尚未包含 pool 输出与 head 参数）；等官方 ONNX 导出后再实现 act head（candle 后端已可验证）。

**取舍**：路由行为以官方 golden 为准（159/159 用例的 model、workflow、reason 与 detection 字段全部一致），而不是按“看起来合理”重写。`actions` 是显式扩展，序列化官方响应时会丢弃，避免协议漂移。act head 只在权重实际存在时启用，缺少张量的 checkpoint 保持原行为。

**后果**：使用 ONNX 后端的调用方最初拿不到 `actions`；ADR 0016 通过 pooled 输出与 `act_head.safetensors` sidecar 补齐，修改 wire 协议不是必要条件。路由 golden 与真实 checkpoint 的 act head 对齐都需要本地官方 vectors，分别由 `SYS1_LAYA_ROUTING_GOLDEN` 与 `SYS1_LAYA_TEST_ROOT` / `SYS1_LAYA_GOLDEN_DIR` 驱动，默认跳过。
