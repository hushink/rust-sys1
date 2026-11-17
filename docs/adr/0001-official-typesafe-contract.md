# 以官方 TypeSafe 契约作为唯一兼容基线

我们只为 `POST /v1/systemone` 与 `GET /v1/models` 实现官方 OpenAPI 语义，错误体使用官方 `detail` 形状，不实现 `/v1/decisions` 等非官方别名。sys1 0.0.3 与 ollaya 的偏差（`/v1/models` 形状、错误码、Laya 私有字段）不进入 `/v1/*`；Laya 的能力缺口用 422 `detail[]` + `type: "unsupported"` 表达，而不是静默忽略。

**考虑过的替代方案**：兼容 sys1 0.0.3 的 HTTP 契约；实现 ollaya 的超集；改用 OpenAI 风格。这些都会让“兼容”无法用官方客户端验收。

**后果**：现有 sys1 0.0.3 客户端不能直接迁移；需要迁移文档与兼容性测试。非官方扩展只能放在 `/api/*` 或响应头。
