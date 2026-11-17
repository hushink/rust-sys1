# 单进程、进程内模型，不做 runner 子进程

CLI/daemon 在自身进程内加载模型，模型注册表负责懒加载、LRU 与 keep_alive。我们借鉴 ollaya 的分层（protocol/hub/runner/server/decision/cli），但不采用 per-model runner 子进程或 MCP，因为本项目的核心价值是可嵌入的进程内推理；未来若需要隔离，可在 `InferenceBackend` 下增加子进程实现。

**考虑过的替代方案**：完整 ollaya 形态（每模型 runner 子进程、MCP）；纯库、不带 daemon。前者破坏“进程内”卖点，后者不满足 CLI 服务需求。

**后果**：一个模型崩溃会影响整个进程；内存/显存回收依赖 LRU 与 keep_alive；隔离性弱于子进程方案。
