# sys1-core

[English](README.md)

rust-sys1 的核心 trait、manifest、模型注册表、typed 进程内 API 与设备选择。

## 包含内容

- `DecisionModel`：同步 `predict(&State, &Questions) -> DecisionOutput`。
  可选的 `predict_long` 用重叠窗口扫描长 state；见 `LongContextOptions` 与
  `WindowAttribution`。
- `ModelFactory`：从 `ModelManifest` 懒加载模型。
- `ModelResolver`：可选的运行时引用解析，把外部引用变成可加载的 manifest。
- `ModelRegistry`：名称/别名解析、懒加载、LRU、`keep_alive`、卸载，以及运行时
  `register` / `unregister` 修改目录。
- `Questions` / `Answers`：typed builder 与类型化读取。
- `ModelManifest` / `ModelFile`：模型包契约。
- `DevicePreference` / `ResolvedDevice`：运行时设备选择辅助类型。

## 示例

```rust
use std::sync::Arc;
use sys1_core::{
    Answers, DecisionModel, DecisionOutput, ModelError, ModelRegistry, Questions,
    RegistryConfig,
};
use sys1_protocol::{Answer, NoulAnswer, State, Usage};

struct Echo;

impl DecisionModel for Echo {
    fn predict(&self, _state: &State, questions: &Questions) -> Result<DecisionOutput, ModelError> {
        let answers = questions
            .iter()
            .map(|(id, _)| (id.clone(), Answer::Noul(NoulAnswer::new(0.5))))
            .collect();
        Ok(DecisionOutput::new(Answers::new(answers), Usage::new(1, 0)))
    }
}

let registry = ModelRegistry::new(
    vec![manifest],          // Vec<ModelManifest>
    Arc::new(factory),       // Arc<dyn ModelFactory>
    RegistryConfig::default(),
)?;

let response = registry.predict("laya-latest", &state, &questions)?;
```

## 注册表语义

- 模型名通过 canonical name 与 alias 解析。
- 懒加载；每个模型一个 `OnceLock`，避免重复插入。
- `max_loaded` 会淘汰最老且没有外部 `Arc` 持有的模型。
- `keep_alive` 在 `maintenance()` / `predict()` 时清理空闲模型。
- 正在使用的模型可能临时超过 `max_loaded`。
- `register(manifest)` 以 `ModelOrigin::Runtime` 添加目录项；
  `register_configured` 标记配置来源。重名和别名冲突会被拒绝，目录保持不变。
  `unregister(name)` 按名称或别名移除目录项并丢弃常驻实例。
- `is_loaded(name)` 查询常驻状态，不刷新 LRU 时间。
- `origin(name)` 返回 `ModelOrigin::Configured`（传给 `new` 的 manifest）或
  `ModelOrigin::Runtime`（通过 `register` 添加）；别名会解析到规范条目。
- `predict_output` 返回 typed `DecisionOutput`；`predict` 再包装为带规范模型名的
  官方 `SystemOneResponse`。
- `DecisionOutput::actions` 携带可选的逐问题动作概率（例如 Laya 的
  `action.act_probability`）。它是 typed 扩展，官方 wire 响应不会包含。
- `DecisionOutput::windows` / `scored_windows` 携带长上下文归因：每个答案的
  决定 token 窗口，以及实际打分的窗口数（`predict` 为 `0`，单窗口为 `1`）。
  `predict_long_output` 对长调用复用同样的校验与注册表路径。
- `ModelResolver` 不属于注册表：调用方把引用解析成 manifest、注册它们，并提供
  能加载这些 manifest 的 factory。CLI 用它实现服务运行期间从 hub 添加模型。

## 测试

```bash
cargo test --offline -p sys1-core
```
