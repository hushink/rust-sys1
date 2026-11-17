# sys1-core

[English](README.md)

rust-sys1 的核心 trait、manifest、模型注册表、typed 进程内 API 与设备选择。

## 包含内容

- `DecisionModel`：同步 `predict(&State, &Questions) -> DecisionOutput`。
- `ModelFactory`：从 `ModelManifest` 懒加载模型。
- `ModelRegistry`：名称/别名解析、懒加载、LRU、`keep_alive`、卸载。
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

## 测试

```bash
cargo test --offline -p sys1-core
```
