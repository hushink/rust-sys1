# 注册中心管运行时选择，Cargo features 管编译期可用性

模型注册中心从 manifest 选择已经编译进二进制的 family/backend；Cargo features 决定哪些实现进入二进制。内置组件默认全编译，features 只用于裁剪与加速器；ONNX 模型在 ORT backend 编译后可仅靠 manifest 添加，candle 原生 family 仍需编译期代码。不做动态插件 ABI。

**考虑过的替代方案**：细粒度 `model-*`/`hub-*` features 让用户按需组合（增加理解成本）；cdylib/WASM 动态插件（ABI 稳定性成本过高）；只支持数据驱动 ONNX 模型（放弃 candle 原生 family）。

**后果**：普通用户不需要理解 feature 组合；服务端遇到未编译的 family/backend 时返回明确的 unsupported 错误；新增 candle family 需要重新编译。
