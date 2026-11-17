# Laya v1 采用 candle，ONNX/ORT 作为可选后端

Laya 的已发布权重是 safetensors，candle 可直接消费；官方 ONNX 产物尚未发布到 HF。v1 用 candle clean-room 重写 Laya（不依赖 candle-backend / candle-laya），只实现官方 TypeSafe 所需的核心面 + Router + batch。ONNX/ORT 保留为可选 backend，用于长尾模型与 TensorRT EP；它不改变 candle 作为参考实现的地位。TensorRT 不通过 candle 实现。

**考虑过的替代方案**：ONNX-first（需要 Python 导出产物、引入 ORT 原生依赖）；自研 TensorRT FFI backend（工程量与版本绑定成本过高）；直接依赖 `laya-onnx`（无法满足独立重写要求）。

**后果**：需要自行实现 ModernBERT、decision head、tokenizer/序列布局、校准与 Router；数值对齐用官方 Laya golden vectors。TensorRT 与 ONNX 模型延后到 v1.1。

**调研补充（2026-11-17）**：candle 0.11 没有 TensorRT 支持，且其 `Device`/`BackendStorage` 是逐 op 后端，无法直接承载 TensorRT 的 graph engine。Rust TensorRT 绑定整体不成熟（多数为 runtime-only 或 experimental），因此 TensorRT 的现实路径只有 ORT 的 TensorRT Execution Provider：`ort 2.0.0-rc.13` 对应 ONNX Runtime 1.28，仅提供 CUDA 13 二进制，需要本机 TensorRT SDK。

**实现补充（2026-11-17）**：采用 `ort 2.0.0-rc.15` + `default-features = false, features = ["std", "load-dynamic", "api-17"]`，避免构建期下载二进制，也不链接 ORT。实测以 `load-dynamic` 加载 Windows 系统自带 ONNX Runtime 1.17 成功，说明 API 17 绑定兼容 1.17 运行时；预编译 CUDA 二进制并非唯一选择。rc.13 在 Windows 进程退出时偶发 `ReleaseEnv` 崩溃，rc.15 改用显式 `Environment` 生命周期后该问题消失；`sys1-ort` 在进程内保留一个 Environment 引用，避免静态析构顺序问题。新增 `sys1-ort` 负责 session 与 token-scoring 契约，`sys1-laya/onnx` 复用 tokenization、序列布局、校准与答案格式化，只替换前向图。真实 Laya ONNX 导出已补齐：`scripts/export_laya_onnx.py` 用 NumPy + safetensors + ONNX 直接构建与 candle 前向等价的计算图，不依赖 PyTorch。三个 checkpoint 在官方 golden logits 上的最大误差为 1.7e-6–2.5e-6，Rust `onnx_parity` 测试进一步比较 candle 与 ONNX 的完整答案。`scripts/check_laya_onnx.py` 提供可重复的 logits 校验。合成图测试仍保留，用于 batching、padding、marker 索引、factory 与 HTTP 链路。多 GB 导出建议使用较新的 ONNX Runtime（Windows 1.17 加载 1.2 GiB 图不稳定，1.32 稳定）。TensorRT EP 的实机 CI 仍待补。
