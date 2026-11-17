# Laya v1 采用 candle，ONNX/ORT 后置

Laya 的已发布权重是 safetensors，candle 可直接消费；官方 ONNX 产物尚未发布到 HF。v1 用 candle clean-room 重写 Laya（不依赖 candle-backend / candle-laya），只实现官方 TypeSafe 所需的核心面 + Router + batch；ONNX/ORT 作为未来可选 backend，用于长尾模型与 TensorRT EP。TensorRT 不通过 candle 实现。

**考虑过的替代方案**：ONNX-first（需要 Python 导出产物、引入 ORT 原生依赖）；自研 TensorRT FFI backend（工程量与版本绑定成本过高）；直接依赖 `laya-onnx`（无法满足独立重写要求）。

**后果**：需要自行实现 ModernBERT、decision head、tokenizer/序列布局、校准与 Router；数值对齐用官方 Laya golden vectors。TensorRT 与 ONNX 模型延后到 v1.1。

**调研补充（2026-11-17）**：candle 0.11 没有 TensorRT 支持，且其 `Device`/`BackendStorage` 是逐 op 后端，无法直接承载 TensorRT 的 graph engine。Rust TensorRT 绑定整体不成熟（多数为 runtime-only 或 experimental），因此 TensorRT 的现实路径只有 ORT 的 TensorRT Execution Provider：`ort 2.0.0-rc.13` 对应 ONNX Runtime 1.28，仅提供 CUDA 13 二进制，需要本机 TensorRT SDK。
