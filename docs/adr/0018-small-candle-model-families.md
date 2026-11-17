# 小型 candle 模型族：NLI 与 GLiClass

首批 Laya 之外的模型族选择 **NLI ModernBERT**（`nli-pairs-v1`）与 **GLiClass edge**（`gliclass-uni-v1`）。两者都基于 ModernBERT，参数量分别约 150M 与 33M，CPU 可运行；NLI 用 cross-encoder pair 分类，GLiClass 用 label-span 池化 + MLP 打分。实现使用 candle，共享 `sys1-modernbert` 编码器；`sys1-families` 按目录 config 自动识别并统一分发，CLI 与 `InferenceEngine` 不需要知道具体家族。

**考虑过的替代方案**：Decima small（122M，ONNX/torch late-interaction，需额外 BERT 与自定义 head）；Kev 0.8B/4B（decoder + LoRA + pointer head，模型与布局复杂、CPU 慢）；Clef/Winnow/Nimble（9B–27B，不符合“小且快”）；只做 Laya 的更多 checkpoint（不能验证多家族扩展）；直接绑定 Ollaya 的 ONNX 图（其 weightless 格式不是标准 ONNX，且用户要求 candle 基础）。

**取舍**：NLI base 的 150M 比 Decima small 大，但复用 ModernBERT 编码器，且上游有清晰的 cross-encoder 语义与可重复的 Python 参考；GLiClass edge 只有 33M，响应最快，但只实现 ModernBERT uni-encoder 变体，DeBERTa-large 与 v4 calibrator 暂不支持。两者都是英文、512/1024 token 预算。`sys1-modernbert` 从 `sys1-laya` 抽出，Laya 的 golden 保持不变；NLI/GLiClass 的模板、字段渲染与置信度由各自 crate 负责。

**后果**：新增 `sys1-modernbert`、`sys1-nli`、`sys1-gliclass`、`sys1-families` 四个 crate；`sys1` 默认 feature 增加 `families`。NLI/GLiClass 目录可从 `local:` 或 `hf:` 直接 serve/run，CLI 自动识别 family 并写入 manifest；`sys1 run` 与 HTTP 路径均已验证。NLI 与 GLiClass 的测试分别与上游 `transformers` / `gliclass==0.1.20` 前向在 1e-4 内一致。
