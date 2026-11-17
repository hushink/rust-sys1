# Qwen3 / Qwen3.5 解码器模型族

**状态**：已采纳（2026-11-18）

**背景**：在 NLI 与 GLiClass 之后，需要接入 Ollaya 目录中的 Qwen3Guard、Von、
Decima、Kev、Decision 1.0 与 Clef。前三个使用编码器，后三个共享 Qwen3.5
hybrid decoder；Qwen3Guard 使用 dense Qwen3。若每个 family 各自实现 decoder，
权重加载、LoRA、RoPE、Gated DeltaNet 与 KV cache 形状会重复且容易漂移。

**决定**：

1. 新增 `sys1-qwen`，实现 Qwen3 与 Qwen3.5 的 candle 推理骨干：
   - `WeightStore` 统一 safetensors 分片、PyTorch zip、LoRA 合并；
   - Qwen3.5 内部把任意长度右填充到 64 token 的整数倍，Gated DeltaNet 用
     64-token chunk 的 gated delta rule；
   - 支持 tied / untied LM head，供 Clef 读取 lexical 向量。
2. 每个 head 保持独立 crate：`sys1-qwen3guard`、`sys1-kev`、
   `sys1-decision`、`sys1-clef`；Von 与 Decima 使用共享 ModernBERT。
3. `sys1-families` 负责识别所有目录并统一注册；`local_manifest` 递归收集
   checkpoint 文件，支持 `pytorch/`、`backbone/`、`base/` 等嵌套布局。
4. Kev 的 base 不随 adapter 仓库发布。加载器依次接受显式
   `KevOptions::base_directory` / `[families.kev] base_dir`、同目录或 `base/`
   的本地 base，以及 `kev.json`、`head.pt`、`provenance.json`、
   `adapter_config.json` 中声明的仓库（含 revision）；后者由 CLI/engine 通过
   hub 回调自动拉取。温度同样优先使用显式配置，其次 `kev.json`，再次
   `head.pt` 内嵌的标定值。

**取舍**：

- 选择 candle 而非 ONNX：与现有 Laya/NLI/GLiClass 一致，权重直接从作者
  原始 safetensors / torch zip 读取，不依赖 ORT。
- Qwen3.5 的 delta rule 不使用自定义 kernel，按 chunk 用普通张量运算实现，
  优先正确性与 CPU 可用性；GPU 上后续可换 fused kernel。
- 9B Clef 默认 F32 在 CPU 上不可行；文档要求使用 BF16 + GPU，crate 本身
  不改变全局 dtype 策略。

**后果**：新增 7 个 crate；`sys1` 与 `sys1-cli` 的 accelerator features
向下传递。Qwen3 在 [151644] / [151644,872] 上与 transformers 对齐；
Qwen3.5 在单 chunk 与 130-token 多 chunk 行上与 transformers 对齐；
Qwen3Guard、Von、Decima、Kev、Decision 的 Python 参考输出均在 1e-3 内。
Clef 的 9B 权重未在开发机验证，保留 ignored 测试与真实 checkpoint 入口。
