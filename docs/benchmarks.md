# 新模型族压测（2026-11-18）

[English](benchmarks.en.md)

本文记录 NLI、GLiClass、Von、Decima（base / small）、Qwen3Guard、Kev 与
Decision 在 CPU 和 CUDA 上的 HTTP 压测结果。测试通过 `sys1 serve` 的
`/v1/systemone` 接口完成，工具为 `uv run sys1-bench`。

## 环境

| 项目 | 值 |
|---|---|
| CPU | Intel64 Family 6 Model 151（12 逻辑核） |
| RAM | 31.8 GiB |
| GPU | NVIDIA GeForce RTX 3060 12 GiB |
| OS | Windows 11 Pro for Workstations |
| 构建 | `cargo build --release -p sys1-cli`；CUDA 为 `--features cuda` |
| dtype | 两种设备均为 F32（见下文“F16 与 dtype”） |
| 工具 | `python/tools` 的 `sys1-bench`，`--confidence typesafe` |

复现：

```bash
cargo build --release -p sys1-cli
cargo build --release -p sys1-cli --features cuda

uv run sys1-bench \
  --exe target/release/sys1.exe \
  --model-dir target/models/decima-small-pytorch \
  --model-name decima-small --device cpu --dtype f32 \
  --family generic --option-sweep --concurrency-sweep --json
```

CUDA 构建请将 `--exe` 指向 `target-cuda/release/sys1.exe`，`--device cuda`。
Qwen3Guard 需要 `--family qwen3guard`；其他模型使用 `--family generic`。
`--no-concurrent` 可跳过固定的 x4/x8 并发场景，适合 Kev/Decision 这类
Qwen3.5 混合模型。

## 单请求与并发

`POST noul` 只包含一个 noul 问题；`batch` 为 noul+score+choice 三问。
`x8` 为 8 个并发客户端、每个 5 次请求；RPS 为整个场景的墙钟吞吐。

| 模型 | 设备 | noul p50 (ms) | batch p50 (ms) | x8 p50 (ms) | x8 RPS |
|---|---:|---:|---:|---:|---:|
| NLI (ModernBERT) | CPU | 54.6 | 159.6 | 162.7 | 45.8 |
| NLI (ModernBERT) | CUDA | 18.6 | 23.0 | 225.2 | 33.7 |
| NLI (DeBERTa-v3-large) | CPU | 382.2 | 1532.2 | 1590.2 | 4.9 |
| NLI (DeBERTa-v3-large) | CUDA | 30.2 | 78.7 | 240.5 | 31.8 |
| GLiClass | CPU | 15.1 | 42.4 | 27.7 | 213.1 |
| GLiClass | CUDA | 18.7 | 12.4 | 116.8 | 62.1 |
| Von | CPU | 276.2 | 491.5 | 1684.4 | 3.9 |
| Von | CUDA | 32.6 | 39.3 | 263.3 | 28.9 |
| Decima-small | CPU | 53.1 | 179.4 | 114.5 | 62.7 |
| Decima-small | CUDA | 23.4 | 63.4 | 195.1 | 38.7 |
| Decima-base | CPU | 102.8 | 358.5 | 341.7 | 22.0 |
| Decima-base | CUDA | 36.6 | 122.0 | 444.0 | 17.5 |
| Qwen3Guard 0.6B | CPU | 1677.2 | 3164.8 | 7530.5 | 1.05 |
| Qwen3Guard 0.6B | CUDA | 95.3 | 179.1 | 659.8 | 11.7 |
| Kev 0.8B | CPU | 585.0 | 1627.6 | — | — |
| Kev 0.8B | CUDA | 54.2 | 148.6 | 573.4 | 13.4 |
| Kev 4B (BF16) | CUDA | 83.2 | 248.6 | 779.8 | 9.9 |
| Decision 0.8B | CPU | 1124.3 | 3287.9 | 4937.8 | 1.56 |
| Decision 0.8B | CUDA | 76.7 | 215.0 | 665.4 | 11.7 |

Kev/Decision 的 CUDA 数字是深度卷积优化后的结果（见下节）。优化前分别为
4379 ms / 4404 ms，x8 场景无法在合理时间内跑完。

## 选项数量扫描

每个选项数量 3 次请求（Decima-base 为 2 次）；`choice` 从 2 到 40 个选项，
`score` 从 2 到 10 个等级。表中为 p50（ms）。

| 模型 | 设备 | choice 2 | choice 5 | choice 10 | choice 40 | score 2 | score 5 | score 10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Decima-small | CPU | 79.9 | 68.9 | 91.9 | 247.9 | 44.6 | 53.2 | 91.9 |
| Decima-small | CUDA | 28.9 | 28.0 | 28.9 | 36.2 | 23.9 | 21.7 | 20.6 |
| Decima-base | CPU | 139.6 | 139.2 | 233.1 | 900.7 | 100.1 | 136.6 | 177.9 |
| Decima-base | CUDA | 37.6 | 36.8 | 38.7 | 94.7 | 34.6 | 35.7 | 34.7 |
| Von | CPU | 162.9 | 168.9 | 207.0 | 521.9 | 134.5 | 165.0 | 206.3 |
| Von | CUDA | 24.6 | 25.4 | 26.2 | 42.2 | 26.0 | 24.9 | 25.3 |

Decima 的选项行按 token 长度分组批处理：长度相同的选项共享一次编码与
scorer 前向，长度不同则保持独立，避免 padding 改变编码器的局部注意力。
40 个选项在 Decima-small 上仍是 36 ms（CUDA），说明该优化把额外选项的
编码次数从 40 次降到少数几组。

## F16 与 cuDNN

CUDA 构建分别使用 `--features cuda` 和 `--features cudnn`，dtype 均为 F16。
`noul` 为单请求 p50（ms），`x8` 为 8 并发 40 次请求的吞吐（RPS）。

| 模型 | CUDA noul | CUDA x8 | cuDNN noul | cuDNN x8 |
|---|---:|---:|---:|---:|
| NLI (ModernBERT) | 17.5 | 36.6 | 18.4 | 36.8 |
| NLI (DeBERTa-v3-large) | 22.8 | 28.3 | — | — |
| GLiClass | 11.2 | 72.3 | 16.4 | 72.0 |
| Von | 25.0 | 27.9 | 26.2 | 27.8 |
| Decima-small | 18.1 | 39.8 | 20.4 | 39.7 |
| Decima-base | 39.4 | 16.9 | 34.7 | 17.2 |
| Qwen3Guard | 77.6 | 16.6 | 77.1 | 16.6 |
| Kev 0.8B | 54.3 | 13.1 | 51.9 | 13.2 |
| Decision 0.8B | 63.6 | 11.7 | 61.2 | 11.7 |

cuDNN 对 Decima-base 和 Qwen3.5 解码器有约 3–8% 的单请求收益，但会让
GLiClass/Von/Decima-small 这类小编码器变慢（2%–45%）；x8 吞吐基本不变。
因此默认 CUDA 构建即可，只有大模型或特定 GEMM 形状值得启用 cuDNN。
构建 cuDNN 时需要把 `cudnn.lib` 所在目录加入 `LIB`（本机为
`D:\Applications\cudnn-windows\lib\x64`），运行时 DLL 目录需在 `PATH`。

## TF32

`--tf32`（或 `[server] tf32 = true`）只影响 CUDA 上的 F32 GEMM；F16/BF16
路径不变。RTX 3060、F32、单请求、无并发：

| 模型 | 场景 | F32 | F32 + TF32 |
|---|---|---:|---:|
| DeBERTa-v3 NLI | noul p50 | 33.4 ms | 32.6 ms |
| DeBERTa-v3 NLI | noul+score+choice p50 | 79.7 ms | 70.6 ms |
| Decima guard-small | noul p50 | 36.1 ms | 31.9 ms |
| Decima guard-small | noul+score+choice p50 | 105.5 ms | 120.3 ms |

大编码器的多问题请求有约 10% 收益；小编码器在运行间波动内。TF32 的尾数
精度低于 F32，对数值敏感的对比应继续用 F32/F16 或 BF16。

## 跨请求批处理

`sys1 serve` 默认把同一模型上的并发请求合并为一次前向。CPU 上的对比：

| 模型 | 模式 | noul p50 | x8 p50 | x8 RPS |
|---|---|---:|---:|---:|
| Von | 合并批处理 | 276.2 | 1684.4 | 3.9 |
| Von | `--no-batch` | 270.6 | 1131.4 | 6.9 |
| Decima-small | 合并批处理 | 53.1 | 114.5 | 62.7 |
| Decima-small | `--no-batch` | 44.6 | 92.6 | 77.8 |

对这些小编码器，CPU 上把 8 个请求合成一批并没有提高吞吐：代价是单次前向
的序列被拉长，且服务端必须串行执行批内请求；不合并时 8 个请求可以各自利用
多核并行。Qwen3Guard 这类解码器则不同，批处理与 CUDA 结合后吞吐提升明显
（CPU 1.05 RPS → CUDA 11.7 RPS）。

## 结论与后续

- **编码器**：单请求 CUDA 通常快 1.5–8 倍（GLiClass 例外：CPU 15.1 ms，
  CUDA 18.7 ms）；8 路并发时 NLI/GLiClass/Decima-small 的 CPU 吞吐反而更高，
  Von 与 Decima-base 仍是 CUDA 更好。部署时应按模型和并发度实测选择。
- **Qwen3Guard（dense Qwen3）**：GPU 收益最大，单请求 1.68 s → 95 ms。
- **Kev / Decision（Qwen3.5 hybrid）**：优化前 CUDA 单请求 4.4 s，比 CPU 慢
  3–5 倍。逐层剖析显示 18 个 DeltaNet 层的通用 `conv1d` 每次约 254 ms，
  占整个前向的 98%；把它换成 4-tap 的 depthwise 移位乘加后，每层降到
  0.36 ms。Kev CUDA 4.38 s → **54 ms**（约 80×），Decision 4.40 s →
  **77 ms**（约 57×），CPU 也分别从 785/1464 ms 降到 585/1124 ms。
  现在 Qwen3.5 家族在 CUDA 上比 CPU 快 10–15 倍。
- **F16**：CUDA F16 路径在 NLI/Von/GLiClass 上暴露过 dtype 与 contiguous
  问题，已修复（mask 转 dtype、encoder/scorer 输入 contiguous）。完整 F16
  矩阵见上节；编码器与 Qwen3Guard 均优于 F32，Kev/Decision 也比 F32 再快
  约 20%。cuDNN 没有稳定收益。
- **Clef 9B**：权重约 18.8 GB，超出 12 GiB 显存；CPU F32 需要约 36 GB 内存。
  使用 BF16 CPU 的验证结果见“Clef 9B 验证”一节。

## DeBERTa-v3 NLI 验证

`MoritzLaurer/deberta-v3-large-zeroshot-v2.0@cf44676c`（870 MB F16）的
`model.safetensors` SHA256 与 Hugging Face 一致。新的 disentangled attention
编码器在 choice/score/noul 三类问题上与 transformers 的 F32 结果在 1e-4 内
一致（tone 0.859564/0.140436，urgency 1.539309，noul 0.998042）。
CPU 单请求 382 ms；CUDA F16 23 ms，8 并发 28 RPS。

## Decima v2（guard-small）验证

`amyrmahdy/decima-guard-small@ac110248` 是 decima v2 布局：mmBERT-small
（384 维、22 层、6 头）、`choice_fit: true`、`question_in_state: true`、
`max_state_tokens: 2048`、`max_choice_tokens: 64`、温度 0.754586。
encoder 562 MB + head 18 MB，全部下载完成后可直接加载。

与作者代码 `decima@1b07634e`（v2.3.0，经 Ollaya 的独立参考封装）的 F32
结果对比：

| 问题 | 上游 log-probs | Rust |
|---|---|---|
| team choice | [−0.003964, −5.532571] | [−0.003963, −5.532570] |
| urgent noul | 0.414360（P(yes)） | 0.414360 |
| anger score | [0.429440, 0.357271, 0.213289] | [0.429440, 0.357271, 0.213289] |
| 长问题 choice（触发 `choice_fit`） | billing 0.920987 | 0.920987 |

长问题超过 64 token 时，两边都生成
`<问题词前缀> … <完整选项>` 的对比行（选项保持完整，问题按词边界二分缩短），
说明 v2 的 option-fit 渲染与上游 `render.choice_texts` 一致。

```bash
$env:SYS1_DECIMA_GUARD_SMALL_TEST_DIR="C:/path/to/decima-guard-small"
cargo test -p sys1-decima --test decima predicts_with_the_guard_small_checkpoint -- --ignored
```

## Kev 4B 验证

`jaredpalmer/kev-4b`（adapter 124 MB）与 `Qwen/Qwen3.5-4B-Base`（两个分片，
9.3 GB）的 SHA256 均与 Hugging Face 一致。CPU F32 与独立 PyTorch
LoRA 合并 + pointer head 参考在 1e-6 内一致（billing 0.900975，refund
0.853919，urgency 0.953128）；12 GB CUDA 卡上 BF16 的误差在 0.003 内，
单请求 83 ms。4B 的 BF16 权重加 LoRA 合并后可以放入 12 GB 显存。

## Kev 9B 验证

`jaredpalmer/kev-9b@2629c06a`（adapter 173 MB、`head.pt` 8.4 MB）与
`Qwen/Qwen3.5-9B-Base@68c46c4b`（4 个分片共约 19 GB）的 SHA256 均与
Hugging Face 一致。`head.pt` 记录温度 **2.297397**、base revision
`68c46c4b…`；adapter 有 248 个 LoRA 对，merge scale 2.0。

12 GiB 显存放不下 BF16 主干，因此与 Clef 一样使用 CPU + BF16。独立
PyTorch 参考（transformers 的 BF16 主干、手动 LoRA 合并、pointer head）
与 Rust 的结果：

| 问题 | PyTorch BF16 | Rust BF16 |
|---|---|---|
| team | billing 0.976657 / tech 0.023343 | 0.978158 / 0.021842 |
| refund | 0.920695 | 0.924340 |
| urgency | [0.272992, 0.349913, 0.377095] | [0.267890, 0.349262, 0.382848] |
| urgency score | 1.104103 | 1.114958 |

概率差 ≤0.006，决策全部一致；差异来自 BF16 累加顺序。`predict` 跑 3 个
问题约 **47 s**，加载后峰值内存约 19–20 GB。

```bash
$env:SYS1_KEV9B_TEST_DIR="C:/path/to/kev-9b"   # contains base/
cargo test --release -p sys1-kev --test kev predicts_with_the_9b_checkpoint \
  -- --ignored --nocapture --test-threads=1
```

## Clef 9B 验证

`Cloudflare/clef-flash` 的四个 BF16 shard 共约 18.8 GB，下载后与 Hugging Face
LFS SHA256 全部一致。12 GiB 显存放不下主干，因此使用 CPU + BF16 验证：

```bash
$env:SYS1_CLEF_TEST_DIR="C:/Users/hushink/AppData/Local/Temp/sys1-clef"
$env:SYS1_CLEF_DTYPE="bf16"
cargo test --release -p sys1-clef --test clef matches_the_recorded_triage_decisions \
  -- --ignored --nocapture --test-threads=1
```

结果：

- 加载约 20 s，峰值工作集约 18–19 GB。
- 单次 triage 请求（5 个问题）`predict` 耗时 **52.1 s**（CPU BF16，复测）。
- 通过 `sys1 serve` 的 HTTP `/v1/systemone`：单 noul 请求 18.3 s，
  noul+score+choice 三问 26.0 s（状态较短；证明服务和批处理路径可用）。
- 决策与 Ollaya 记录的 `clef-flash-fp16` integration corpus 一致：
  `team=billing`、`refund/urgent/churn=true`、`upset` 的 argmax 为第 4 档
  （索引 3）。
- 这次验证还暴露并修复了两个真实 checkpoint 才会触发的问题：head 中
  4096 维向量的 Linear 需要先升到二维，`residual_gate` 需要以 0 维张量
  做 sigmoid。
- **logits 级对齐**：以作者仓库中的 `joint_schema_model.py`（BF16，CPU）
  为参考，Rust 的输入 token 行与上游 `encode_record` 完全一致（team 单问
  为 179 个 token）。概率最大偏差 ≤0.033（BF16 在 candle 与 PyTorch 间的
  累加顺序差异），argmax 全部一致：team billing 0.9565 → 0.9874，triage
  refund 0.7873 → 0.7994，upset 第 4 档 0.7144 → 0.6867。复现：

  ```bash
  cargo test --release -p sys1-clef --test clef matches_upstream_bf16_probabilities \
    -- --ignored --nocapture --test-threads=1
  ```

GPU 上需要至少 24 GB 显存（BF16 权重约 18 GB，4096-token 请求峰值约
22.8 GB）；RTX 3060 12 GB 不适用。
