# local 层长序列使用精确分块窗口 attention

ModernBERT 的 local 层只允许 `|i-j| <= local_attention/2`（Laya 为 64），但 eager 路径仍计算完整 `S×S` 分数矩阵。CUDA 上 state ≥512 token 时，`sys1-laya` 按 128 个 query 分块：每个块只取 `[start-window, end+window]` 的连续 K/V，构造窗口加 padding mask，再做小块 matmul。global 层仍走 eager。CPU 保持 eager，避免 gather/分块的额外开销。`LayaOptions::local_window=false` / CLI `--no-local-window` 可关闭。

**考虑过的替代方案**：完整 eager（长序列浪费 7x local 计算）；按 query gather 全部 K/V 窗口（显存/带宽开销大，实测更慢）；FlashAttention（932 token 快 2.6x，但答案与 eager/F32 不一致，且 CUDA 13.4/Windows 构建崩溃）；修改训练/模型结构（改变权重语义）。

**后果**：932-token F16 从 ~194 ms 降到 ~141 ms（1.38x），F32 从 ~185 ms 降到 ~142 ms（1.30x）；F32 答案与 eager 完全一致，F16 因 GEMM 形状/舍入最大差 ~6e-4；4 并发长请求约 1.67x。窗口路径仅 CUDA、长度 ≥512；单次 block 元素数超过 512M 时回退 eager；global 层与极短序列保持 eager。
