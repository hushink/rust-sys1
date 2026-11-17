# 跨请求 batching 放在服务层，模型层保持同步 predict

`sys1-server` 用 mpsc 把并发请求聚合，按模型分组后调用 `DecisionModel::predict_batch`；模型对象不感知 HTTP、连接或请求 ID。默认 `max_batch = 32`、`max_wait = 2 ms`。收集以 `yield_now` 自旋为主：单请求只付 500 µs grace，避免 Windows 定时器粒度给每个请求增加约 13 ms；有并发时才继续等到 2 ms 或凑满一批。进程内 API 不自动批处理，调用方需要时可直接使用 `predict_batch`。

模型可通过 `DecisionModel::supports_cross_request_batching` 退出。CPU 上的 Laya 返回 `false`：CPU 单次前向已经使用线程池，把不同的并发请求串行塞进一个 batch 会让 8 并发从 ~24 req/s 降到 ~13 req/s。GPU 上的 Laya 返回 `true`。Laya 的 batch 预处理还会对完全相同的请求去重，这只对重复负载有效。

**考虑过的替代方案**：在模型内部维护 async 队列（把 HTTP 生命周期泄漏进 core）；用 `sleep`/`timeout` 作为主要收集窗口（Windows 上单请求多约 13 ms）；每个请求独立 forward（GPU 利用率低，8 并发不同 state 仅 ~36 req/s）；固定大小批处理（空闲时延迟不可控）。

**后果**：CUDA 上不同 state 的 8 并发吞吐从 ~36 提升到 ~164 req/s，32 并发约 209 req/s（PowerShell 闭环）；Rust `loadgen` 在 F16 + `--batch-max 32` 下饱和约 345 req/s。单请求延迟基本不变；同一批只能包含同一模型；大 batch 会增加尾延迟和显存峰值，长上下文需要调小 `max_batch`；请求取消要等批次完成后才能释放；CPU 模型默认不经过 batcher。
