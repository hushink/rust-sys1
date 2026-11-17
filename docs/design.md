# rust-sys1 设计摘要（v1）

> 状态：v1 已实现；使用方式见 [README.md](../README.md)。
> 术语以 [GLOSSARY.md](../GLOSSARY.md) 为准；关键取舍见 [docs/adr](./adr/)。

## 1. 目标与非目标

**目标**

- 一个可嵌入其他 Rust 项目的**进程内推理接口**，类型安全、与官方 SDK 的数据模型对齐。
- 一个 `sys1` CLI，运行严格兼容 TypeSafe 官方契约的 HTTP API（`/v1/systemone`、`/v1/models`）。
- 可替换的 System One 模型族；当前内置 Laya、NLI、GLiClass、Qwen3Guard、
  Von、Decima、Kev、Decision 与 Clef，后续可加其他 family/backend。
- 多来源模型获取：本地路径、Hugging Face、ModelScope、自定义静态 HTTP 中心。
- 单进程加载模型，带模型注册表、懒加载、LRU 与 keep_alive。
- 编译期 feature 控制加速器与可选组件；运行时自动探测可用设备。

**非目标（v1）**

- 不做 Jev 远程代理（Jev 只作为兼容目标）。
- 不做 per-model runner 子进程、MCP、动态插件 ABI。
- 不发布 crates.io；先内部库 + 二进制。
- 不做 parallel option layout（v1.1）；act head、long-context 扫描与 abstention 已实现。
- v1 不强制 ONNX/TensorRT；ONNX 作为可选 `sys1-ort` + `sys1-laya/onnx` 路径实现，TensorRT 通过 ORT TensorRT EP。
- 与 `candle-backend` / `candle-laya` / `jev-sdk-rs` 无代码或依赖关联。

## 2. 架构

```text
sys1-protocol   官方 wire types、校验、typed questions/answers、错误体
      ▲
sys1-core       DecisionModel trait、模型注册表、错误、设备/后端选择、manifest 类型
      ▲
sys1-hub        ModelSource / ModelCache：Local、HF、ModelScope、HTTP
      ▲
sys1-ort        动态加载的 ONNX Runtime session 与 token-scoring 契约
      ▲
sys1-modernbert 共享 candle ModernBERT 编码器
      ▲
sys1-laya       Laya family：clean-room candle 实现 + 可选 ONNX 导出路径
sys1-nli         NLI family：ModernBERT zero-shot 交叉编码器
sys1-gliclass    GLiClass family：ModernBERT uni-encoder（32M edge）
sys1-qwen        Qwen3 / Qwen3.5 decoder 骨干（candle）
sys1-qwen3guard  Qwen3Guard Gen 固定预设安全模型
sys1-von         Von 1.1 option-marker 编码器
sys1-decima      Decima late-interaction 编码器
sys1-kev         Kev pointer-head 解码器
sys1-decision    Decision 1.0 endpoint-head 解码器
sys1-clef        Clef joint-schema 解码器
      ▲
sys1-families   目录识别 + 统一 ModelFactory 分发
      ▲
sys1-server     axum：严格官方 /v1/*、/health、运行时 /api/*、跨请求 batching、错误映射
      ▲
sys1-cli        clap 二进制：serve/pull/list/show/rm/run/doctor
      ▲
sys1            umbrella：InferenceEngine + 进程内 API，默认内置全部 candle 模型族 + 全部 hub 源
```

依赖方向单向向上；`sys1-core` 不依赖 `sys1-hub`，`sys1-server` 不依赖具体模型族。CLI 负责把 hub、laya、server 组装起来。

## 3. 进程内推理接口

高层入口是 umbrella crate 的 `InferenceEngine`，把 hub、manifest、后端 factory、
注册表与协议类型串成一个对象：

```rust
use std::path::Path;
use sys1::{EngineOptions, InferenceEngine, Questions, State};

let engine = InferenceEngine::from_directory(
    Path::new("/path/to/laya-multilingual"),
    EngineOptions::default(),
)?;
let questions = Questions::new()
    .choice("tone", [("angry", "..."), ("calm", "...")])
    .score("urgency", ["Can wait", "This week", "Today"])
    .noul("billing", "Is this about billing?");

let output = engine.predict(
    "laya-multilingual",
    &State::text("..."),
    &questions,
)?;
let tone = output.answers.choice("tone")?; // 类型化读取
```

- `EngineOptions` 分组：`hub`（缓存/端点/凭据）、`registry`（LRU/keep_alive）、
  `model`（`ModelOptions`：device/dtype/max_len/confidence/attention）。
- `new` 空目录；`from_reference` / `from_directory` 立即注册；`add` / `add_as`
  通过 hub 物化并注册；Laya 仓库根自动注册 router 与三个变体，其余 candle
  模型族（NLI/GLiClass/Qwen3Guard/Von/Decima/Kev/Decision/Clef）按 checkpoint
  布局自动识别；`remove` 卸载并移除。
- `predict` 返回 typed `DecisionOutput`；`systemone` 返回官方 `SystemOneResponse`；
  `predict_batch` 调用模型原生 batch；`load` / `unload` 控制常驻。
- Engine 实现 `ModelResolver`，可直接作为 `sys1-server` 的运行时 resolver。
- `Questions` 是保持插入顺序的 map；`Question` 是按 type 判别的 enum；`Answers` 同名 map，提供 `choice/score/noul` 类型化读取。
- 核心 `predict` 同步执行，不绑定 tokio；提供可选 async wrapper。
- `DecisionModel` trait 位于答案层，不抽象原始 logits；`predict_output` 提供 typed 结果。
- 动态注册表使用对象安全 facade（`Arc<dyn DecisionModel>`）；库用户可直接用具体类型。
- 命名边界：Laya 前缀类型/函数只定义在 `sys1-laya`。其他 crate 使用通用名
  （如 `HubModelResolver`、`make_model_options`、`CheckpointFactory` 导入别名），
  checkpoint 布局辅助（`local_checkpoint_manifest`、`register_checkpoint_bundle`）
  也放在 `sys1-laya`；umbrella 只再导出 `sys1-laya` 的既有类型以保持原有调用方式，
  不新定义 Laya 前缀 API。

## 4. HTTP API

- 只实现官方端点：`POST /v1/systemone`、`GET /v1/models`。不实现 `/v1/decisions`、`/v1/decide`。
- 完整接受官方 schema，包括 `assets`、`box` 语法；Laya 不支持时返回 422 `detail[]`（`type: "unsupported"`），不静默忽略。
- 错误体严格使用官方 `detail` 形状：
  - 400：`{"detail": string | object}`
  - 422：`{"detail": [{"loc": [string|int], "msg": string, "type": string, "input"?: ..., "ctx"?: ...}]}`
  - 其他：401、413、415、429、529。
- 传输规则：32 MiB、UTF-8 无 BOM、嵌套 ≤64、有限数字、gzip/zstd 单层。
- 响应头包含 `x-typesafe-request-id` 与 W3C `traceparent`；`/health` 位于 `/v1` 之外。
- CORS 默认 permissive；`ServerConfig::cors_origins` 可设 allowlist，空列表关闭。允许 GET/POST/DELETE/OPTIONS，请求头含 content-type、authorization、danger-zone 与 traceparent。
- 优雅关闭：`serve_*_and_shutdown` 在 shutdown future 完成时停止接收并等待在途请求；CLI 接入 Ctrl-C/SIGTERM。
- Bearer 可选：配置了 key 就强制校验；未配置则接受任意/无 key；非 loopback 绑定且未配置 key 时告警。
- 未知模型返回 400 `{detail: ...}`。
- `/v1/*` 不输出 Laya 私有字段；扩展字段放到 `/api/*` 或响应头（后续）。
- 运行时模型管理：`/api/models` 提供目录/常驻状态、`load`/`unload`/`reload`、按引用添加与删除；`GET /api/models` 标记 `configured`/`runtime`，`POST /api/models` 可用 `"load": true` 添加并加载（失败回滚）。与 `/v1/*` 共用 Bearer，库内默认关闭（`ServerConfig::enable_admin`），`sys1 serve` 默认开启并可用 `--no-admin` 关闭。运行时增删不持久化，重启后以配置 manifest 为准。细节见 ADR 0008。
- 磁盘缓存管理：`GET /api/cache` / `DELETE /api/cache/{key}` 在服务进程内列出/删除模型缓存；`sys1-server` 通过 `CacheManager` trait 注入（CLI 用 hub 实现），不依赖 `sys1-hub`；常驻模型返回 409，需先 unload。
- 配置热加载：`POST /api/config/reload` 通过 `ConfigReloader` 注入；CLI 只应用 `[models]` 新增/移除，新增模型标记 `configured`。`[hub]`/`[server]` 或已有模型引用变化返回 409，解析/解析模型失败返回 400，均不修改目录。细节见 ADR 0011。
- 可观测性：`/metrics`（Prometheus 文本）与 `/api/metrics`（JSON）记录 systemone 请求/错误/延迟、batch 大小与耗时、模型加载/卸载/失败、按模型请求数、常驻模型数与 uptime；`ServerConfig::enable_metrics` 库内默认关闭，CLI 默认开启。端点与 `/v1/*` 共用 Bearer。
- 预加载：`[server] preload` / `--preload` 在监听端口前调用注册表 `acquire`，避免首请求冷加载；失败带模型名终止启动。
- 可替换日志/telemetry：`sys1-server` 不安装全局 subscriber，只发 `tracing` span/事件；通过 `ServerExtensions.telemetry` 注入 `Telemetry`（`RequestEvent`/`ModelEvent`/`BatchEvent`/`CacheEvent`），默认 `NoopTelemetry`，CLI 使用 `TracingTelemetry`。CLI 支持 text/JSON、文件追加与级别过滤（CLI/`SYS1_LOG_*` > `RUST_LOG` > `[server] log_*`）。
- Trace context：解析并返回 W3C `traceparent`（保留 trace id、生成 server span id），与 `x-typesafe-request-id` 一起进入日志、telemetry 事件与 `systemone` span。
- 超长 state 的 v1 行为待确认：建议返回 400，detail 含 `max_tokens_exceeded`（不做静默截断）。
- 跨请求 batching：并发请求经 mpsc 聚合，按模型分组后调用 `predict_batch`；默认 `max_batch=32`、`max_wait=2 ms`，GPU 单请求只付 500 µs yield grace。模型可用 `supports_cross_request_batching` 退出；CPU 上的 Laya 退出，因为独立 CPU 前向已经使用线程池。进程内接口不自动批处理，调用方可直接使用 `predict_batch`。细节见 ADR 0006。
- GPU 精度档位：默认 F32；`--dtype f16` 在 Ampere+ 上提高高并发吞吐，抽查概率差异 ~5e-4；`--tf32` 为 F32 GEMM 启用 TF32，收益在波动范围内。
- 压测：`crates/sys1-cli/examples/loadgen.rs` 提供闭环与开环（固定 offered rate、in-flight permit）两种模式；Python 工具 `sys1-bench` 不计入库 API。
- 长上下文：CUDA 上 local 层在 ≥512 token 时使用精确分块窗口 attention（128 query/块、±local window K/V、padding key mask）；932 token F16 从 ~194 降到 ~141 ms（1.38x），F32 从 ~185 降到 ~142 ms（1.30x），F32 答案差 0、F16 最大 ~6e-4。CPU 继续 eager。FlashAttention 原型在 932 token 下 ~69 ms，但输出与 eager/F32 不一致，未启用；上游完整 kernel 在 CUDA 13.4/Windows 上 cicc 崩溃。CPU int8/Q8_0 探测收益不足，未集成；MKL crate 不在离线缓存。

## 5. 模型、manifest 与 hub

最小 manifest：

```json
{
  "schema_version": 1,
  "name": "laya-multilingual",
  "description": "Laya multilingual decision model",
  "release_date": "2026-01-01",
  "family": "laya",
  "variant": "multilingual",
  "backend": "candle-safetensors",
  "format": "safetensors",
  "version": "1.0.0",
  "files": [
    { "path": "model.safetensors", "url": "...", "sha256": "...", "size": 123 },
    { "path": "encoder/config.json", "url": "...", "git_sha1": "...", "size": 123 }
  ]
}
```

- `sha256` 可选；HF 非 LFS 文件用 `git_sha1` 兜底，下载后本地补算 sha256。
- 模型源：`local`、`hf`、`modelscope`、`http`。自定义中心只需提供 manifest 与文件。
- 缓存：按模型落盘，`.partial` 断点续传、并发锁、离线模式、内容校验。
- 端点可配置：`HF_ENDPOINT`（含 hf-mirror）、`MODELSCOPE_ENDPOINT`；token 用 `HF_TOKEN` / `MODELSCOPE_API_TOKEN`。
- ModelScope 使用 `/api/v1/models/{repo}/repo?Revision=&FilePath=`；文件列表 `Recursive=true`，注意 3000 条静默截断。
- HF 使用 `resolve/{revision}/{path}`，revision 整体 `%2F` 编码；列表 cursor 分页。
- 不自研前不复用非官方 ModelScope Rust crate；hub 用薄 HTTP 适配层。

## 6. Laya v1 范围

- clean-room 重写，仅用公开 crate（candle-core/nn、tokenizers 0.23 等）。
- 支持 english / multilingual / typed-decisions 三个 checkpoint；`laya-latest` 由 Router 自动选择，另提供显式别名。
- 支持 choice / score / noul、batch、confidence、语言温度、Router。
- 实现状态：三个 checkpoint、batch、TypeSafe/Laya confidence、官方 parity Router 已落地；routing golden 159/159 对齐。
- 暂缓：parallel option layout。
- 数值对齐：官方 Laya golden vectors（`laya-rust/tests/golden`）作为 oracle；标签完全一致，probabilities/score 容差 1e-4。
- 渲染细节：temperature 按 `qtype:bucket` 选择并 clamp 到 `[0.5, 5.0]`；softmax 在 f32；`round4` 复现 CPython half-even。
- `confidence`：`/v1/*` 默认使用 TypeSafe/ollaya 归一化公式 `(K·p_max-1)/(K-1)`；配置可切换 Laya 原生熵置信度；Laya 原生值后续通过扩展端点/头暴露。

## 7. Features 与设备

内置组件默认全编译，features 只用于裁剪与加速：

| feature | 作用 | 默认 |
|---|---|---|
| `laya` | 内置 Laya family | 是（umbrella） |
| `families` | 全部 candle 模型族统一识别与分发 | 是（umbrella） |
| `hub-hf` / `hub-modelscope` / `hub-http` | 模型源适配器 | 是 |
| `server` / `cli` | HTTP 服务 / 二进制 | 否（由 CLI 组合） |
| `cuda` | candle CUDA | 否 |
| `cudnn` | candle cuDNN（隐含 cuda） | 否 |
| `metal` | candle Metal | 否 |
| `accelerate` | CPU BLAS 加速 | 否 |
| `mkl` | CPU BLAS 加速（预留；当前离线依赖不可用） | 否 |
| `nvml` | CUDA 显卡空闲显存探测 | 否 |
| `onnx` | ONNX Runtime backend（`sys1-ort` + `LayaOnnxModel`） | CLI 是；umbrella 否 |
| `ort-cuda` / `ort-tensorrt` / `ort-directml` / `ort-coreml` | ORT 执行提供器（隐含 `onnx`） | 否 |

- `cuda` 与 `metal` 互斥，用 `compile_error!` 强制。
- 运行时：`--device auto|cpu|cuda[:N]|metal` + `SYS1_DEVICE`。
- `auto` 只在已编译后端中探测；CUDA 默认选空闲显存最多的卡（NVML），失败回退 CPU 并告警；显式指定不可用则硬失败。
- `sys1 doctor` 输出编译 feature、驱动/设备、cuDNN、未来 TensorRT/ORT 状态；`serve` 启动日志与 `/health` 带上 backend/device。

## 8. CLI 与配置

- 命令：`serve`、`pull`、`list`、`show`、`rm`、`run`、`doctor`。
- 配置优先级：CLI > `SYS1_*` 环境变量 > `sys1.toml` > 默认。
- 默认绑定 `127.0.0.1:8080`；缓存目录用平台默认，`SYS1_CACHE` 覆盖。
- 缓存配额：`[hub] max_cache_bytes` / `--max-cache-bytes` / `SYS1_MAX_CACHE_BYTES`；`None`/`0` 不限制。pull 前按 LRU 淘汰非驻留条目，常驻模型与进行中的 pull 受保护；`/api/cache` 与 `sys1 list` 暴露 limit/used/free/reclaimable。详见 ADR 0012。
- `--api-key` / `SYS1_API_KEY` 可选；日志用 `tracing` + `RUST_LOG`。
- `/api/models` 管理 API 默认开启；`--no-admin` 或 `[server] admin = false` 关闭；未设置 API key 时启动日志告警。
- `/metrics` 与 `/api/metrics` 默认开启；`--no-metrics` 或 `[server] metrics = false` 关闭；`--metrics` 可强制开启。
- `[server] preload` / `--preload NAME` 在 serve 前加载模型，失败则启动失败。
- admin 开启时，CLI 额外提供 `/api/cache`（列出 key/大小/文件）与 `DELETE /api/cache/{key}`，与 `sys1 list`/`rm` 对齐。
- 日志：`--log-level`、`--log-format text|json`、`--log-file`，或 `SYS1_LOG_*` / `[server] log_*`；库不安装 subscriber，CLI 负责初始化。
- 热加载：`POST /api/config/reload` 重读 TOML；仅 `[models]` 增删可应用，其他变更提示重启。
- 运维收尾：`--cors-origin` / `--no-cors` / `[server] cors_origins`；Ctrl-C/SIGTERM 优雅关闭。
- 二进制名暂定 `sys1`。

## 9. 验收

- `sys1-protocol` 单测 + golden fixtures：请求/响应/错误形状。
- Laya golden vectors：tokenizer、序列布局、校准、Router、batch；标签完全一致，概率容差 1e-4。
- 官方 Python SDK `typesafe-sdk==0.7.5` 指向本地 server 的集成测试。
- `jev-sdk-rs` 作为 Rust 客户端回归。
- 不依赖线上 `api.typesafe.ai`（本地区 451）。
- CI：默认 CPU 三平台；Linux `--features cuda`、macOS `--features metal` 做编译检查；GPU 测试 self-hosted 可选。

## 10. 路线图

- **v1**：candle Laya、官方 API、hub、CLI、进程内接口、CUDA/Metal feature、doctor。
- **v1.1**：parallel layout；ORT backend；ONNX 模型（含 act head）；TensorRT EP。ORT 路径使用 `ort 2.0.0-rc.15` + `load-dynamic`，构建期不下载运行时；API 17 已在 ONNX Runtime 1.17 上验证。TensorRT 需要本机 TensorRT SDK；candle 不能直接承载 TensorRT。本地 ONNX 导出器已实现并通过 golden 对齐；上游官方 ONNX 产物仍待补。
- **后续**：Kev 等 decoder family；动态插件评估（仅在设计稳定后）。

### 实现状态（2026-11-17）

- 已完成：`sys1-protocol`、`sys1-core`、`sys1-hub`、`sys1-laya`、`sys1-server`、`sys1-cli`、`sys1`。
- 已用本地缓存权重验证：`sys1 run`、`sys1 serve` 的 `/v1/models`、`/v1/systemone`、`/health`、gzip/zstd 请求；`laya-latest` 选中 English 并成功加载。
- 已用本地缓存权重验证运行时管理：`/api/models` 列表、load/unload/reload、按 `local:` 引用动态添加、推理、删除；CPU F32 multilingual 冷加载约 12 s。
- 已用本地回环 HTTP 模型中心验证：manifest 解析、相对文件 URL、Bearer token、Range 断点续传，以及 `[models]` 名称映射。
- 已用本地假端点验证 HF（分页、子目录、LFS sha256、release date、token、URL 编码）与 ModelScope（Root/Recursive、token+cookie、3000 条截断拆分）。
- 已用本地缓存权重验证 `InferenceEngine` 进程内链路：`from_directory` → `add` → `predict` → `unload`；示例输出 noul 0.9678、score 1.7664，ignored 集成测试 19.4 s 通过。
- 命名边界已整理：非 `sys1-laya` crate 不再定义 `Laya*` 标识符；checkpoint 布局辅助移入 `sys1-laya`。
- 已实现 `/metrics` 与 `/api/metrics`（systemone、batching、模型生命周期、按模型请求数）；`preload` 启动预加载在 `serve` 绑定端口前执行，未知模型即失败。
- 管理 API 支持 `POST /api/models` 的 `"load": true`（加载失败回滚注册），`GET /api/models` 标记 `configured` / `runtime`。
- 服务进程内磁盘缓存管理已实现：`GET /api/cache` 与 `DELETE /api/cache/{key}`；`Hub::remove_cached` 校验 cache key 防止路径穿越，常驻模型拒绝删除。
- Engine 开销基准（CPU F32 multilingual，30 次交替）：直接推理 121.7 ms 均值、engine 122.6 ms，约 0.9 ms（<1%）；无需进一步优化 manifest 克隆。
- 可替换 telemetry 与 trace context 已实现：`Telemetry` 事件、`TracingTelemetry`、`ServerExtensions` 注入、`traceparent` 响应头；CLI text/JSON/文件日志。测试覆盖自定义 telemetry、trace id 透传、batch 事件与 JSON 日志文件。
- 配置热加载已实现：CLI 只应用 `[models]` 增删，新增项标记 `configured`；server 设置/引用变化 409，解析失败 400 且不改目录。真实进程验证 add 92 ms、remove、restart-required 409。
- 可选 ORT backend 已实现：`sys1-ort` 负责动态 session 与 token-scoring 契约；`sys1-laya/onnx` 共享 tokenization/校准/答案格式化，只替换前向图；`onnx.json` 可覆盖 I/O 名、线程与 `max_batch`。已用合成 ONNX 图验证 batching、padding、marker 索引、factory 与 HTTP 全链路；`ort 2.0.0-rc.15` 在 Windows 上以 `load-dynamic` 成功加载系统 ONNX Runtime 1.17；rc.15 的显式 Environment 生命周期避免了 rc.13 在进程退出时的偶发崩溃。
- 纯 NumPy/ONNX 导出器已实现（`python/tools` 的 `sys1-export-onnx`，不依赖 PyTorch）：从 safetensors 生成符合契约的 encoder/type embedding/decision head/scorer 计算图，并自动复制 tokenizer、`rl_agent_config.json` 与 `onnx.json`。三个 checkpoint 的导出 logits 与官方 golden 最大误差 1.7e-6–2.5e-6；`sys1-check-onnx` 可重复校验。Rust `onnx_parity` 测试比较 candle 与 ONNX 的完整答案。多 GB 导出建议 ONNX Runtime ≥1.20；Windows 1.17 加载 1.2 GiB 图不稳定，1.32 稳定。TensorRT EP CI 与上游官方 ONNX 产物仍待补。
- 小型模型族已实现（candle）：`sys1-nli` 在 ModernBERT zero-shot 交叉编码器上实现 `nli-pairs-v1`（base/large），`sys1-gliclass` 在 ModernBERT edge 上实现 `gliclass-uni-v1`；二者共享 `sys1-modernbert`，并接入 `sys1-families`、CLI 与 `InferenceEngine`。概率与上游 Python 参考在 1e-4 内一致。
- 缓存配额与 LRU 淘汰已实现：`HubConfig::max_cache_bytes`、`Hub::cache_usage` / `cache_entries` / `enforce_quota`、`CacheResidency` 常驻保护；CLI `/api/cache` 与 `sys1 list` 输出 limit/used/free/reclaimable，`sys1 doctor` 输出配额状态。测试覆盖 LRU 顺序、常驻保护、超大模型拒绝、pull 触发淘汰与活动下载保护。ADR 0012。
- 生产收尾：CORS allowlist/关闭与优雅关闭已实现并有测试；CLI Ctrl-C/SIGTERM 停止接收并 drain。
- CI：三平台 fmt/clippy/test；可选 self-hosted GPU workflow 运行长上下文对齐与 loadgen 冒烟。
- 官方 golden：三个 checkpoint 的 `case_dict_state` 答案在 1e-3 内一致，含 `action.act_probability`；multilingual 特殊 token 已修正。
- 路由与 act head：官方 routing probe 159/159 用例的 model、workflow、reason 与 detection 完全一致；act head 由 checkpoint 的 `act_head.*` 自动启用并写入 `DecisionOutput.actions`（官方 wire 不包含）。ONNX 导出新增 `pooled` 输出与 `act_head.safetensors` sidecar，actions 与 candle 在 2e-3 内一致。细节见 ADR 0013 / 0016。
- 长上下文扫描：`DecisionModel::predict_long`、`LongContextOptions`、`WindowAttribution` 与 `sys1 run --long` 已实现；默认窗口受问题 state 余量限制、50% 重叠，`noul` 取最大、`choice`/`score` 取最自信窗口。英文 checkpoint 与官方 `predict_long` 的答案、act、窗口归因与 usage 对齐。细节见 ADR 0014。
- Abstention：`PredictionOptions`/`MinConfidence` 支持标量与按 question type/选项数分桶的阈值，`DecisionOutput.abstentions` 保存 `passed`/`abstained`/`unevaluated` 与 `low_confidence`，答案 payload 与 wire 不变；`sys1 run --min-confidence` 输出官方同名字段。英文 checkpoint 的标量与 bucket gate 与官方一致。细节见 ADR 0015。
- 真实上游：`sys1-hub` 的 `SYS1_LIVE_HUB=1` 测试可解析真实 HF 与 ModelScope manifest；受限网络可用 `HF_ENDPOINT=https://hf-mirror.com`。
- 请求解压：gzip 与 zstd 均在进程内解压，压缩前后都受 `max_body_bytes` 限制；损坏帧返回 400，解压后超限返回 413（`ruzstd` 纯 Rust 后端）。
- 待补：parallel layout、ORT EP/TensorRT 实机、GPU CI。

## 11. 已确认与遗留

已确认：

1. 超长 state：400 + `max_tokens_exceeded`，不静默截断（已实现）。
2. 二进制名：`sys1`（已实现）。
3. `laya-latest`：自动注册三个显式变体，并用与官方 golden 对齐的 Router 选择 checkpoint。

遗留：

1. Laya 权重许可证：再分发前确认；v1 不内置权重。
2. Laya 原生 confidence 的 HTTP 暴露方式：独立 `/api/*` 端点或响应头（v1.1）。
3. ORT EP/TensorRT 实机、parallel layout；GPU CI。

## 12. 决策索引

| ID | 决策 |
|---|---|
| Q1/Q2 | 官方 TypeSafe 契约是唯一兼容基线；不追求 sys1 0.0.3 平替 |
| Q3 | 只做 Rust 进程内推理接口；HTTP 客户端由 jev-sdk-rs 负责 |
| Q4 | typed questions/answers；`Model<M>` + 对象安全 facade |
| Q5 | 编译期 trait + feature + 运行时注册表；无动态插件 ABI |
| Q6 | 独立 hub 层，SDK/CLI 共用缓存 |
| Q7 | 暂不发布 crates.io；内部库 + 二进制 |
| Q8 | 与 Shotspace 无关 |
| Q9 | 单进程/单二进制，进程内模型；无 runner 子进程、无 MCP |
| Q10 | Jev 仅兼容目标，不代理 |
| Q11 | v1 本地模型族仅 Laya |
| Q12 | 严格官方 `/v1/*`；能力缺口用 422 `unsupported` |
| Q13 | 同步核心 + 可选 async；server 层可选 Batcher |
| Q14 | `laya-latest` Router + 三个显式变体；`jev-latest` 别名默认关闭 |
| Q15 | 最小 JSON manifest + 按模型缓存 + sha256/断点续传 |
| Q16 | 多 crate workspace；Laya clean-room 重写，不依赖旧库 |
| Q17 | CLI `serve/pull/list/show/rm/run` + 配置优先级 |
| Q18 | 官方 Python SDK + jev-sdk-rs + golden fixtures 验收 |
| Q19 | v1 用 candle；ONNX/ORT 后续 |
| Q20 | 独立实现；允许参考 Apache-2.0 代码并保留声明 |
| Q21 | 官方 Laya golden vectors；标签一致，概率容差 1e-4 |
| Q22 | 自定义中心先做静态 HTTP manifest |
| Q23 | Laya v1 核心面 + Router + batch + confidence；暂缓 long/act/parallel |
| Q24 | `/v1/*` 默认 TypeSafe 归一化 confidence；可切 Laya 原生 |
| Q25 | 超长 state 返回 400 `max_tokens_exceeded`，不静默截断 |
| Q32 | CLI 二进制名 `sys1` |
| Q33 | 实现顺序：protocol → core/hub → laya → server → cli → umbrella |
| Q26/Q30 | 内置默认全编译；features 只做裁剪/加速 |
| Q27 | `--device auto` + NVML + doctor + 启动日志/health |
| Q28 | CPU 三平台 CI；CUDA/Metal 编译检查；GPU self-hosted 可选 |
| Q29 | manifest 预留 `backend`/`format` |
| Q31 | TensorRT 走 v1.1 ORT TensorRT EP，不做 candle 原生 TensorRT |
| Q34 | 运行时模型管理：可变目录 + 可选 resolver + `/api/models` 管理端点；不持久化，配置为准 |
| Q35 | 高层 `InferenceEngine` 放 umbrella；Laya 前缀 API 只留在 `sys1-laya`，其他 crate 用通用命名 |
| Q36 | 可替换 telemetry：库只发事件、CLI 装 subscriber；W3C traceparent 贯穿请求、模型与缓存审计 |
| Q37 | 配置热加载只做 `[models]` 安全增删；其他设置与已有引用变更要求重启 |
