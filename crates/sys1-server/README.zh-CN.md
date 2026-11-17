# sys1-server

[English](README.md)

基于 axum 的严格 TypeSafe-compatible System One HTTP 服务。

## 端点

| 方法 | 路径 | 含义 |
|---|---|---|
| `POST` | `/v1/systemone` | 官方决策端点 |
| `GET` | `/v1/models` | 官方模型/别名列表 |
| `GET` | `/health` | 运维健康检查，位于 `/v1` 之外 |
| `GET` | `/api/models` | 运行时目录、别名与常驻状态（admin） |
| `POST` | `/api/models/{name}/load` | 加载到内存 |
| `POST` | `/api/models/{name}/unload` | 释放常驻实例 |
| `POST` | `/api/models/{name}/reload` | 卸载后重新加载 |
| `POST` | `/api/models` | 从引用添加模型（需要 resolver） |
| `DELETE` | `/api/models/{name}` | 卸载并从运行时目录移除 |
| `GET` | `/metrics` | Prometheus 指标（启用时） |
| `GET` | `/api/metrics` | JSON 指标快照（启用时） |

有意不实现 `/v1/decisions`。

## 示例

```rust
use std::sync::Arc;
use sys1_core::{ModelRegistry, RegistryConfig};
use sys1_server::{ServerConfig, router, serve};

let registry = Arc::new(ModelRegistry::new(
    manifests,
    factory,
    RegistryConfig::default(),
)?);

// 挂进已有 axum app：
let app = router(registry.clone(), ServerConfig::default());

// 或独立运行：
serve("127.0.0.1:8080".parse()?, registry, ServerConfig::default()).await?;
```

`/api/*` 默认关闭，只有 `ServerConfig::enable_admin = true` 时才注册；CLI 默认开启。
需要 `POST /api/models` 时，用 resolver 构建 router：

```rust
use sys1_core::ModelResolver;
use sys1_server::router_with_resolver;

let resolver: Arc<dyn ModelResolver> = Arc::new(my_resolver);
let app = router_with_resolver(
    registry.clone(),
    ServerConfig { enable_admin: true, ..ServerConfig::default() },
    Some(resolver),
);
```

## 运行时模型管理

- `ModelRegistry::is_loaded` 查询常驻状态；`register` / `unregister` 修改运行时
  目录并丢弃常驻实例。
- `POST /api/models` 接收 `{"reference": "hf:owner/repo"}`，通过
  `ModelResolver` 解析并注册返回的所有 manifest。若名称已存在则返回 `409`，
  原有目录保持不变。
- 加 `"load": true` 会在返回 `201` 前加载主模型（bundle 的主模型是 router，
  各变体保持懒加载）；加载失败会回滚注册。
- `GET /api/models` 为每个模型输出 `origin: "configured" | "runtime"`。
- `DELETE /api/models/{name}` 还会调用 `ModelResolver::release`，让 resolver
  清理本地目录映射。
- 未配置 resolver 时，`POST /api/models` 返回 `501`；已配置模型的
  load/unload/reload 和 DELETE 仍可用。
- admin 请求与 `/v1/*` 使用同一个 Bearer key。未配置 key 时 admin API 是开放的，
  请保持绑定地址私密。
- 运行时添加/移除不会持久化；重启后以传给 `ModelRegistry::new` 的 manifest 为准。
- unload 是幂等的（`was_loaded` 返回之前的状态）。已经持有模型句柄的请求会正常完成。

## 指标

`ServerConfig::enable_metrics`（默认 `false`；CLI 默认开启）会注册两个只读端点：

- `GET /metrics` — Prometheus 文本（`text/plain; version=0.0.4`）。
- `GET /api/metrics` — 同样数据的 JSON 快照。

计数器从进程启动开始累计：

- `sys1_systemone_requests_total`、`sys1_systemone_errors_total`、
  `sys1_systemone_latency_ms_sum`、`sys1_systemone_latency_ms_max`
- `sys1_batches_total`、`sys1_batch_items_total`、`sys1_batch_items_max`、
  `sys1_batch_duration_ms_sum`
- `sys1_model_loads_total`、`sys1_model_load_errors_total`、
  `sys1_model_load_duration_ms_sum`、`sys1_model_unloads_total`
- `sys1_model_requests_total{model="..."}`（按规范化模型名）
- `sys1_loaded_models`、`sys1_uptime_seconds`

配置 API key 时，两个端点与 `/v1/*` 使用同一个 Bearer key。延迟求和单位为毫秒；
Prometheus 中除以对应的请求/batch/加载计数即可得到均值。计数器从 HTTP 服务启动时
开始；由 `sys1 serve --preload` 加载的模型会出现在 `loaded`/`loaded_count` 中，
但不会计入 `sys1_model_loads_total`。

## 行为

- 设置 `ServerConfig::api_key` 时启用 Bearer 认证。
- 请求体上限 32 MiB；支持 gzip 解压；zstd 返回 `415`。
- 响应带 `x-typesafe-request-id`。
- 校验失败 `422 {"detail": [...]}`。
- 未知模型 `400 {"detail": "..."}`。
- `box` / `assets` 会被解析，但 Laya 不支持：`422 unsupported`。
- box/assets 缺少 `TypeSafe-Danger-Zone-Images: true` 时返回 `400`。
- 超长 state 返回 `400 max_tokens_exceeded`。
- 推理与模型加载在 `tokio::task::spawn_blocking` 中执行。
- 默认启用跨请求 batching：`ServerConfig::batching` 中的
  `BatchConfig { max_batch: 32, max_wait: 2 ms }`。设为 `None` 可关闭。
  模型可通过 `DecisionModel::supports_cross_request_batching` 退出；CPU 上的
  Laya 返回 `false`，因为独立 CPU 前向已经使用线程池。GPU 单请求只付 0.5 ms
  的 yield grace period。

## 压测

i5-12400F（6C/12T）+ RTX 3060 12 GiB，`laya-multilingual`（mmBERT-base，322M），
并发请求使用不同 state：

| 场景 | CPU mean | CPU p95 | CPU req/s | CUDA F32 mean | CUDA F32 p95 | CUDA req/s |
|---|---:|---:|---:|---:|---:|---:|
| `GET /health` | 0.29 ms | 0.47 ms | 2534 | 0.31 ms | 0.54 ms | 2254 |
| `GET /v1/models` | 0.23 ms | 0.35 ms | 3879 | 0.28 ms | 0.43 ms | 3263 |
| `POST /v1/systemone`（noul） | 84.62 ms | 99.89 ms | 11.81 | 17.39 ms | 21.12 ms | 57.32 |
| `POST /v1/systemone`（noul+score+choice） | 199.72 ms | 214.56 ms | 5.01 | 21.26 ms | 24.06 ms | 46.96 |
| `POST /v1/systemone`（noul），8 并发 | 316.50 ms | 348.79 ms | 23.92 | 35.06 ms | 56.99 ms | 163.96 |
| `POST /v1/systemone`（noul），32 并发 | 1108.97 ms | 1486.42 ms | 25.82 | 88.49 ms | 152.53 ms | 209.25 |

CPU 无 batching，在 ~24–26 req/s 饱和。PowerShell 压测下 CUDA 在 32 并发达到
~209 req/s；Rust `loadgen` 示例在 F16 + `--batch-max 32` 下饱和约 ~345 req/s，
`--batch-max 64` + 256 in-flight 约 ~359 req/s。F16 抽查概率差异约 0.0005。
完整并发、F16、选项数量扫描与复现命令见根 README。

本机可以链接 cuDNN，但 Laya 没有卷积算子；不同 state 下 32 并发约
211 vs 普通 CUDA 209 req/s，在波动范围内。

## 测试

```bash
cargo test --offline -p sys1-server
```
