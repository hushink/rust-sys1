# sys1-hub

[English](README.md)

rust-sys1 的模型源、manifest 解析与磁盘缓存。

## 模型源

| Scheme | 示例 |
|---|---|
| 本地路径 | `local:/models/laya-multilingual` 或已存在的路径 |
| Hugging Face | `hf:convaiinnovations/laya@main#multilingual` |
| ModelScope | `modelscope:owner/name@master` |
| 静态 HTTP 中心 | `https://models.example.com/laya/model.json` |

语法：`scheme:location[@revision][#subfolder]`。

## Manifest

```json
{
  "schema_version": 1,
  "name": "laya-multilingual",
  "description": "Laya multilingual decision model",
  "release_date": "2026-01-01",
  "family": "laya",
  "variant": "multilingual",
  "backend": "candle",
  "format": "safetensors",
  "version": "main",
  "files": [
    {"path": "model.safetensors", "url": "...", "sha256": "...", "size": 643835514}
  ]
}
```

`sha256` 可选；HF 非 LFS 文件用 `git_sha1` 兜底。路径必须是相对路径、使用正斜杠、
且不能包含 `..`。

## 示例

```rust
use sys1_hub::{Hub, HubConfig};

let hub = Hub::new(HubConfig::from_env())?;
let pulled = hub.pull("hf:convaiinnovations/laya@main#multilingual")?;
println!("{} at {}", pulled.manifest.name, pulled.directory.display());
```

`pull` 返回 `PulledModel`（manifest、目录、每个文件的路径）。`resolve` 只解析
manifest，不下载文件。

## 缓存与网络行为

- 缓存 key 稳定、适合文件系统、带哈希。
- 下载使用 `.partial`、HTTP Range 断点续传与 sha256 校验。
- 同一模型的并发拉取在进程内串行化。
- 离线模式读取缓存 manifest 并校验已有文件。
- `HF_ENDPOINT` 支持镜像（`https://hf-mirror.com`）。
- 支持 `MODELSCOPE_API_TOKEN`、`HF_TOKEN`、`SYS1_HUB_TOKEN`。
- ModelScope 文件列表的 3000 条静默截断通过按目录拆分处理。

自定义中心实现 `ModelSource`，通过 `Hub::with_sources` 注入。

## 缓存配额

设置 `HubConfig::max_cache_bytes`（`None` 或 `0` 表示不限制）。在物化远程模型前，
`Hub::pull` 会估算缺失字节，并按 LRU 淘汰非驻留缓存目录，直到新文件能够放入配额。

- LRU 顺序使用 `model.json` 的修改时间；每次成功 pull（含缓存命中）都会刷新。
- 通过 `Hub::set_residency` 安装 `CacheResidency`，保护进程内已加载的模型。
  `InferenceEngine` 与 `sys1-cli` 会自动安装基于注册表的实现。
- 进行中的 pull 与被拉取的条目不会被淘汰；下载期间 `remove` / `remove_cached`
  返回 `HubError::CacheBusy`，常驻 provider 报告模型已加载时返回
  `HubError::CacheResident`。
- 模型本身大于配额，或空间被常驻条目占满时，返回
  `HubError::CacheQuotaExceeded`，包含 `limit`、`used`、`additional`、
  `reclaimable` 字节数。
- `cache_entries()` 返回实际目录大小、时间戳与 `evictable`；`cache_usage()`
  返回 `limit_bytes`、`used_bytes`、`free_bytes`、`reclaimable_bytes` 与计数。
  `enforce_quota(additional)` 可在不 pull 的情况下单独执行策略。
- `local:` 源不会复制进缓存，不占用配额。

## 测试

```bash
cargo test --offline -p sys1-hub
```

集成测试会启动本地回环模型中心，覆盖：

- 静态 HTTP manifest、相对 URL、Bearer token 与 Range 断点续传下载；
- Hugging Face tree 分页（`Link` 头）、子目录、LFS sha256、revision/路径 URL
  编码、发布日期与 token 头；
- ModelScope `Root`/`Recursive` 列表、token + cookie 头，以及 3000 条递归列表
  截断后的目录拆分；
- LRU 配额淘汰、常驻模型保护、超大模型拒绝，以及 pull 触发的淘汰。

真实上游检查默认跳过，需显式开启：

```powershell
$env:SYS1_LIVE_HUB = "1"
# huggingface.co 不可达时可选用镜像：
$env:HF_ENDPOINT = "https://hf-mirror.com"
cargo test --offline -p sys1-hub --test live -- --nocapture
```

测试会解析真实的 Hugging Face manifest（`convaiinnovations/laya-multilingual`）
与 ModelScope manifest（`Qwen/Qwen2.5-0.5B-Instruct`），并校验返回的文件列表与大小。
