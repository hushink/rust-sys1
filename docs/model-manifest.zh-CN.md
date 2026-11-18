# 自定义 HTTP 模型中心 manifest

[English](model-manifest.md)

自定义模型中心为每个模型提供一个 `model.json` 以及它引用的文件。CLI 和 SDK
通过 `http`/`https` source 获取：

```bash
sys1 pull https://example.com/models/laya-typed-decisions/model.json
# 等价写法：自动追加 /model.json
sys1 pull https://example.com/models/laya-typed-decisions
```

或在 `sys1.toml` 中：

```toml
[models]
laya-typed = "https://example.com/models/laya-typed-decisions/model.json"
```

返回值是一个 `ModelManifest`。所有字段都会在下载任何文件之前完成校验。
随文档提供了一份机器可读的
[JSON Schema](model-manifest.schema.json)。

## 结构示例

```json
{
  "schema_version": 1,
  "name": "laya-typed-decisions",
  "description": "Laya typed-decisions decision model",
  "release_date": "2026-11-18",
  "family": "laya",
  "variant": "typed-decisions",
  "backend": "candle",
  "format": "safetensors",
  "version": "2026-11-18",
  "aliases": ["laya-typed", "typed-decisions"],
  "files": [
    {
      "path": "model.safetensors",
      "url": "files/model.safetensors",
      "sha256": null,
      "size": 643835514
    },
    {
      "path": "encoder/config.json",
      "url": "files/encoder/config.json",
      "sha256": null,
      "size": 1897
    },
    {
      "path": "rl_agent_config.json",
      "url": "files/rl_agent_config.json",
      "sha256": null,
      "size": 316
    },
    {
      "path": "tokenizer/tokenizer.json",
      "url": "files/tokenizer/tokenizer.json",
      "sha256": null,
      "size": 34363440
    }
  ]
}
```

`"sha256": null`（或省略该字段）会关闭校验和验证，生产环境不建议这样做。
请替换为每个文件的小写十六进制摘要；下载器会校验，不匹配时最多重试四次。
上例中的文件大小和文件集合仅作示意；请按 checkpoint 实际需要的文件填写。

## 字段说明

### Manifest

| 字段 | 类型 | 必填 | 规则 |
|---|---|---|---|
| `schema_version` | 整数 | 是 | 必须为 `1`。 |
| `name` | string | 是 | 非空。`/v1/systemone` 使用的模型名；别名也可以。 |
| `description` | string | 是 | 非空；显示在 `sys1 list` 和 `GET /api/models`。 |
| `release_date` | string | 是 | `YYYY-MM-DD`，且必须是有效日历日期。 |
| `family` | string | 是 | 非空，取值 `laya`、`nli`、`gliclass`、`qwen3guard`、`von`、`decima`、`kev`、`decision`、`clef` 之一。注册时会根据下载目录识别布局并规范化该字段，但仍应填写 checkpoint 的真实家族。 |
| `variant` | string | 否 | 可选标签，如 `multilingual`、`typed-decisions`。 |
| `backend` | string | 是 | 非空，如 `candle`、`onnx`。 |
| `format` | string | 是 | 非空，如 `safetensors`、`onnx`、`gguf`。 |
| `version` | string | 是 | 非空；自由格式，可用 tag、日期或 commit id。 |
| `aliases` | string[] | 否 | 每个非空、不重复、不等于 `name`，且不能与注册表中其他名称/别名冲突。 |
| `files` | object[] | 是 | 至少一个条目，见下表。 |

未知属性会被忽略，因此模型中心可以增加元数据字段而不破坏旧客户端。

### File

| 字段 | 类型 | 必填 | 规则 |
|---|---|---|---|
| `path` | string | 是 | 非空、相对路径、使用 `/`、不含 `..` 或反斜杠，且在同一 `files` 中唯一。这是文件在缓存模型目录中的落盘路径，必须符合对应 family 要求的布局（见该 crate 的 README）。 |
| `url` | string | 否 | 绝对 URL，或相对于 `model.json` 的 URL。省略或空字符串 `""` 表示用 `path` 相对 `model.json` 解析。 |
| `sha256` | string \| null | 否 | 64 位十六进制字符。存在时下载内容必须匹配。 |
| `git_sha1` | string \| null | 否 | 40 位十六进制字符。用于记录 Hugging Face 非 LFS 文件，不参与校验。 |
| `size` | 整数 | 是 | 字节数，必须大于 0；下载长度必须等于该值，同时用于缓存配额和 Range 续传。 |

决定磁盘布局的是 `path` 而不是 URL。例如 Decima checkpoint 需要
`pytorch/decima.json`、`pytorch/encoder/config.json` 等路径，即使文件实际
放在 CDN 的其他路径下。

## URL 解析

manifest URL 是所有文件的解析基准。对
`https://example.com/models/laya-typed-decisions/model.json`：

| `url` | 解析结果 |
|---|---|
| `""`，`path = "model.safetensors"` | `https://example.com/models/laya-typed-decisions/model.safetensors` |
| `"files/model.safetensors"` | `https://example.com/models/laya-typed-decisions/files/model.safetensors` |
| `"https://cdn.example.com/model.safetensors"` | 保持不变 |

引用不带 `.json` 后缀时，会自动追加 `/model.json`。

## HTTP 行为

- 所有请求都是 `GET`，并带 `Accept-Encoding: identity` 与
  `User-Agent: rust-sys1/<version>`。
- 设置 `SYS1_HUB_TOKEN` 或 `[hub] http_token` 后，manifest 与文件请求都会带
  `Authorization: Bearer <token>`。
- 下载使用 HTTP Range 续传；超时后保留 `.partial` 文件，超过单次下载超时
  （`[hub] download_timeout_secs`）的大文件会从已下载字节继续。
- manifest 和文件缓存在 `[hub] cache_dir` 下；manifest 未变化时仍会刷新缓存
  条目的最近使用时间。
- `sha256` 不匹配会删除 partial 文件并重试，四次后以 `HashMismatch` 失败。

## 最小服务端

任何静态文件服务器都可以：以 `Content-Type: application/json` 提供
`model.json`，并在对应路径提供引用文件。Range 请求是可选的，但能让大权重
支持断点续传。

不下载文件、只检查 manifest：

```bash
sys1 show https://example.com/models/laya-typed-decisions/model.json
```

`sys1 show` 会执行与 `pull` 相同的反序列化、校验和相对 URL 解析。

## 另见

- [`sys1-hub` README](../crates/sys1-hub/README.zh-CN.md)：source、缓存配额
  以及非静态中心使用的 `ModelSource` trait。
- 各 family 的 checkpoint 布局：
  [`sys1-laya`](../crates/sys1-laya/README.zh-CN.md)、
  [`sys1-nli`](../crates/sys1-nli/README.zh-CN.md)、
  [`sys1-gliclass`](../crates/sys1-gliclass/README.zh-CN.md)、
  [`sys1-qwen3guard`](../crates/sys1-qwen3guard/README.zh-CN.md)、
  [`sys1-von`](../crates/sys1-von/README.zh-CN.md)、
  [`sys1-decima`](../crates/sys1-decima/README.zh-CN.md)、
  [`sys1-kev`](../crates/sys1-kev/README.zh-CN.md)、
  [`sys1-decision`](../crates/sys1-decision/README.zh-CN.md)、
  [`sys1-clef`](../crates/sys1-clef/README.zh-CN.md)。
