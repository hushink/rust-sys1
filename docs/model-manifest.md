# Static HTTP model center manifest

[简体中文](model-manifest.zh-CN.md)

A custom model center serves one `model.json` per model plus the files it
references. The CLI and SDK fetch it through the `http`/`https` source:

```bash
sys1 pull https://example.com/models/laya-typed-decisions/model.json
# The same, with /model.json appended automatically:
sys1 pull https://example.com/models/laya-typed-decisions
```

or in `sys1.toml`:

```toml
[models]
laya-typed = "https://example.com/models/laya-typed-decisions/model.json"
```

The response is a `ModelManifest`. All fields are validated before any file is
downloaded. A machine-readable
[JSON Schema](model-manifest.schema.json) is provided alongside this document.

## Schema

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

`sha256: null` (or omitting the field) disables checksum verification and is
not recommended for production. Replace it with the lowercase hex digest of
each file; the downloader verifies it and retries a mismatch up to four times.
The file sizes and file set above are illustrative; list the files your
checkpoint actually needs.

## Fields

### Manifest

| Field | Type | Required | Rules |
|---|---|---|---|
| `schema_version` | integer | yes | Must be `1`. |
| `name` | string | yes | Non-empty. The model name used by `/v1/systemone`; aliases may also be passed. |
| `description` | string | yes | Non-empty; shown by `sys1 list` and `GET /api/models`. |
| `release_date` | string | yes | `YYYY-MM-DD` and a valid calendar date. |
| `family` | string | yes | Non-empty, one of `laya`, `nli`, `gliclass`, `qwen3guard`, `von`, `decima`, `kev`, `decision`, `clef`. Registration detects the layout from the downloaded directory and normalizes this field, but it should identify what the checkpoint actually is. |
| `variant` | string | no | Optional label such as `multilingual` or `typed-decisions`. |
| `backend` | string | yes | Non-empty, e.g. `candle` or `onnx`. |
| `format` | string | yes | Non-empty, e.g. `safetensors`, `onnx`, or `gguf`. |
| `version` | string | yes | Non-empty. Free-form; a tag, date, or commit id. |
| `aliases` | string[] | no | Each non-empty, unique, and different from `name`. Aliases must also be unique across the registry. |
| `files` | object[] | yes | At least one entry; see below. |

Unknown properties are ignored, so a center may add metadata fields without
breaking older clients.

### File

| Field | Type | Required | Rules |
|---|---|---|---|
| `path` | string | yes | Non-empty, relative, forward slashes, no `..` or backslashes, unique within `files`. This is the path inside the cached model directory, so it must match the layout the family expects (see the family's crate README). |
| `url` | string | no | Absolute URL or a URL relative to `model.json`. Omitted or `""` resolves `path` against `model.json`. |
| `sha256` | string \| null | no | 64 lowercase/uppercase hex characters. When present, the downloaded bytes must match. |
| `git_sha1` | string \| null | no | 40 hex characters. Informational for Hugging Face non-LFS files; not used for verification. |
| `size` | integer | yes | Bytes, greater than zero. Must equal the downloaded length; used for cache quota and Range resume. |

The `path` values, not the URLs, determine the on-disk layout. For example a
Decima checkpoint needs `pytorch/decima.json`, `pytorch/encoder/config.json`,
and so on, even if the files live under a CDN path.

## URL resolution

The manifest URL is the base for every file:

| `url` | Resolved against `https://example.com/models/laya-typed-decisions/model.json` |
|---|---|
| `""`, `path = "model.safetensors"` | `https://example.com/models/laya-typed-decisions/model.safetensors` |
| `"files/model.safetensors"` | `https://example.com/models/laya-typed-decisions/files/model.safetensors` |
| `"https://cdn.example.com/model.safetensors"` | unchanged |

A reference without a `.json` suffix gets `/model.json` appended.

## HTTP behavior

- Every request is a `GET` with `Accept-Encoding: identity` and
  `User-Agent: rust-sys1/<version>`.
- Set `SYS1_HUB_TOKEN`, or `[hub] http_token`, to send
  `Authorization: Bearer <token>` to both the manifest and file requests.
- Downloads use HTTP Range resume and keep a `.partial` file on timeouts.
  A file larger than the per-attempt timeout resumes from the bytes already on
  disk (see `[hub] download_timeout_secs`).
- The manifest and files are cached under `[hub] cache_dir`. An unchanged
  `model.json` still refreshes the cache entry's last-used time.
- `sha256` mismatches delete the partial file and retry; after four attempts
  the pull fails with `HashMismatch`.

## Minimal server

Any static file server works: serve `model.json` with
`Content-Type: application/json` and the referenced files at their paths.
Range requests are optional but enable resume for large weights.

Check a manifest without downloading files:

```bash
sys1 show https://example.com/models/laya-typed-decisions/model.json
```

`sys1 show` performs the same deserialization, validation, and relative-URL
resolution as `pull`.

## See also

- [`sys1-hub` README](../crates/sys1-hub/README.md) — sources, cache quota,
  and the `ModelSource` trait for non-static centers.
- Family checkpoint layouts: [`sys1-laya`](../crates/sys1-laya/README.md),
  [`sys1-nli`](../crates/sys1-nli/README.md),
  [`sys1-gliclass`](../crates/sys1-gliclass/README.md),
  [`sys1-qwen3guard`](../crates/sys1-qwen3guard/README.md),
  [`sys1-von`](../crates/sys1-von/README.md),
  [`sys1-decima`](../crates/sys1-decima/README.md),
  [`sys1-kev`](../crates/sys1-kev/README.md),
  [`sys1-decision`](../crates/sys1-decision/README.md),
  [`sys1-clef`](../crates/sys1-clef/README.md).
