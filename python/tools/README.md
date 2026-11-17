# sys1-tools

[简体中文](README.zh-CN.md)

Python tooling for rust-sys1, managed with [uv](https://docs.astral.sh/uv/).
It contains the ONNX exporter/checker, the synthetic ORT fixture generator, and
the HTTP benchmark that formerly lived in `scripts/bench.ps1`.

## Setup

Run from the repository root:

```bash
uv sync --all-extras
uv run sys1-bench --help
```

`uv` creates `.venv` and `uv.lock`; Python 3.10 or newer is required. Each tool
is its own console script, configured in `[project.scripts]`:

| Script | Purpose |
|---|---|
| `sys1-export-onnx` | export a checkpoint to ONNX |
| `sys1-check-onnx` | check logits and pooled output against golden data |
| `sys1-generate-scoring-stub` | regenerate the synthetic ORT fixture |
| `sys1-bench` | run the HTTP benchmark |

The package has optional extras so a benchmark-only install does not pull in
ONNX:

| Extra | Adds |
|---|---|
| `onnx` | `numpy`, `onnx`, `safetensors` for export and fixture generation |
| `check` | `numpy`, `onnxruntime` for golden-logit checks |
| `all` | both |

## Commands

### Export a Laya checkpoint to ONNX

```bash
uv run --extra onnx sys1-export-onnx \
  /path/to/laya-multilingual /path/to/laya-onnx
```

The export contains `model.onnx` (with `logits` and `pooled` outputs),
`onnx.json`, `rl_agent_config.json`, the tokenizer, and `act_head.safetensors`
when the checkpoint has an act head. `--skip-assets` writes only `model.onnx`.

### Check against official golden logits

```bash
uv run --extra check sys1-check-onnx \
  /path/to/laya-onnx /path/to/golden/multilingual/case_dict_state.json
```

The command validates the `pooled` shape and the maximum
`|onnx - golden|` logit error; `--tolerance` defaults to `1e-4`.

### Regenerate the synthetic ORT fixture

```bash
uv run --extra onnx sys1-generate-scoring-stub
```

Without `--output`, it writes
`crates/sys1-ort/tests/fixtures/scoring_stub.onnx` in the repository.

### Run the HTTP benchmark

Build the release CLI first:

```bash
cargo build --release --offline -p sys1-cli
```

Then point the benchmark at a local checkpoint:

```bash
uv run sys1-bench --model-dir /path/to/laya-multilingual
uv run sys1-bench --model-dir ... --concurrency-sweep --option-sweep
uv run sys1-bench --model-dir ... --json
```

The tool starts `sys1 serve` on `127.0.0.1:18082`, waits for `/health`, runs
the same scenarios as the old PowerShell script, prints markdown-friendly
tables, and stops the server on exit. Useful flags mirror the old script:
`--exe`, `--model-name`, `--device`, `--dtype`, `--confidence`, `--warmup`,
`--sequential`, `--batch-sequential`, `--option-requests`, `--batch-max`,
`--batch-wait-ms`, `--no-batch`, `--tf32`, `--unique-states`, and `--json`.
`SYS1_BENCH_MODEL` and `SYS1_BENCH_EXE` provide the same defaults.

## Development

```bash
uv run --package sys1-tools --group dev ruff check python/tools
uv run --package sys1-tools --group dev ruff format --check python/tools
```
