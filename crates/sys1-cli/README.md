# sys1-cli

[简体中文](README.zh-CN.md)

The `sys1` binary: model hub, one-shot inference, and TypeSafe-compatible server.

## Commands

| Command | Purpose |
|---|---|
| `sys1 serve` | Run the HTTP server |
| `sys1 pull <ref>` | Download/materialize a model |
| `sys1 list` | List cached models |
| `sys1 show <ref>` | Print a resolved manifest |
| `sys1 rm <ref>` | Remove a cached model |
| `sys1 run <ref> --input request.json` | One-shot inference |
| `sys1 doctor` | Print build/device/cache diagnostics |

## Model references

- `local:/path/to/model` or an existing path
- `hf:convaiinnovations/laya@main#multilingual`
- `modelscope:owner/name@master`
- `https://models.example.com/laya/model.json`

A local Laya directory without `model.json` is accepted directly:

```bash
sys1 run local:/path/to/laya-multilingual --input request.json
```

## Serve

```bash
# Single checkpoint
sys1 serve --model laya-multilingual=local:/path/to/laya-multilingual

# Repository root: registers three variants + lazy laya-latest router
sys1 serve --model laya-latest=local:/path/to/convaiinnovations/laya
```

Useful flags:

```text
--bind 127.0.0.1:8080
--api-key <key>            # or SYS1_API_KEY
--device auto|cpu|cuda:N|metal
--dtype auto|f32|f16|bf16   # f16 is faster at high CUDA concurrency
--tf32                     # optional TF32 for F32 CUDA GEMMs
--no-local-window          # use eager attention for local layers
--max-len <tokens>
--laya-confidence typesafe|laya
--batch-max 32             # requests per forward pass
--batch-wait-ms 2          # batch collection window
--no-batch                 # disable cross-request batching
--offline
--cache <dir>              # or SYS1_CACHE
```

## Load generator

```bash
cargo build --release --offline -p sys1-cli --example loadgen
target/release/examples/loadgen --url http://127.0.0.1:8080 \
  --concurrency 64 --requests 512
target/release/examples/loadgen --url http://127.0.0.1:8080 \
  --rate 300 --concurrency 128 --requests 1500
```

Closed-loop mode keeps `--concurrency` requests in flight. `--rate` is an
open-loop offered rate; requests that cannot acquire an in-flight permit are
reported as `dropped`.

## Features

```bash
cargo build --release -p sys1-cli --features cuda
cargo build --release -p sys1-cli --features cudnn
cargo build --release -p sys1-cli --features metal
```

`sys1 doctor` prints compiled features and the resolved device.

## Request file

```json
{
  "state": "I was charged twice. Please refund me today.",
  "questions": {
    "billing": {"type": "noul", "instructions": "Is this about billing?"}
  }
}
```

```bash
sys1 run local:/path/to/laya-multilingual --input request.json
```

## Benchmark

```powershell
$env:SYS1_BENCH_MODEL="C:/path/to/laya-multilingual"
pwsh ./scripts/bench.ps1
```

See the root README for results.
