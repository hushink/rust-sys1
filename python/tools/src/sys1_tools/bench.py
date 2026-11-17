"""HTTP benchmark for a running `sys1 serve` process.

The tool starts the release CLI with a local checkpoint, waits for `/health`,
then measures sequential, concurrent, and option-sweep scenarios. It replaces
`scripts/bench.ps1` and keeps the same scenario names and statistics.

Usage:

    uv run sys1-bench --model-dir C:/models/laya-multilingual
    uv run sys1-bench --model-dir ... --concurrency-sweep --option-sweep
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import httpx

STATE_TEXT = "I was charged twice. Please refund the extra charge today."
DEFAULT_TIMEOUT_SECS = 300.0


def repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def default_exe() -> Path:
    name = "sys1.exe" if os.name == "nt" else "sys1"
    return repo_root() / "target" / "release" / name


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=(
            Path(os.environ["SYS1_BENCH_MODEL"]) if os.environ.get("SYS1_BENCH_MODEL") else None
        ),
        help="Laya checkpoint directory (default: SYS1_BENCH_MODEL)",
    )
    parser.add_argument("--model-name", default="laya-multilingual")
    parser.add_argument(
        "--family",
        default="generic",
        choices=["generic", "qwen3guard"],
        help="question preset: generic noul/score/choice, or Qwen3Guard's fixed questions",
    )
    parser.add_argument(
        "--exe",
        type=Path,
        default=Path(os.environ["SYS1_BENCH_EXE"]) if os.environ.get("SYS1_BENCH_EXE") else None,
        help="sys1 executable (default: target/release/sys1[.exe])",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="f32")
    parser.add_argument("--confidence", default="laya")
    parser.add_argument("--port", type=int, default=18082)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--sequential", type=int, default=20)
    parser.add_argument("--batch-sequential", type=int, default=10)
    parser.add_argument("--option-requests", type=int, default=8)
    parser.add_argument("--batch-max", type=int, default=32)
    parser.add_argument("--batch-wait-ms", type=int, default=2)
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_SECS)
    parser.add_argument("--no-batch", action="store_true")
    parser.add_argument(
        "--no-concurrent",
        action="store_true",
        help="skip the fixed x4/x8 concurrency scenarios (and the concurrency sweep)",
    )
    parser.add_argument("--tf32", action="store_true")
    parser.add_argument("--unique-states", action="store_true")
    parser.add_argument("--concurrency-sweep", action="store_true")
    parser.add_argument("--option-sweep", action="store_true")
    parser.add_argument("--json", action="store_true", help="print one JSON document")


def run(args: argparse.Namespace) -> int:
    if args.model_dir is None:
        raise SystemExit("Set SYS1_BENCH_MODEL or pass --model-dir <checkpoint directory>")
    model_dir = Path(args.model_dir).expanduser().resolve()
    if not model_dir.is_dir():
        raise SystemExit(f"model directory does not exist: {model_dir}")
    exe = Path(args.exe).expanduser().resolve() if args.exe else default_exe()
    if not exe.is_file():
        raise SystemExit(
            f"build first: cargo build --release --offline -p sys1-cli (missing {exe})"
        )

    base = f"http://127.0.0.1:{args.port}"
    log_out = Path(tempfile.gettempdir()) / f"sys1-bench-{args.port}.out.log"
    log_err = Path(tempfile.gettempdir()) / f"sys1-bench-{args.port}.err.log"
    serve_args = [
        str(exe),
        "serve",
        "--bind",
        f"127.0.0.1:{args.port}",
        "--model",
        f"{args.model_name}=local:{model_dir}",
        "--device",
        args.device,
        "--dtype",
        args.dtype,
        "--laya-confidence",
        args.confidence,
        "--batch-max",
        str(args.batch_max),
        "--batch-wait-ms",
        str(args.batch_wait_ms),
    ]
    if args.no_batch:
        serve_args.append("--no-batch")
    if args.tf32:
        serve_args.append("--tf32")

    out_handle = log_out.open("wb")
    err_handle = log_err.open("wb")
    process = subprocess.Popen(
        serve_args,
        cwd=repo_root(),
        stdout=out_handle,
        stderr=err_handle,
    )
    try:
        timeout = httpx.Timeout(args.timeout)
        with httpx.Client(timeout=timeout) as client:
            _wait_ready(client, base)
            results = _run_scenarios(client, base, args)
    except Exception:
        _print_log_tail(log_out, "stdout")
        _print_log_tail(log_err, "stderr")
        raise
    finally:
        _stop(process)
        out_handle.close()
        err_handle.close()

    payload: dict[str, Any] = {
        "cpu": _cpu_name(),
        "cpu_logical": os.cpu_count(),
        "ram_gib": _ram_gib(),
        "model": args.model_name,
        "family": args.family,
        "dtype": args.dtype,
        "device": args.device,
        "model_dir": str(model_dir),
        "results": results["results"],
    }
    if results["concurrency"]:
        payload["concurrency"] = results["concurrency"]
    if results["options"]:
        payload["options"] = results["options"]

    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        _print_human(payload)
    return 0


def _qwen3guard_questions(full: bool) -> dict[str, Any]:
    questions: dict[str, Any] = {
        "unsafe": {
            "type": "noul",
            "instructions": "The content is unsafe.",
        }
    }
    if full:
        questions["safety"] = {
            "type": "choice",
            "criteria": {
                "safe": None,
                "controversial": None,
                "unsafe": None,
            },
        }
        questions["category"] = {
            "type": "choice",
            "criteria": {
                label: None
                for label in [
                    "none",
                    "violent",
                    "non_violent_illegal",
                    "sexual",
                    "pii",
                    "suicide_self_harm",
                    "unethical",
                    "politically_sensitive",
                    "copyright",
                    "jailbreak",
                ]
            },
        }
    return questions


def _run_scenarios(
    client: httpx.Client, base: str, args: argparse.Namespace
) -> dict[str, list[dict[str, Any]]]:
    if args.family == "qwen3guard":
        noul = json.dumps(
            {
                "model": args.model_name,
                "state": STATE_TEXT,
                "questions": _qwen3guard_questions(False),
            },
            separators=(",", ":"),
        )
        batch3 = json.dumps(
            {
                "model": args.model_name,
                "state": STATE_TEXT,
                "questions": _qwen3guard_questions(True),
            },
            separators=(",", ":"),
        )
        noul_label = "POST unsafe noul"
        batch_label = "POST safety+unsafe+category"
    else:
        noul = json.dumps(
            {
                "model": args.model_name,
                "state": STATE_TEXT,
                "questions": {
                    "billing": {
                        "type": "noul",
                        "instructions": "Is this message about billing?",
                    }
                },
            },
            separators=(",", ":"),
        )
        batch3 = json.dumps(
            {
                "model": args.model_name,
                "state": STATE_TEXT,
                "questions": {
                    "billing": {
                        "type": "noul",
                        "instructions": "Is this message about billing?",
                    },
                    "urgency": {
                        "type": "score",
                        "instructions": "How urgent is this message?",
                        "criteria": ["Can wait", "This week", "Today"],
                    },
                    "tone": {
                        "type": "choice",
                        "instructions": "What is the tone of this message?",
                        "criteria": {
                            "angry": "An upset or hostile message",
                            "calm": "A neutral or polite message",
                        },
                    },
                },
            },
            separators=(",", ":"),
        )
        noul_label = "POST noul"
        batch_label = "POST noul+score+choice"

    # The first request loads the checkpoint.
    _post_json(client, f"{base}/v1/systemone", noul)

    results = [
        _measure_sequential_gets(client, base, "GET /health", "/health", 50, args.warmup),
        _measure_sequential_gets(client, base, "GET /v1/models", "/v1/models", 50, args.warmup),
        _measure_sequential_posts(client, base, noul_label, noul, args.sequential, args.warmup),
        _measure_sequential_posts(
            client, base, batch_label, batch3, args.batch_sequential, args.warmup
        ),
    ]
    if not args.no_concurrent:
        results.extend(
            [
                _measure_concurrent(
                    base, "POST noul x4", noul, 4, 5, args.unique_states, args.timeout
                ),
                _measure_concurrent(
                    base, "POST noul x8", noul, 8, 5, args.unique_states, args.timeout
                ),
            ]
        )

    concurrency: list[dict[str, Any]] = []
    if args.concurrency_sweep and not args.no_concurrent:
        for workers in (1, 2, 4, 8, 16, 32):
            concurrency.append(
                _measure_concurrent(
                    base,
                    f"noul x{workers}",
                    noul,
                    workers,
                    4,
                    args.unique_states,
                    args.timeout,
                )
            )

    options: list[dict[str, Any]] = []
    if args.option_sweep and args.family == "qwen3guard":
        print(
            "Qwen3Guard accepts only its fixed questions; skipping the option sweep.",
            file=sys.stderr,
        )
    elif args.option_sweep:
        for count in (2, 3, 5, 10, 20, 40):
            options.append(
                _measure_option(client, base, args, "choice", count, args.option_requests)
            )
        for count in (2, 3, 5, 10):
            options.append(
                _measure_option(client, base, args, "score", count, args.option_requests)
            )

    return {"results": results, "concurrency": concurrency, "options": options}


def _post_json(client: httpx.Client, url: str, body: str) -> dict[str, Any]:
    response = client.post(url, content=body, headers={"content-type": "application/json"})
    response.raise_for_status()
    return response.json()


def _wait_ready(client: httpx.Client, base: str) -> None:
    for _ in range(240):
        try:
            if client.get(f"{base}/health").status_code == 200:
                return
        except httpx.HTTPError:
            pass
        time.sleep(0.5)
    raise RuntimeError("server did not become ready")


def _measure_sequential_posts(
    client: httpx.Client,
    base: str,
    label: str,
    body: str,
    count: int,
    warmup: int,
) -> dict[str, Any]:
    for _ in range(warmup):
        _post_json(client, f"{base}/v1/systemone", body)
    times: list[float] = []
    wall = time.perf_counter()
    for _ in range(count):
        started = time.perf_counter()
        _post_json(client, f"{base}/v1/systemone", body)
        times.append((time.perf_counter() - started) * 1000.0)
    return _row(label, count, 1, times, time.perf_counter() - wall)


def _measure_sequential_gets(
    client: httpx.Client,
    base: str,
    label: str,
    path: str,
    count: int,
    warmup: int,
) -> dict[str, Any]:
    for _ in range(warmup):
        client.get(f"{base}{path}").raise_for_status()
    times: list[float] = []
    wall = time.perf_counter()
    for _ in range(count):
        started = time.perf_counter()
        response = client.get(f"{base}{path}")
        response.raise_for_status()
        response.read()
        times.append((time.perf_counter() - started) * 1000.0)
    return _row(label, count, 1, times, time.perf_counter() - wall)


def _measure_concurrent(
    base: str,
    label: str,
    body: str,
    workers: int,
    per_worker: int,
    unique_states: bool,
    timeout_secs: float,
) -> dict[str, Any]:
    def worker(index: int) -> list[float]:
        times: list[float] = []
        timeout = httpx.Timeout(timeout_secs)
        with httpx.Client(timeout=timeout) as client:
            for request_index in range(per_worker):
                request_body = (
                    body.replace(
                        STATE_TEXT,
                        f"{STATE_TEXT} Case {index}-{request_index}.",
                    )
                    if unique_states
                    else body
                )
                started = time.perf_counter()
                _post_json(client, f"{base}/v1/systemone", request_body)
                times.append((time.perf_counter() - started) * 1000.0)
        return times

    wall = time.perf_counter()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        batches = list(pool.map(worker, range(workers)))
    elapsed = time.perf_counter() - wall
    times = [value for batch in batches for value in batch]
    return _row(label, workers * per_worker, workers, times, elapsed)


def _measure_option(
    client: httpx.Client,
    base: str,
    args: argparse.Namespace,
    kind: str,
    options: int,
    requests: int,
) -> dict[str, Any]:
    if kind == "choice":
        question: dict[str, Any] = {
            "type": "choice",
            "instructions": "Pick the best option.",
            "criteria": {f"c{index}": f"option {index}" for index in range(options)},
        }
        answer_key = "pick"
    else:
        question = {
            "type": "score",
            "instructions": "Rate the content.",
            "criteria": [f"level {index}" for index in range(options)],
        }
        answer_key = "rating"
    body = json.dumps(
        {
            "model": args.model_name,
            "state": STATE_TEXT,
            "questions": {answer_key: question},
        },
        separators=(",", ":"),
    )

    _post_json(client, f"{base}/v1/systemone", body)
    times: list[float] = []
    sums: list[float] = []
    valid = True
    wall = time.perf_counter()
    for _ in range(requests):
        started = time.perf_counter()
        value = _post_json(client, f"{base}/v1/systemone", body)
        times.append((time.perf_counter() - started) * 1000.0)
        answer = value["answers"][answer_key]
        probabilities = answer.get("probabilities", {})
        if len(probabilities) != options:
            valid = False
        sums.append(sum(float(item) for item in probabilities.values()))
        if kind == "score" and len(answer.get("legend", {})) != options:
            valid = False
    stats = _stats(times)
    return {
        "question": kind,
        "options": options,
        "requests": requests,
        "mean_ms": stats["mean_ms"],
        "p50_ms": stats["p50_ms"],
        "p95_ms": stats["p95_ms"],
        "max_ms": stats["max_ms"],
        "rps": round(requests / max(time.perf_counter() - wall, 1e-9), 2),
        "prob_sum": round(sum(sums) / len(sums), 4) if sums else 0.0,
        "valid": valid,
    }


def _stats(values: list[float]) -> dict[str, float | int]:
    ordered = sorted(values)
    count = len(ordered)
    if count == 0:
        return {"count": 0, "mean_ms": 0.0, "p50_ms": 0.0, "p95_ms": 0.0, "max_ms": 0.0}
    return {
        "count": count,
        "mean_ms": round(sum(ordered) / count, 2),
        "p50_ms": round(ordered[max(0, math.floor(count * 0.50) - 1)], 2),
        "p95_ms": round(ordered[max(0, math.ceil(count * 0.95) - 1)], 2),
        "max_ms": round(ordered[-1], 2),
    }


def _row(
    label: str, requests: int, concurrency: int, times: list[float], wall: float
) -> dict[str, Any]:
    stats = _stats(times)
    return {
        "scenario": label,
        "requests": requests,
        "concurrency": concurrency,
        "mean_ms": stats["mean_ms"],
        "p50_ms": stats["p50_ms"],
        "p95_ms": stats["p95_ms"],
        "max_ms": stats["max_ms"],
        "rps": round(requests / max(wall, 1e-9), 2),
    }


def _stop(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=10)


def _print_log_tail(path: Path, label: str) -> None:
    if not path.is_file():
        return
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()[-100:]
    if lines:
        print(f"--- server {label} ---", file=sys.stderr)
        print("\n".join(lines), file=sys.stderr)


def _cpu_name() -> str:
    return platform.processor() or platform.machine() or "unknown"


def _ram_gib() -> float | None:
    try:
        import psutil  # type: ignore[import-not-found]

        return round(psutil.virtual_memory().total / (1024**3), 1)
    except ImportError:
        pass
    if os.name == "nt":
        import ctypes

        class MemoryStatus(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        status = MemoryStatus()
        status.dwLength = ctypes.sizeof(MemoryStatus)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return round(status.ullTotalPhys / (1024**3), 1)
        return None
    if sys.platform == "linux":
        try:
            for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
                if line.startswith("MemTotal:"):
                    return round(int(line.split()[1]) / (1024**2), 1)
        except OSError:
            return None
    if sys.platform == "darwin":
        try:
            value = subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True)
            return round(int(value.strip()) / (1024**3), 1)
        except (OSError, ValueError, subprocess.SubprocessError):
            return None
    return None


def _print_human(payload: dict[str, Any]) -> None:
    print("## sys1 benchmark")
    print()
    print(f"- CPU: {payload['cpu']} ({payload['cpu_logical']} logical cores)")
    if payload["ram_gib"] is not None:
        print(f"- RAM: {payload['ram_gib']} GiB")
    print(f"- Model: {payload['model']} ({payload['dtype']}, device={payload['device']})")
    print(f"- Model dir: {payload['model_dir']}")
    print()
    _print_table(payload["results"])
    if payload.get("concurrency"):
        print()
        print("## Concurrency sweep (noul)")
        _print_table(payload["concurrency"])
    if payload.get("options"):
        print()
        print("## Option sweep")
        _print_table(payload["options"])


def _print_table(rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    headers = {
        "scenario": "Scenario",
        "question": "Question",
        "options": "Options",
        "requests": "Requests",
        "concurrency": "Concurrency",
        "mean_ms": "MeanMs",
        "p50_ms": "P50Ms",
        "p95_ms": "P95Ms",
        "max_ms": "MaxMs",
        "rps": "Rps",
        "prob_sum": "ProbSum",
        "valid": "Valid",
    }
    columns = list(rows[0].keys())
    table = [[headers.get(column, column) for column in columns]]
    for row in rows:
        line = []
        for column in columns:
            value = row[column]
            if isinstance(value, float):
                line.append(f"{value:.2f}")
            else:
                line.append(str(value))
        table.append(line)
    widths = [max(len(line[index]) for line in table) for index in range(len(columns))]
    for line in table:
        print("  ".join(value.ljust(widths[index]) for index, value in enumerate(line)).rstrip())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_arguments(parser)
    return run(parser.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["add_arguments", "main", "run"]
