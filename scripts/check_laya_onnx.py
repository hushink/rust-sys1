#!/usr/bin/env python3
"""Check an exported Laya ONNX model against the official golden logits.

Usage:

    uv run --with onnxruntime --with numpy --python 3.13 \
      scripts/check_laya_onnx.py /path/to/laya-onnx \
      /path/to/golden/multilingual/case_dict_state.json

The exported directory is the one produced by `export_laya_onnx.py`. The
golden file provides `batch.input_ids`, `attention_mask`, `marker_pos`,
`marker_mask`, `qtype`, and the expected `logits`.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort


def flattened_markers(batch: dict, seq: int) -> tuple[list[int], list[tuple[int, int]]]:
    positions = np.asarray(batch["marker_pos"], dtype=np.int64)
    masks = np.asarray(batch["marker_mask"], dtype=bool)
    markers: list[int] = []
    locations: list[tuple[int, int]] = []
    for row in range(positions.shape[0]):
        for column, (position, valid) in enumerate(zip(positions[row], masks[row])):
            if valid:
                markers.append(int(row * seq + position))
                locations.append((row, column))
    return markers, locations


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("onnx", type=Path, help="exported Laya ONNX directory")
    parser.add_argument("golden", type=Path, help="official case_dict_state.json")
    parser.add_argument("--tolerance", type=float, default=1e-4)
    args = parser.parse_args()

    with args.golden.open(encoding="utf-8") as handle:
        case = json.load(handle)
    batch = case["batch"]
    input_ids = np.asarray(batch["input_ids"], dtype=np.int64)
    attention_mask = np.asarray(batch["attention_mask"], dtype=np.int64)
    type_ids = np.asarray(batch["qtype"], dtype=np.int64)
    if input_ids.ndim != 2:
        raise ValueError(f"expected input_ids [batch, seq], got {input_ids.shape}")
    markers, locations = flattened_markers(batch, int(input_ids.shape[1]))

    session = ort.InferenceSession(
        str(args.onnx / "model.onnx"), providers=["CPUExecutionProvider"]
    )
    output = session.run(
        ["logits"],
        {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "type_ids": type_ids,
            "markers": np.asarray(markers, dtype=np.int64),
        },
    )[0]
    expected = np.asarray(case["logits"], dtype=np.float32)
    expected_values = np.asarray(
        [expected[row][column] for row, column in locations], dtype=np.float32
    )

    difference = np.abs(output - expected_values)
    worst = float(difference.max()) if difference.size else 0.0
    print(f"markers: {len(markers)}")
    print(f"max |onnx - golden|: {worst:.9g}")
    if worst > args.tolerance:
        worst_index = int(difference.argmax())
        print(
            f"mismatch at marker {worst_index}: "
            f"onnx={output[worst_index]:.9g} golden={expected_values[worst_index]:.9g}",
            file=sys.stderr,
        )
        return 1
    print("ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
