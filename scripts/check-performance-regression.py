#!/usr/bin/env python3
"""Fail CI when focused performance or allocation results regress."""

from __future__ import annotations

import json
import os
import pathlib
import sys


ROOT = pathlib.Path(os.environ.get("LW_PPOCR_BENCHMARK_RESULTS", "benchmark-results"))


def load_json(name: str) -> dict:
    path = ROOT / name
    if not path.is_file():
        raise SystemExit(f"missing benchmark result: {path}")
    for line in reversed(path.read_text(encoding="utf-8").splitlines()):
        line = line.strip()
        if not line:
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    raise SystemExit(f"no JSON object found in {path}")


def number(result: dict, key: str, name: str) -> float:
    value = result.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SystemExit(f"{name}: missing numeric field {key!r}")
    return float(value)


def require_lt(result: dict, key: str, limit: float, name: str) -> None:
    value = number(result, key, name)
    print(f"{name}: {key}={value:g}, required < {limit:g}")
    if not value < limit:
        raise SystemExit(f"performance regression: {name} {key}={value:g} >= {limit:g}")


def require_le(result: dict, key: str, limit: float, name: str) -> None:
    value = number(result, key, name)
    print(f"{name}: {key}={value:g}, required <= {limit:g}")
    if not value <= limit:
        raise SystemExit(f"performance regression: {name} {key}={value:g} > {limit:g}")


def main() -> int:
    focused = {
        "rec-projection-matmul.json": 5.0,
        "rec-stride-two-conv.json": 10.0,
        "det-stem-stride-two-conv.json": 5.0,
        "det-downsample-stride-two-conv.json": 12.0,
        "det-downsample960-stride-two-conv.json": 35.0,
        "det-stride-one-conv.json": 15.0,
        "det-stride-one-conv-det960.json": 60.0,
    }
    for filename, limit in focused.items():
        require_lt(load_json(filename), "median_ms", limit, filename)

    full = load_json("full-ocr-allocation-vector.json")
    require_lt(full, "mean_ms", 400.0, "full-ocr-allocation-vector.json")
    require_le(full, "allocated_bytes_per_ocr", 1_000_000.0, "full-ocr-allocation-vector.json")
    require_le(full, "gc_count_delta", 1.0, "full-ocr-allocation-vector.json")
    require_le(full, "packed_weight_bytes", 2_300_000.0, "full-ocr-allocation-vector.json")
    efficiency = number(full, "workspace_efficiency", "full-ocr-allocation-vector.json")
    print(f"full-ocr-allocation-vector.json: workspace_efficiency={efficiency:g}, required >= 0.95")
    if efficiency < 0.95:
        raise SystemExit(
            "performance regression: full-ocr-allocation-vector.json "
            f"workspace_efficiency={efficiency:g} < 0.95"
        )

    print("performance regression checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
