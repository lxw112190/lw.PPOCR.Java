#!/usr/bin/env python3
"""Fail CI when focused performance or allocation results regress."""

from __future__ import annotations

import json
import math
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


def check_low_cpu_allocation(result: dict, name: str) -> None:
    for key, expected in {"available_processors": 2, "lines": 16,
                          "detector_limit_side": 320, "warmup": 10, "iterations": 5}.items():
        if number(result, key, name) != expected:
            raise SystemExit(f"{name}: expected {key}={expected}")
    if result.get("backend") != "vector" or result.get("parallelism_policy") != "auto":
        raise SystemExit(f"{name}: expected Vector AUTO pipeline")
    for key in ("allocated_bytes_per_ocr", "gc_count_delta"):
        if number(result, key, name) < 0:
            raise SystemExit(f"{name}: unsupported measurement {key}")
    require_le(result, "allocated_bytes_per_ocr", 1_000_000.0, name)
    require_le(result, "gc_count_delta", 1.0, name)


def check_stem_contract(result: dict, name: str) -> None:
    # Keep the same fixture and 5 ms gate, but don't score a ten-call JIT startup.
    for key, expected in {"input_width": 320, "input_height": 320,
                          "input_channels": 3, "output_width": 160,
                          "output_height": 160, "output_channels": 16}.items():
        if number(result, key, name) != expected:
            raise SystemExit(f"{name}: expected {key}={expected}")
    for key, expected in {"benchmark": "det-stride-two-conv", "backend": "vector",
                          "workload": "stem", "checksum": "10973022362502076197"}.items():
        if result.get(key) != expected:
            raise SystemExit(f"{name}: expected {key}={expected}")
    for key, minimum in {"requested_warmup": 100, "warmup": 100,
                         "warmup_min_ms": 1000, "iterations": 30}.items():
        value = number(result, key, name)
        if not math.isfinite(value) or not value.is_integer() or value < minimum:
            raise SystemExit(f"{name}: expected integer {key}>={minimum}")
    if number(result, "warmup", name) < number(result, "requested_warmup", name):
        raise SystemExit(f"{name}: incomplete warmup count")
    elapsed = number(result, "warmup_elapsed_ms", name)
    if not math.isfinite(elapsed) or elapsed < number(result, "warmup_min_ms", name):
        raise SystemExit(f"{name}: incomplete elapsed warmup")
    samples = result.get("samples_ms")
    if not isinstance(samples, list) or len(samples) != result["iterations"]:
        raise SystemExit(f"{name}: missing chronological timing samples")
    if any(isinstance(x, bool) or not isinstance(x, (float, int))
           or not math.isfinite(x) or x <= 0 for x in samples):
        raise SystemExit(f"{name}: invalid timing sample")
    # Java reports the upper median for even sample counts. Never drop slow samples.
    median = number(result, "median_ms", name)
    if not math.isfinite(median) or abs(median - sorted(samples)[len(samples)//2]) > .001:
        raise SystemExit(f"{name}: reported median does not match all samples")


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
        result = load_json(filename)
        if filename == "det-stem-stride-two-conv.json":
            check_stem_contract(result, filename)
        require_lt(result, "median_ms", limit, filename)

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

    for replica in (1, 2):
        filename = f"full-ocr-allocation-vector-cpu2-r{replica}.json"
        check_low_cpu_allocation(load_json(filename), filename)

    print("performance regression checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
