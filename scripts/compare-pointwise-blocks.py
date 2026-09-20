#!/usr/bin/env python3
"""Compare fixed pointwise output blocks without changing the production default."""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from typing import Any


ROOT = pathlib.Path("benchmark-results")
BLOCKS = (4, 8, 12)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--bits", type=int, choices=(256, 512), required=True)
    return result


def read_json(path: pathlib.Path) -> dict[str, Any]:
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


def number(value: Any, key: str, path: pathlib.Path) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SystemExit(f"{path}: missing numeric field {key!r}")
    return float(value)


def text(value: Any, key: str, path: pathlib.Path) -> str:
    if not isinstance(value, str):
        raise SystemExit(f"{path}: missing string field {key!r}")
    return value


def load_hot(bits: int, block: int) -> list[dict[str, Any]]:
    path = ROOT / f"rec-pointwise-vector{bits}-block{block}.jsonl"
    if not path.is_file():
        raise SystemExit(f"missing benchmark result: {path}")
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            rows.append(value)
    if not rows:
        raise SystemExit(f"no JSON objects found in {path}")
    return rows


def check_common(rows: list[dict[str, Any]], bits: int, block: int, name: str) -> None:
    for row in rows:
        path_name = f"{name}/block{block}"
        if text(row.get("backend"), "backend", pathlib.Path(path_name)) != "vector":
            raise SystemExit(f"{path_name}: expected vector backend")
        if text(row.get("vector_bits"), "vector_bits", pathlib.Path(path_name)) != str(bits):
            raise SystemExit(f"{path_name}: expected vector_bits={bits}")
        if text(row.get("pointwise_block"), "pointwise_block", pathlib.Path(path_name)) != str(block):
            raise SystemExit(f"{path_name}: expected pointwise_block={block}")


def compare(bits: int) -> dict[str, Any]:
    hot: dict[int, list[dict[str, Any]]] = {block: load_hot(bits, block) for block in BLOCKS}
    for block, rows in hot.items():
        check_common(rows, bits, block, "hot")

    shape_keys = lambda row: (
        row.get("operator"), row.get("input_channels"), row.get("output_channels"),
        row.get("height"), row.get("width"), row.get("output_height"),
        row.get("output_width"), row.get("kernel"), row.get("stride"),
    )
    by_shape: dict[int, dict[tuple[Any, ...], dict[str, Any]]] = {}
    for block, rows in hot.items():
        by_shape[block] = {shape_keys(row): row for row in rows}
    common_shapes = set.intersection(*(set(value) for value in by_shape.values()))
    if not common_shapes:
        raise SystemExit("pointwise block matrix has no common hot shapes")

    hot_summary: dict[str, Any] = {}
    for shape in sorted(common_shapes, key=str):
        checksums = {str(block): by_shape[block][shape].get("checksum") for block in BLOCKS}
        if len(set(checksums.values())) != 1:
            raise SystemExit(f"checksum mismatch for hot shape {shape}: {checksums}")
        hot_summary["|".join(map(str, shape))] = {
            "checksum": checksums["4"],
            "median_ms": {str(block): number(by_shape[block][shape].get("median_ms"), "median_ms", pathlib.Path("hot"))
                          for block in BLOCKS},
            "allocated_bytes_per_op": {
                str(block): number(by_shape[block][shape].get("allocated_bytes_per_op"),
                                   "allocated_bytes_per_op", pathlib.Path("hot"))
                for block in BLOCKS
            },
        }

    full: dict[int, dict[str, Any]] = {}
    for block in BLOCKS:
        path = ROOT / f"full-ocr-vector{bits}-block{block}.json"
        row = read_json(path)
        check_common([row], bits, block, "full")
        full[block] = row

    full_summary = {
        str(block): {
            key: full[block].get(key)
            for key in ("mean_ms", "median_ms", "p95_ms", "recognition_mean_ms", "lines",
                        "allocated_bytes_per_ocr", "gc_count_delta")
        }
        for block in BLOCKS
    }
    hot_alloc = {
        str(block): max(
            number(row.get("allocated_bytes_per_op"), "allocated_bytes_per_op", pathlib.Path("hot"))
            for row in by_shape[block].values()
        )
        for block in BLOCKS
    }
    full_alloc = {
        str(block): number(full[block].get("allocated_bytes_per_ocr"),
                           "allocated_bytes_per_ocr", pathlib.Path("full"))
        for block in BLOCKS
    }
    full_gc = {
        str(block): number(full[block].get("gc_count_delta"), "gc_count_delta", pathlib.Path("full"))
        for block in BLOCKS
    }
    full_median = {str(block): number(full[block].get("median_ms"), "median_ms", pathlib.Path("full"))
                   for block in BLOCKS}
    eligible = {
        str(block): 0 <= hot_alloc[str(block)] <= 64
        and 0 <= full_alloc[str(block)] <= 20_000
        and full_gc[str(block)] == 0
        for block in BLOCKS
    }
    eligible_blocks = [block for block in BLOCKS if eligible[str(block)]]
    winner = min(eligible_blocks, key=lambda block: full_median[str(block)]) if eligible_blocks else 12
    baseline = full_median["12"]
    improvement = 0.0 if baseline <= 0 else (baseline - full_median[str(winner)]) / baseline
    recommendation = f"use block {winner}" if winner != 12 and improvement >= 0.03 else "keep block 12"
    result = {
        "benchmark": "pointwise-block-comparison",
        "vector_bits": bits,
        "blocks": list(BLOCKS),
        "common_hot_shapes": len(common_shapes),
        "hot": hot_summary,
        "full_ocr": full_summary,
        "max_hot_allocated_bytes_per_op": hot_alloc,
        "full_allocated_bytes_per_ocr": full_alloc,
        "full_gc_count_delta": full_gc,
        "eligible": eligible,
        "winner_by_full_ocr_median": winner,
        "improvement_vs_block12": improvement,
        "recommendation": recommendation,
        "default_changed": False,
    }
    return result


def main() -> int:
    args = parser().parse_args()
    result = compare(args.bits)
    output = ROOT / f"pointwise-block-summary-{args.bits}.json"
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    sys.exit(main())
