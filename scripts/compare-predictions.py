#!/usr/bin/env python3
"""Compare two DatasetOcrPerformanceMain prediction JSONL files.

The comparison is deliberately stricter than a performance comparison:
prediction files must contain the same images and the same number of lines,
texts, class labels, rotation decisions, and nearly identical boxes.  A
non-zero exit status makes the helper suitable for an optimization gate.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def load_predictions(path: Path) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for line_number, raw_line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw_line.strip():
            continue
        try:
            record = json.loads(raw_line)
        except json.JSONDecodeError as error:
            raise ValueError(f"{path}:{line_number}: invalid JSON: {error}") from error
        if not isinstance(record, dict) or not isinstance(record.get("file"), str):
            raise ValueError(f"{path}:{line_number}: record must contain a string file")
        file = record["file"]
        if file in records:
            raise ValueError(f"{path}:{line_number}: duplicate file {file!r}")
        lines = record.get("lines")
        if not isinstance(lines, list):
            raise ValueError(f"{path}:{line_number}: lines must be an array")
        records[file] = record
    return records


def number(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"line field {field!r} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"line field {field!r} must be finite")
    return result


def box_values(line: dict[str, Any]) -> list[float]:
    values = line.get("box")
    if not isinstance(values, list) or len(values) != 8:
        raise ValueError("line box must contain eight coordinates")
    return [number(value, "box") for value in values]


def compare_line(baseline: dict[str, Any], candidate: dict[str, Any],
                 box_tolerance: float) -> dict[str, Any]:
    baseline_box = box_values(baseline)
    candidate_box = box_values(candidate)
    box_delta = max(abs(left - right)
                    for left, right in zip(baseline_box, candidate_box))
    baseline_text = baseline.get("text")
    candidate_text = candidate.get("text")
    if not isinstance(baseline_text, str) or not isinstance(candidate_text, str):
        raise ValueError("line text must be a string")

    baseline_cls = baseline.get("cls", -1)
    candidate_cls = candidate.get("cls", -1)
    return {
        "text_equal": baseline_text == candidate_text,
        "cls_equal": baseline_cls == candidate_cls,
        "rotation_equal": baseline.get("rotate", False) == candidate.get("rotate", False),
        "box_equal": box_delta <= box_tolerance,
        "box_delta": box_delta,
        "baseline_text": baseline_text,
        "candidate_text": candidate_text,
        "baseline_cls": baseline_cls,
        "candidate_cls": candidate_cls,
    }


def compare(baseline: dict[str, dict[str, Any]], candidate: dict[str, dict[str, Any]],
            box_tolerance: float) -> dict[str, Any]:
    baseline_files = set(baseline)
    candidate_files = set(candidate)
    missing_files = sorted(baseline_files - candidate_files)
    extra_files = sorted(candidate_files - baseline_files)
    details = []
    counters = {
        "line_count_mismatches": 0,
        "text_mismatches": 0,
        "class_mismatches": 0,
        "rotation_mismatches": 0,
        "box_mismatches": 0,
    }
    max_box_delta = 0.0

    for file in sorted(baseline_files & candidate_files):
        baseline_lines = baseline[file]["lines"]
        candidate_lines = candidate[file]["lines"]
        if len(baseline_lines) != len(candidate_lines):
            counters["line_count_mismatches"] += 1
        mismatches = []
        for index, (baseline_line, candidate_line) in enumerate(
                zip(baseline_lines, candidate_lines)):
            if not isinstance(baseline_line, dict) or not isinstance(candidate_line, dict):
                raise ValueError(f"{file}: line {index} must be an object")
            result = compare_line(baseline_line, candidate_line, box_tolerance)
            max_box_delta = max(max_box_delta, result["box_delta"])
            for key, counter in (
                    ("text_equal", "text_mismatches"),
                    ("cls_equal", "class_mismatches"),
                    ("rotation_equal", "rotation_mismatches"),
                    ("box_equal", "box_mismatches")):
                if not result[key]:
                    counters[counter] += 1
            if not all(result[key] for key in
                       ("text_equal", "cls_equal", "rotation_equal", "box_equal")):
                mismatches.append({
                    "line": index,
                    "text_equal": result["text_equal"],
                    "cls_equal": result["cls_equal"],
                    "rotation_equal": result["rotation_equal"],
                    "box_equal": result["box_equal"],
                    "box_delta": result["box_delta"],
                    "baseline_text": result["baseline_text"],
                    "candidate_text": result["candidate_text"],
                })
        if mismatches or len(baseline_lines) != len(candidate_lines):
            details.append({
                "file": file,
                "baseline_lines": len(baseline_lines),
                "candidate_lines": len(candidate_lines),
                "mismatches": mismatches,
            })

    status = "ok" if not (
        missing_files or extra_files or any(counters.values())
    ) else "mismatch"
    return {
        "status": status,
        "files": len(baseline_files | candidate_files),
        "baseline_files": len(baseline_files),
        "candidate_files": len(candidate_files),
        "missing_files": missing_files,
        "extra_files": extra_files,
        "box_tolerance": box_tolerance,
        "max_box_delta": max_box_delta,
        **counters,
        "changed_images": details,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="compare DatasetOcrPerformanceMain prediction JSONL files")
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--box-tolerance", type=float, default=1.0,
                        help="maximum absolute coordinate delta (default: 1.0)")
    parser.add_argument("--output", type=Path,
                        help="write the detailed JSON report to this path")
    args = parser.parse_args()
    if args.box_tolerance < 0.0 or not math.isfinite(args.box_tolerance):
        parser.error("--box-tolerance must be a finite non-negative number")

    report = compare(load_predictions(args.baseline), load_predictions(args.candidate),
                     args.box_tolerance)
    rendered = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8", newline="\n")
    print(json.dumps({key: value for key, value in report.items()
                      if key != "changed_images"}, ensure_ascii=False))
    return 0 if report["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
