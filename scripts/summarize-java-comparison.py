#!/usr/bin/env python3
"""Create Java cross-runtime comparison and steady-state reports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


CONTRACT_FIELDS = (
    "backend", "vector_bits", "pointwise_block", "parallelism_policy",
    "available_processors", "detector_limit_side", "image_width", "image_height", "lines",
)


def load_last_json(path: Path) -> dict[str, Any]:
    result: dict[str, Any] | None = None
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            result = value
    if result is None:
        raise SystemExit(f"no JSON object found in {path}")
    return result


def mib(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
        return None
    return float(value) / 1024.0 / 1024.0


def signed_mib(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value == -1:
        return None
    return float(value) / 1024.0 / 1024.0


def fmt_mib(value: Any) -> str:
    converted = mib(value)
    return "n/a" if converted is None else f"{converted:.1f} MB"


def fmt_signed_mib(value: Any) -> str:
    converted = signed_mib(value)
    return "n/a" if converted is None else f"{converted:+.1f} MB"


def fmt_bytes(value: Any) -> str:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return "n/a"
    return f"{int(value):,} B"


def throughput(source: dict[str, Any]) -> float:
    mean_ms = float(source["mean_ms"])
    return 0.0 if mean_ms <= 0 else 1000.0 / mean_ms


def pct(current: float, baseline: float) -> float:
    return 0.0 if baseline == 0 else (current / baseline - 1.0) * 100.0


def validate_contract(comparable: dict[str, Any], steady: dict[str, Any]) -> dict[str, Any]:
    for field in CONTRACT_FIELDS:
        left = comparable.get(field)
        right = steady.get(field)
        if left != right:
            raise SystemExit(
                f"comparison contract mismatch: {field}: "
                f"comparable={left!r}, steady={right!r}"
            )
    for label, source in (("comparable", comparable), ("steady", steady)):
        if source.get("process_memory_source") != "linux-procfs":
            raise SystemExit(f"{label} run has no Linux RSS probe")
        required = (
            "mean_ms", "median_ms", "p95_ms", "warmup", "iterations", "total_ocr_calls",
            "process_rss_timed_start_bytes", "process_rss_timed_peak_bytes",
            "process_rss_timed_delta_bytes", "process_rss_timed_peak_delta_bytes",
        )
        missing = [field for field in required if field not in source]
        if missing:
            raise SystemExit(f"{label} run is missing fields: {', '.join(missing)}")
    contract = {field: comparable.get(field) for field in CONTRACT_FIELDS}
    contract["detector_limit"] = comparable.get("detector_limit_side")
    contract["lines_per_image"] = comparable.get("lines")
    return contract


def performance_json(source: dict[str, Any]) -> dict[str, Any]:
    ips = throughput(source)
    lines = int(source.get("lines", 0))
    return {
        "mean_ms": float(source["mean_ms"]),
        "median_ms": float(source["median_ms"]),
        "p95_ms": float(source["p95_ms"]),
        "images_per_second": ips,
        "lines_per_second": ips * lines,
        "detection_mean_ms": float(source["detection_mean_ms"]),
        "classification_mean_ms": float(source["classification_mean_ms"]),
        "recognition_mean_ms": float(source["recognition_mean_ms"]),
    }


def comparable_memory_json(source: dict[str, Any]) -> dict[str, Any]:
    return {
        "rss_loaded_mb": mib(source.get("process_rss_loaded_bytes")),
        "rss_last_mb": mib(source.get("process_rss_last_bytes")),
        "rss_peak_mb": mib(source.get("process_rss_sampled_peak_bytes")),
        "rss_delta_mb": signed_mib(source.get("process_rss_delta_bytes")),
        "rss_peak_delta_mb": signed_mib(source.get("process_rss_peak_delta_bytes")),
        "rss_after_gc_mb": mib(source.get("process_rss_after_gc_bytes")),
        "process_hwm_mb": mib(source.get("process_rss_hwm_bytes")),
    }


def steady_memory_json(source: dict[str, Any]) -> dict[str, Any]:
    return {
        "rss_warmup_peak_mb": mib(source.get("process_rss_warmup_peak_bytes")),
        "rss_timed_start_mb": mib(source.get("process_rss_timed_start_bytes")),
        "rss_timed_peak_mb": mib(source.get("process_rss_timed_peak_bytes")),
        "rss_timed_last_mb": mib(source.get("process_rss_last_bytes")),
        "rss_timed_delta_mb": signed_mib(source.get("process_rss_timed_delta_bytes")),
        "rss_timed_peak_delta_mb": signed_mib(source.get("process_rss_timed_peak_delta_bytes")),
        "rss_after_gc_mb": mib(source.get("process_rss_after_gc_bytes")),
        "process_hwm_mb": mib(source.get("process_rss_hwm_bytes")),
        "retained_heap_mb": mib(source.get("retained_heap_delta_bytes")),
        "peak_heap_delta_mb": mib(source.get("peak_heap_delta_bytes")),
        "workspace_mb": mib(source.get("workspace_bytes")),
        "packed_weight_mb": mib(source.get("packed_weight_bytes")),
        "gc_count": source.get("gc_count_delta"),
        "gc_time_ms": source.get("gc_time_ms_delta"),
    }


def run_json(source: dict[str, Any], mode: str) -> dict[str, Any]:
    result: dict[str, Any] = {
        "mode": mode,
        "warmup": source.get("warmup"),
        "iterations": source.get("iterations"),
        "total_ocr_calls": source.get("total_ocr_calls"),
        "performance": performance_json(source),
    }
    if mode == "cold-to-steady":
        result["process_memory"] = comparable_memory_json(source)
        result["allocation_bytes_per_ocr"] = source.get("allocated_bytes_per_ocr")
        result["allocation_semantics"] = "ramp-up-sensitive"
        result["gc_count"] = source.get("gc_count_delta")
        result["gc_time_ms"] = source.get("gc_time_ms_delta")
    else:
        result["memory"] = steady_memory_json(source)
        result["allocation_bytes_per_ocr"] = source.get("allocated_bytes_per_ocr")
        result["allocation_semantics"] = "post-warmup"
    return result


def fmt_value(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.1f} MB"


def fmt_signed_value(value: float | None) -> str:
    return "n/a" if value is None else f"{value:+.1f} MB"


def performance_rows(item: dict[str, Any], include_lines: bool = True) -> list[str]:
    perf = item["performance"]
    rows = [
        f"| Mean | {perf['mean_ms']:.2f} ms |",
        f"| Median | {perf['median_ms']:.2f} ms |",
        f"| P95 | {perf['p95_ms']:.2f} ms |",
        f"| Throughput | {perf['images_per_second']:.2f} img/s |",
    ]
    if include_lines:
        rows.append(f"| Lines throughput | {perf['lines_per_second']:.1f} lines/s |")
    rows.extend([
        f"| DET | {perf['detection_mean_ms']:.2f} ms |",
        f"| CLS | {perf['classification_mean_ms']:.2f} ms |",
        f"| REC | {perf['recognition_mean_ms']:.2f} ms |",
    ])
    return rows


def markdown(comparable: dict[str, Any], steady: dict[str, Any], contract: dict[str, Any]) -> str:
    cp = comparable["performance"]
    sp = steady["performance"]
    cm = comparable["process_memory"]
    sm = steady["memory"]
    lines = [
        "## Java PP-OCR Comparison Benchmark", "",
        "> Two independent JVM runs over the same committed Tiny image; this is a runtime benchmark, not a dataset CER benchmark.", "",
        "### Comparable run — cold-to-steady", "",
        "> Cross-runtime profile: 1 warmup OCR + 99 timed OCR calls. Allocation is ramp-up-sensitive and is not treated as steady allocation.", "",
        "| Metric | Java |", "| --- | ---: |", *performance_rows(comparable), "",
        "### Comparable process memory", "",
        "| Metric | Java |", "| --- | ---: |",
        f"| RSS loaded | {fmt_value(cm['rss_loaded_mb'])} |",
        f"| RSS last | {fmt_value(cm['rss_last_mb'])} |",
        f"| RSS full sampled peak | {fmt_value(cm['rss_peak_mb'])} |",
        f"| RSS delta | {fmt_signed_value(cm['rss_delta_mb'])} |",
        f"| RSS peak delta | {fmt_signed_value(cm['rss_peak_delta_mb'])} |",
        f"| RSS after GC | {fmt_value(cm['rss_after_gc_mb'])} |",
        f"| Process VmHWM | {fmt_value(cm['process_hwm_mb'])} |",
        f"| Comparable allocation / OCR (ramp-up-sensitive) | {fmt_bytes(comparable['allocation_bytes_per_ocr'])} |",
        f"| Comparable GC count | {comparable.get('gc_count', 'n/a')} |", "",
        "### Steady run", "",
        "> Steady-state profile: 10 warmup OCR calls + 20 timed OCR calls in a fresh JVM.", "",
        "| Metric | Java |", "| --- | ---: |", *performance_rows(steady), "",
        "### Steady process memory", "",
        "| Metric | Java |", "| --- | ---: |",
        f"| RSS warmup peak | {fmt_value(sm['rss_warmup_peak_mb'])} |",
        f"| RSS timed start | {fmt_value(sm['rss_timed_start_mb'])} |",
        f"| RSS timed peak | {fmt_value(sm['rss_timed_peak_mb'])} |",
        f"| RSS timed last | {fmt_value(sm['rss_timed_last_mb'])} |",
        f"| RSS timed delta | {fmt_signed_value(sm['rss_timed_delta_mb'])} |",
        f"| RSS timed peak delta | {fmt_signed_value(sm['rss_timed_peak_delta_mb'])} |",
        f"| RSS after GC | {fmt_value(sm['rss_after_gc_mb'])} |", "",
        "### Java steady runtime memory", "", "| Metric | Java |", "| --- | ---: |",
        f"| Retained heap delta | {fmt_value(sm['retained_heap_mb'])} |",
        f"| Peak heap delta | {fmt_value(sm['peak_heap_delta_mb'])} |",
        f"| Workspace | {fmt_value(sm['workspace_mb'])} |",
        f"| Packed weights | {fmt_value(sm['packed_weight_mb'])} |",
        f"| Allocation / OCR (post-warmup) | {fmt_bytes(steady['allocation_bytes_per_ocr'])} |",
        f"| GC count | {sm['gc_count'] if sm['gc_count'] is not None else 'n/a'} |",
        f"| GC time | {sm['gc_time_ms'] if sm['gc_time_ms'] is not None else 'n/a'} ms |", "",
        "### Comparable versus steady", "", "| Metric | Comparable | Steady | Change |", "| --- | ---: | ---: | ---: |",
        f"| Median | {cp['median_ms']:.2f} ms | {sp['median_ms']:.2f} ms | {pct(sp['median_ms'], cp['median_ms']):+.2f}% |",
        f"| P95 | {cp['p95_ms']:.2f} ms | {sp['p95_ms']:.2f} ms | {pct(sp['p95_ms'], cp['p95_ms']):+.2f}% |",
        "", "### Benchmark contract", "",
        *[f"- {key}: `{value}`" for key, value in contract.items()],
        "- Comparable: `warmup=1`, `timed=99`; steady: `warmup=10`, `timed=20`.",
        "- Process RSS source: `linux-procfs`.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--comparable", type=Path, required=True)
    parser.add_argument("--steady", type=Path, required=True)
    parser.add_argument("--json-out", type=Path, required=True)
    parser.add_argument("--markdown-out", type=Path, required=True)
    args = parser.parse_args()
    comparable_source = load_last_json(args.comparable)
    steady_source = load_last_json(args.steady)
    contract = validate_contract(comparable_source, steady_source)
    comparable = run_json(comparable_source, "cold-to-steady")
    steady = run_json(steady_source, "steady")
    result = {
        "schema": 2,
        "engine": "java",
        "model": "ppocrv6-tiny",
        "contract": contract,
        "comparable": comparable,
        "steady": steady,
        "comparison": {
            "median_ms_delta_percent": pct(
                steady["performance"]["median_ms"], comparable["performance"]["median_ms"]
            ),
            "p95_ms_delta_percent": pct(
                steady["performance"]["p95_ms"], comparable["performance"]["p95_ms"]
            ),
        },
    }
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    rendered = markdown(comparable, steady, contract)
    args.markdown_out.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
