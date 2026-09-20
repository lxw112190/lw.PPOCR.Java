#!/usr/bin/env python3
"""Create a compact Java OCR speed and process-memory comparison report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_last_json(path: Path) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            records.append(value)
    if not records:
        raise SystemExit(f"no JSON object found in {path}")
    return records[-1]


def mib(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
        return None
    return value / 1024.0 / 1024.0


def signed_mib(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value == -1:
        return None
    return value / 1024.0 / 1024.0


def fmt_mb(value: Any) -> str:
    converted = mib(value)
    return "n/a" if converted is None else f"{converted:.1f} MB"


def fmt_signed_mb(value: Any) -> str:
    converted = signed_mib(value)
    return "n/a" if converted is None else f"{converted:.1f} MB"


def fmt_bytes(value: Any) -> str:
    return f"{value:,} B" if isinstance(value, (int, float)) else "n/a"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--json-out", type=Path, required=True)
    parser.add_argument("--markdown-out", type=Path, required=True)
    args = parser.parse_args()

    source = load_last_json(args.input)
    mean_ms = float(source["mean_ms"])
    median_ms = float(source["median_ms"])
    p95_ms = float(source["p95_ms"])
    images_per_second = 1000.0 / mean_ms if mean_ms > 0 else 0.0
    lines = int(source.get("lines", 0))
    lines_per_second = images_per_second * lines

    result = {
        "schema": 1,
        "engine": "java",
        "model": "ppocrv6-tiny",
        "benchmark": source.get("benchmark"),
        "backend": source.get("backend"),
        "vector_bits": source.get("vector_bits"),
        "pointwise_block": source.get("pointwise_block"),
        "parallelism_policy": source.get("parallelism_policy"),
        "available_processors": source.get("available_processors"),
        "lines_per_image": lines,
        "warmup": source.get("warmup"),
        "iterations": source.get("iterations"),
        "total_ocr_calls": source.get("total_ocr_calls"),
        "mean_ms": mean_ms,
        "median_ms": median_ms,
        "p95_ms": p95_ms,
        "images_per_second": images_per_second,
        "lines_per_second": lines_per_second,
        "detection_mean_ms": source.get("detection_mean_ms"),
        "classification_mean_ms": source.get("classification_mean_ms"),
        "recognition_mean_ms": source.get("recognition_mean_ms"),
        "working_set_mb_loaded": mib(source.get("process_rss_loaded_bytes")),
        "working_set_mb_last": mib(source.get("process_rss_last_bytes")),
        "working_set_mb_peak": mib(source.get("process_rss_sampled_peak_bytes")),
        "working_set_mb_delta": signed_mib(source.get("process_rss_delta_bytes")),
        "working_set_mb_after_gc": mib(source.get("process_rss_after_gc_bytes")),
        "process_hwm_mb": mib(source.get("process_rss_hwm_bytes")),
        "retained_heap_mb": mib(source.get("retained_heap_delta_bytes")),
        "peak_heap_delta_mb": mib(source.get("peak_heap_delta_bytes")),
        "workspace_mb": mib(source.get("workspace_bytes")),
        "packed_weight_mb": mib(source.get("packed_weight_bytes")),
        "allocation_bytes_per_ocr": source.get("allocated_bytes_per_ocr"),
        "gc_count": source.get("gc_count_delta"),
        "gc_time_ms": source.get("gc_time_ms_delta"),
    }

    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    markdown = [
        "## Java PP-OCR Comparison Benchmark",
        "",
        "### Performance",
        "",
        "| Metric | Java |",
        "| --- | ---: |",
        f"| Mean | {mean_ms:.2f} ms |",
        f"| Median | {median_ms:.2f} ms |",
        f"| P95 | {p95_ms:.2f} ms |",
        f"| Throughput | {images_per_second:.2f} img/s |",
        f"| Lines throughput | {lines_per_second:.1f} lines/s |",
        f"| DET | {float(source['detection_mean_ms']):.2f} ms |",
        f"| CLS | {float(source['classification_mean_ms']):.2f} ms |",
        f"| REC | {float(source['recognition_mean_ms']):.2f} ms |",
        "",
        "### Process memory",
        "",
        "| Metric | Java |",
        "| --- | ---: |",
        f"| RSS loaded | {fmt_mb(source.get('process_rss_loaded_bytes'))} |",
        f"| RSS last | {fmt_mb(source.get('process_rss_last_bytes'))} |",
        f"| RSS sampled peak | {fmt_mb(source.get('process_rss_sampled_peak_bytes'))} |",
        f"| RSS delta | {fmt_signed_mb(source.get('process_rss_delta_bytes'))} |",
        f"| RSS after GC | {fmt_mb(source.get('process_rss_after_gc_bytes'))} |",
        f"| Process VmHWM | {fmt_mb(source.get('process_rss_hwm_bytes'))} |",
        "",
        "### Java runtime memory",
        "",
        "| Metric | Java |",
        "| --- | ---: |",
        f"| Retained heap delta | {fmt_mb(source.get('retained_heap_delta_bytes'))} |",
        f"| Peak heap delta | {fmt_mb(source.get('peak_heap_delta_bytes'))} |",
        f"| Workspace | {fmt_mb(source.get('workspace_bytes'))} |",
        f"| Packed weights | {fmt_mb(source.get('packed_weight_bytes'))} |",
        f"| Allocation / OCR | {fmt_bytes(source.get('allocated_bytes_per_ocr'))} |",
        f"| GC count | {source.get('gc_count_delta', 'n/a')} |",
        f"| GC time | {source.get('gc_time_ms_delta', 'n/a')} ms |",
        "",
        "### Benchmark contract",
        "",
        f"- Backend: `{source.get('backend')}`",
        f"- Vector bits: `{source.get('vector_bits')}`",
        f"- Parallelism: `{source.get('parallelism_policy')}`",
        f"- CPU visible to JVM: `{source.get('available_processors')}`",
        f"- Lines/image: `{lines}`",
        f"- Warmup: `{source.get('warmup')}`",
        f"- Timed iterations: `{source.get('iterations')}`",
        f"- Total OCR calls: `{source.get('total_ocr_calls')}`",
        f"- Detector limit: `{source.get('detector_limit_side')}`",
        f"- Image: `{source.get('image_width')}x{source.get('image_height')}`",
        "- Input image is repeated for the timed calls; this is not a dataset CER benchmark.",
    ]
    args.markdown_out.write_text("\n".join(markdown) + "\n", encoding="utf-8")
    print("\n".join(markdown))


if __name__ == "__main__":
    main()
