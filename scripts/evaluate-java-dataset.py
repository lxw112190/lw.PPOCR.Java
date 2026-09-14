#!/usr/bin/env python3
"""Score DatasetOcrPerformanceMain JSONL against a lw.PPOCR.C dataset manifest.

The image directory and metadata remain external. This helper intentionally
uses the same greedy IoU matching and matched-line CER definition as the C
project's evaluate_ocr_dataset.py script.
"""

from __future__ import annotations

import argparse
import json
import unicodedata
from pathlib import Path


def normalize(value: str) -> str:
    return unicodedata.normalize("NFC", value.replace("\r\n", "\n"))


def edit_distance(expected: str, actual: str) -> int:
    previous = list(range(len(actual) + 1))
    for row, expected_char in enumerate(expected, 1):
        current = [row]
        for column, actual_char in enumerate(actual, 1):
            current.append(min(
                current[-1] + 1,
                previous[column] + 1,
                previous[column - 1] + (expected_char != actual_char),
            ))
        previous = current
    return previous[-1]


def bbox(values: list[float]) -> tuple[float, float, float, float]:
    if len(values) != 8:
        raise ValueError("Java prediction box must contain eight coordinates")
    xs = values[0::2]
    ys = values[1::2]
    return min(xs), min(ys), max(xs), max(ys)


def iou(first: tuple[float, float, float, float], second: tuple[float, float, float, float]) -> float:
    ix1, iy1 = max(first[0], second[0]), max(first[1], second[1])
    ix2, iy2 = min(first[2], second[2]), min(first[3], second[3])
    intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    first_area = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    second_area = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
    union = first_area + second_area - intersection
    return intersection / union if union > 0.0 else 0.0


def match(ground_truth: list[tuple[float, float, float, float]], predicted: list[dict], threshold: float):
    candidates = sorted(
        (iou(gt_box, prediction["box"]), gt_index, prediction_index)
        for gt_index, gt_box in enumerate(ground_truth)
        for prediction_index, prediction in enumerate(predicted)
    )
    candidates.sort(key=lambda item: (-item[0], item[1], item[2]))
    used_gt: set[int] = set()
    used_predictions: set[int] = set()
    result = []
    for overlap, gt_index, prediction_index in candidates:
        if overlap < threshold or gt_index in used_gt or prediction_index in used_predictions:
            continue
        used_gt.add(gt_index)
        used_predictions.add(prediction_index)
        result.append((gt_index, prediction_index, overlap))
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--iou-threshold", type=float, default=0.30)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    manifest = json.loads((args.dataset / "metadata.json").read_text(encoding="utf-8"))
    prediction_records = {}
    for line in args.predictions.read_text(encoding="utf-8").splitlines():
        if line.strip():
            record = json.loads(line)
            prediction_records[record["file"]] = record

    images = manifest["images"]
    results = []
    total_gt = total_predicted = total_matched = total_exact = 0
    total_characters = total_distance = 0
    for image in images:
        file = image["file"]
        prediction_record = prediction_records.get(file)
        if prediction_record is None:
            raise ValueError(f"prediction is missing: {file}")
        ground_truth = [tuple(map(float, line["bbox"])) for line in image["lines"]]
        predicted = [dict(item, box=bbox(item["box"])) for item in prediction_record["lines"]]
        matches = match(ground_truth, predicted, args.iou_threshold)
        distance = characters = exact = 0
        mismatch_details = []
        for gt_index, prediction_index, overlap in matches:
            expected = normalize(str(image["lines"][gt_index]["text"]))
            actual = normalize(str(predicted[prediction_index]["text"]))
            line_distance = edit_distance(expected, actual)
            distance += line_distance
            characters += len(expected)
            exact += int(expected == actual)
            if expected != actual:
                mismatch_details.append({
                    "ground_truth_index": gt_index,
                    "prediction_index": prediction_index,
                    "iou": overlap,
                    "expected": expected,
                    "actual": actual,
                    "edit_distance": line_distance,
                })
        matched = len(matches)
        item = {
            "file": file,
            "ground_truth_lines": len(ground_truth),
            "predicted_lines": len(predicted),
            "matched_lines": matched,
            "missing_lines": len(ground_truth) - matched,
            "extra_lines": len(predicted) - matched,
            "exact_lines": exact,
            "mean_matched_iou": (sum(overlap for _, _, overlap in matches) / matched
                                  if matched else 0.0),
            "edit_distance": distance,
            "reference_characters": characters,
            "cer_on_matched_lines": distance / characters if characters else 0.0,
            "elapsed_ms": prediction_record.get("elapsed_ms", 0.0),
            "mismatch_details": mismatch_details,
        }
        results.append(item)
        total_gt += len(ground_truth)
        total_predicted += len(predicted)
        total_matched += matched
        total_exact += exact
        total_characters += characters
        total_distance += distance

    report = {
        "status": "ok",
        "images": len(results),
        "ground_truth_lines": total_gt,
        "predicted_lines": total_predicted,
        "matched_lines": total_matched,
        "missing_lines": total_gt - total_matched,
        "extra_lines": total_predicted - total_matched,
        "exact_lines": total_exact,
        "detection_precision": total_matched / total_predicted if total_predicted else 0.0,
        "detection_recall": total_matched / total_gt if total_gt else 0.0,
        "cer_on_matched_lines": total_distance / total_characters if total_characters else 0.0,
        "edit_distance_on_matches": total_distance,
        "reference_characters_on_matches": total_characters,
        "mean_elapsed_ms": sum(item["elapsed_ms"] for item in results) / len(results),
        "max_predicted_lines": max(item["predicted_lines"] for item in results),
        "images_detail": results,
    }
    rendered = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8", newline="\n")
    print(json.dumps({key: value for key, value in report.items() if key != "images_detail"}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
