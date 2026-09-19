#!/usr/bin/env python3
"""Small standard-library smoke test for compare-predictions.py."""

from __future__ import annotations

import importlib.util
from pathlib import Path


def load_helper():
    path = Path(__file__).with_name("compare-predictions.py")
    spec = importlib.util.spec_from_file_location("compare_predictions", path)
    if spec is None or spec.loader is None:
        raise AssertionError("could not load compare-predictions.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def record(text: str = "abc", box=None) -> dict:
    return {
        "file": "sample.jpg",
        "lines": [{
            "text": text,
            "cls": 0,
            "rotate": False,
            "box": box or [0, 0, 10, 0, 10, 10, 0, 10],
        }],
    }


def main() -> int:
    helper = load_helper()
    baseline = {"sample.jpg": record()}

    equal = helper.compare(baseline, {"sample.jpg": record()}, 1.0)
    assert equal["status"] == "ok", equal

    changed_text = helper.compare(
        baseline, {"sample.jpg": record(text="abd")}, 1.0)
    assert changed_text["status"] == "mismatch", changed_text
    assert changed_text["text_mismatches"] == 1, changed_text

    changed_box = helper.compare(
        baseline, {"sample.jpg": record(box=[0, 0, 12, 0, 12, 10, 0, 10])}, 1.0)
    assert changed_box["status"] == "mismatch", changed_box
    assert changed_box["box_mismatches"] == 1, changed_box

    missing = helper.compare(baseline, {}, 1.0)
    assert missing["status"] == "mismatch" and missing["missing_files"] == ["sample.jpg"], missing

    print("compare-predictions tests: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
