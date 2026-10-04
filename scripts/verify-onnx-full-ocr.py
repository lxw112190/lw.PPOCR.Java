#!/usr/bin/env python3
"""Check actual sample crops with independent ORT CLS/REC. Strict text, score and rotation parity."""
import argparse
import base64
import difflib
import hashlib
import json
from pathlib import Path
import numpy as np
import onnxruntime as ort

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--fixtures", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads((args.fixtures / "manifest.json").read_text("utf-8"))
    for name, expected in manifest["files"].items():
        if hashlib.sha256((args.fixtures / name).read_bytes()).hexdigest() != expected:
            raise ValueError("fixture hash mismatch: " + name)
    for asset in manifest["assets"]:
        if hashlib.sha256((args.models / asset["path"]).read_bytes()).hexdigest() != asset["sha256"]:
            raise ValueError("model hash mismatch")
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    cls = ort.InferenceSession(str(args.models / "ppocrv6-tiny/cls.onnx"), options, providers=["CPUExecutionProvider"])
    labels = (args.models / "ppocrv6-shared/PP-OCRv6_small_rec_dict.txt").read_text("utf-8-sig").splitlines()
    labels = [""] + labels + [" "]
    reports = []
    for variant in ("small", "medium"):
        rec = ort.InferenceSession(str(args.models / f"ppocrv6-{variant}/rec.onnx"), options, providers=["CPUExecutionProvider"])
        rows = (args.fixtures / f"{variant}-stage-inputs.tsv").read_text("utf-8").splitlines()
        if len(rows) != 16:
            raise ValueError("expected 16 sample lines")
        texts = []
        maximum_score_error = 0
        for row in rows:
            stem, width, text64, score, cls_label, cls_score, rotated = row.split("\t")
            width = int(width)
            if width not in (192, 320, 480, 640, 960):
                raise ValueError("invalid width bucket")
            data = np.fromfile(args.fixtures / (stem + "-cls.f32"), dtype="<f4").reshape(1, 3, 80, 160)
            probabilities = cls.run(None, {cls.get_inputs()[0].name: data})[0].reshape(-1)
            label = int(probabilities.argmax())
            if label != int(cls_label) or abs(float(probabilities[label]) - float(cls_score)) > 1e-3:
                raise ValueError(f"{stem}: CLS mismatch")
            if (label != 0 and float(probabilities[label]) > .9) != (rotated == "true"):
                raise ValueError(f"{stem}: rotation mismatch")
            data = np.fromfile(args.fixtures / (stem + "-rec.f32"), dtype="<f4").reshape(1, 3, 48, width)
            probabilities = rec.run(None, {rec.get_inputs()[0].name: data})[0].reshape(-1, len(labels))
            ids = probabilities.argmax(axis=1)
            selected = [i for i, current in enumerate(ids) if current != 0 and (i == 0 or current != ids[i-1])]
            text = "".join(labels[int(ids[i])] for i in selected)
            reference_score = float(np.mean([probabilities[i, ids[i]] for i in selected])) if selected else 0
            if text != base64.b64decode(text64).decode("utf-8"):
                raise ValueError(f"{stem}: CTC text mismatch: ORT={text!r}, Java={base64.b64decode(text64).decode('utf-8')!r}")
            error = abs(reference_score - float(score))
            maximum_score_error = max(maximum_score_error, error)
            if error > 1e-3:
                raise ValueError(f"{stem}: REC score mismatch: {error}")
            texts.append(text)
        reference = "\n".join(texts)
        pipeline_text = (args.fixtures / f"{variant}-pipeline-full-ocr.txt").read_text("utf-8").strip()
        if reference != pipeline_text:
            raise ValueError(f"{variant}: independent ORT text does not match committed pipeline Golden")
        c_text = (args.fixtures / f"{variant}-full-ocr.txt").read_text("utf-8").strip()
        difference = list(difflib.unified_diff(c_text.splitlines(), texts, fromfile="C sample fixture", tofile="Java crops + ORT", lineterm=""))
        report = {"variant": variant, "lines": len(texts), "ort_ctc_text_parity": True, "cls_rotation_parity": True,
                  "max_score_error": maximum_score_error, "text_sha256": hashlib.sha256(reference.encode("utf-8")).hexdigest(),
                  "c_reference_text_parity": c_text == reference, "c_reference_diff": difference}
        (args.fixtures / f"{variant}-ort-full-ocr.txt").write_text(reference + "\n", encoding="utf-8")
        print(json.dumps(report, ensure_ascii=True), flush=True)
        reports.append(report)
    (args.fixtures / "full-ocr-reference-report.json").write_text(json.dumps(reports, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

if __name__ == "__main__":
    main()
