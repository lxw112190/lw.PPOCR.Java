#!/usr/bin/env python3
"""Independent ORT CPU oracle, test-only. Never used by the Java runtime."""
import argparse
import hashlib
import json
import shutil
from pathlib import Path
import numpy as np
import onnxruntime as ort

ROOT = Path(__file__).resolve().parents[1]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads((ROOT / "release/ppocrv6-onnx-manifest.json").read_text("utf-8"))
    for asset in manifest["assets"]:
        if hashlib.sha256((args.models / asset["path"]).read_bytes()).hexdigest() != asset["sha256"]:
            raise ValueError("model/dictionary hash mismatch: " + asset["path"])
    args.output.mkdir(parents=True, exist_ok=True)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    descriptors, hashes = [], {}
    def record(model, height, width):
        session = ort.InferenceSession(str(args.models / model), options, providers=["CPUExecutionProvider"])
        count = 3 * height * width
        values = ((((np.arange(count, dtype=np.int64) * 17) % 257) - 128).astype(np.float32) / np.float32(127)).reshape(1, 3, height, width)
        expected = session.run(None, {session.get_inputs()[0].name: values})[0]
        if not np.isfinite(expected).all():
            raise ValueError("non-finite reference output")
        stem = model.replace("/", "-").removesuffix(".onnx") + f"-{height}x{width}"
        input_name, output_name = stem + "-input.f32", stem + "-output.f32"
        values.astype("<f4").tofile(args.output / input_name)
        expected.astype("<f4").tofile(args.output / output_name)
        for name in (input_name, output_name):
            hashes[name] = hashlib.sha256((args.output / name).read_bytes()).hexdigest()
        descriptors.append("\t".join([model, str(height), str(width), input_name, output_name, ",".join(map(str, expected.shape))]))
        print(f"ORT {model} {height}x{width} -> {list(expected.shape)}", flush=True)
    for variant in ("tiny", "small", "medium"):
        for width in (17, 192, 960):
            record(f"ppocrv6-{variant}/rec.onnx", 48, width)
        for height, width in ((32, 64), (320, 320)):
            record(f"ppocrv6-{variant}/det.onnx", height, width)
    record("ppocrv6-tiny/cls.onnx", 80, 160)
    (args.output / "cases.tsv").write_text("\n".join(descriptors) + "\n", encoding="utf-8")
    shutil.copyfile(ROOT / "lw-ppocr-core/src/test/resources/golden/ocr/sample.jpg", args.output / "sample.jpg")
    for variant in ("small", "medium"):
        shutil.copyfile(ROOT / f"lw-ppocr-core/src/test/resources/golden/onnx/{variant}-full-ocr.txt", args.output / f"{variant}-full-ocr.txt")
        shutil.copyfile(ROOT / f"lw-ppocr-core/src/test/resources/golden/onnx/{variant}-pipeline-full-ocr.txt", args.output / f"{variant}-pipeline-full-ocr.txt")
    for name in ("cases.tsv", "sample.jpg", "small-full-ocr.txt", "medium-full-ocr.txt", "small-pipeline-full-ocr.txt", "medium-pipeline-full-ocr.txt"):
        hashes[name] = hashlib.sha256((args.output / name).read_bytes()).hexdigest()
    (args.output / "manifest.json").write_text(json.dumps({
        "schema_version": 1, "oracle": "onnxruntime CPUExecutionProvider", "onnxruntime_version": ort.__version__,
        "reference_commit": manifest["reference_commit"], "assets": manifest["assets"], "files": hashes,
        "max_abs_tolerance": 1e-3, "mean_abs_tolerance": 1e-4,
    }, indent=2) + "\n", encoding="utf-8")

if __name__ == "__main__":
    main()
