# Third-party notices

The Java runtime JARs have no third-party runtime dependency. The repository
and release bundle include PP-OCRv6 Tiny model assets used for reproducible
Golden tests and out-of-the-box evaluation.

## PP-OCRv6 Tiny model assets

The following files are derived from the PP-OCR/PaddleOCR ecosystem and are
redistributed under the Apache License 2.0:

- `lw-ppocr-core/src/test/resources/golden/det/det.lwm`;
- `lw-ppocr-core/src/test/resources/golden/cls/cls.lwm`;
- `lw-ppocr-core/src/test/resources/golden/rec/rec.lwm`;
- `lw-ppocr-core/src/test/resources/golden/rec/ppocr_keys.txt`;
- `lw-ppocr-core/src/test/resources/golden/ocr/sample.jpg`;
- derived Golden input and output fixtures stored beside those assets.

The LWM files are converted Object-form derivatives of the corresponding model
weights. Their source repository, pinned source commit, and SHA-256 values are
recorded in the adjacent `manifest.json` files. See
`licenses/PaddleOCR-models-APACHE-2.0.txt` for the full license text.

PP-OCR and PaddleOCR names remain the property of their respective owners. No
endorsement by the upstream project is implied.

## Build and test dependency

The optional ONNX test/download workflow obtains the seven FP32
Tiny/Small/Medium models and their dictionaries listed in
`release/ppocrv6-onnx-manifest.json`, under Apache-2.0. The manifest pins their
SHA-256 and reference source commit. Copyright (c) 2016 PaddlePaddle Authors.
These downloaded files are not bundled into the existing v0.2.1 release.
See `licenses/PaddleOCR-models-APACHE-2.0.txt`.

NumPy and ONNX Runtime are independent validation tools only. They are not
Java runtime dependencies and are not shipped in the Java JARs.
The importer follows the ONNX protobuf specification and the reviewed graph
contracts in lw.PPOCR.C, SimdPaddleOCR and lw.PPOCR.Vulkan; none of those
projects' native or managed libraries are linked into this Java runtime.

JUnit 4.13.2 is used only while running tests. It is not included in the
runtime JARs or the release bundle.
