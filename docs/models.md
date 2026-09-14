# Models

This page defines the model download, provenance, integrity, and compatibility
policy for `lw.PPOCR.Java`.

## Official model set

| Component | Model | Format | Precision | Status |
|---|---|---|---|---|
| Text detection | PP-OCRv6 Tiny DET | LWM v0.1 | FP32 | Verified |
| Direction classification | PP-OCRv6 Tiny CLS | LWM v0.1 | FP32 | Verified |
| Text recognition | PP-OCRv6 Tiny REC | LWM v0.1 | FP32 | Verified |

The complete DET/CLS/REC set is covered by graph-level Golden tests and the
committed 500×500, 16-line full OCR fixture.

## Download

Official models are shipped with every tagged GitHub Release:

➡️ [Download the latest Release](https://github.com/lxw112190/lw.PPOCR.Java/releases/latest)

The single Release ZIP contains the runtime JARs, models, recognition
dictionary, sample image, documentation, licenses, and SHA-256 manifests. No
separate model download or ONNX conversion is required for the official Tiny
model set.

## Release layout contract

All examples and release smoke tests use this stable relative layout:

```text
models/
└── ppocrv6-tiny/
    ├── det.lwm
    ├── cls.lwm
    ├── rec.lwm
    ├── ppocr_keys.txt
    ├── sample.jpg
    └── manifest.json
```

Applications should resolve the model directory once:

```java
Path modelRoot = Paths.get("models", "ppocrv6-tiny");
```

`det.lwm`, `cls.lwm`, and `rec.lwm` are the detection, direction
classification, and recognition graphs. `ppocr_keys.txt` is the recognition
dictionary. The sample image remains in this directory because it is part of
the versioned, integrity-checked release asset set.

## Provenance and license

- Source repository: [`lw.PPOCR.C`](https://github.com/lxw112190/lw.PPOCR.C)
- Pinned conversion baseline: `9b31f1b`
- Model family: PP-OCRv6 Tiny
- Model asset license: Apache License 2.0
- Java runtime license: MIT

The canonical source manifest is maintained at
[`release/ppocrv6-tiny-manifest.json`](https://github.com/lxw112190/lw.PPOCR.Java/blob/main/release/ppocrv6-tiny-manifest.json).
Inside an extracted Release, the verified copy is
`models/ppocrv6-tiny/manifest.json`.
Redistribution details are recorded in
[`THIRD-PARTY-NOTICES.md`](../THIRD-PARTY-NOTICES.md) and the model license in
[`licenses/PaddleOCR-models-APACHE-2.0.txt`](../licenses/PaddleOCR-models-APACHE-2.0.txt).

## Integrity

Every release build verifies the SHA-256 digest of all five model-set assets
against `manifest.json` before packaging. The complete ZIP also contains
`SHA256SUMS.txt`, and the published ZIP has its own `.sha256` sidecar. CI
extracts the candidate, verifies those checksums, and runs full OCR from the
extracted files before publishing the GitHub Release.

## LWM compatibility

The current public model contract is dynamic-shape FP32 PP-OCRv6 Tiny encoded
as LWM v0.1. The Java runtime validates the model header, checksums, tensors,
nodes, parameters, graph indexes, and supported shapes before execution. See
[`lwm-v0.1-compatibility.md`](lwm-v0.1-compatibility.md) for format boundaries.

Support for an operator does not imply compatibility with every arbitrary ONNX
graph using that operator. A model set is considered officially supported only
after graph-output and full-pipeline Golden coverage has been added.

## Custom models

`lw.PPOCR.Java` deliberately does not parse ONNX and does not include a model
converter. To use another PP-OCR model:

```text
ONNX model
    → lw.PPOCR.C offline converter
    → LWM files and matching dictionary
    → lw.PPOCR.Java
```

Use the conversion tools from
[`lw.PPOCR.C`](https://github.com/lxw112190/lw.PPOCR.C), then validate the
resulting graph against the Java runtime. Keep converted assets outside the
application JAR when independent model updates are required.

Official PP-OCRv6 Tiny models are ready to use from the Release ZIP. Custom
models are the only case that requires the separate conversion workflow.
