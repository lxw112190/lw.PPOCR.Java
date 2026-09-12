# lw.PPOCR.Java

A lightweight pure-Java PP-OCR inference runtime.

- No Python
- No Paddle Inference
- No ONNX Runtime
- No OpenCV native library
- No JNI
- Shared LWM v0.1 model format with `lw.PPOCR.C`
- Scalar correctness path first; Vector API remains optional

## Current milestone

The first milestones implement the Maven multi-module foundation, a defensive
LWM v0.1 loader, concrete shape validation, a lifetime-based workspace planner,
and the first scalar graph execution paths. The loader validates the complete
untrusted model byte stream before publishing an immutable `LwmModel`; the
planner prepares reusable storage for the graph executor.

The scalar executor currently covers equal-shape elementwise arithmetic,
ReLU, Sigmoid, NCHW Conv, ConvTranspose, pooling, Resize, Transpose, Reshape,
Softmax, and rank-2 MatMul.

The PP-OCR layer now includes C-compatible DET/CLS/REC preprocessing,
DB postprocess, perspective crop, UTF-8 dictionary loading, greedy CTC
decoding, orientation correction, reading-order sorting, and the public
`PaddleOcr` pipeline. Thresholds and reading policies are supplied through
immutable `PaddleOcrOptions`.

The runtime intentionally does not parse ONNX. Model conversion remains an
offline responsibility of `lw.PPOCR.C` and its converter.

## Build

The design baseline is JDK 25. The bootstrap sources currently target Java 8
standard APIs so the loader can be compiled in older development environments;
the project does not use Java 8 compatibility as a promise for the final
runtime. The `lw-ppocr-vector` module is deliberately separate and has no role
in scalar correctness.

```text
mvn test
```

## Project layout

```text
lw-ppocr-core/    LWM model layer and scalar-safe foundation
lw-ppocr-vector/  Reserved optional Vector API backend
lw-ppocr-benchmark/  Dependency-free loader benchmark harness
```

GitHub Actions is the build authority for this repository. It compiles and
tests on JDK 25 across Linux, Windows, and macOS, and runs the benchmark as a
separate Linux job. Performance output is a development signal only; v0.x does
not use it as a release gate.

## Scope boundaries

The current code targets the fixed-shape FP32 PP-OCRv6 Tiny/Small/Medium
contract. It does not claim arbitrary ONNX topology compatibility, dynamic
model discovery, image decoding, or a release-ready Vector API backend.
