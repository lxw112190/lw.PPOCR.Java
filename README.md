# lw.PPOCR.Java

A lightweight pure-Java PP-OCR inference runtime.

- No Python
- No Paddle Inference
- No ONNX Runtime
- No OpenCV native library
- No JNI
- Shared LWM v0.1 model format with `lw.PPOCR.C`
- Scalar correctness path plus an optional JDK 25 Vector API backend

## Current milestone

The repository now runs the complete PP-OCRv6 Tiny DET/CLS/REC pipeline. A
defensive LWM v0.1 loader validates the full untrusted model stream, dynamic
shapes are resolved before execution, and a lifetime-based workspace planner
reuses aligned tensor storage without native memory.

The Scalar correctness backend covers the operators used by the committed Tiny
models, including standard binary broadcasting, BatchNormalization,
Conv/ConvTranspose, pooling, Resize, layout operators, Softmax, and rank-3 by
rank-2 MatMul. Golden tests compare Java graph outputs and the 16-line full OCR
result against the pinned `lw.PPOCR.C` baseline.

The PP-OCR layer now includes C-compatible DET/CLS/REC preprocessing,
DB postprocess, perspective crop, UTF-8 dictionary loading, greedy CTC
decoding, orientation correction, reading-order sorting, and the public
`PaddleOcr` pipeline. Thresholds and reading policies are supplied through
immutable `PaddleOcrOptions`; concurrent callers can use `OcrWorkerPool` with
independent sessions.

REC can optionally evaluate different dynamic-width groups concurrently while
preserving input and reading order. The default parallelism is one; applications
opt in through `PaddleOcrOptions.setRecognitionParallelism`.

The optional JDK 25 Vector API backend accelerates all Conv configurations used
by the Tiny models, the DET 2x upsampling ConvTranspose path, MatMul, reductions,
activations, and binary broadcasting. Unsupported generic shapes continue to
fall back to Scalar correctness.

The runtime intentionally does not parse ONNX. Model conversion remains an
offline responsibility of `lw.PPOCR.C` and its converter.

## Build

The design baseline is JDK 25. Core sources target Java 8 bytecode and standard
APIs, while `lw-ppocr-vector` targets JDK 25 and requires the incubating Vector
API at compile time and launch time. The optional module is deliberately
separate and has no role in Scalar correctness.

```text
mvn test
```

## Project layout

```text
lw-ppocr-core/    LWM model layer and scalar-safe foundation
lw-ppocr-imageio/ Optional standard-Java BufferedImage/ImageIO adapter
lw-ppocr-vector/  Optional JDK 25 Vector API backend
lw-ppocr-benchmark/  Dependency-free loader benchmark harness
```

GitHub Actions is the build authority for this repository. It compiles and
tests on JDK 25 across Linux, Windows, and macOS. The Linux performance job
reports loading, DB postprocess, preprocessing, model workload, and complete
OCR results for Scalar, Vector, and Vector with four REC width workers. Full OCR
JSON separates stage timing, GC activity, model memory, retained heap, and
transient heap. Performance output is a development signal only; v0.x does not
use it as a release gate.

For applications using AWT/ImageIO, `lw-ppocr-imageio` also provides
`PaddleOcrImageIo` convenience methods for `Path`, `InputStream`, and
`BufferedImage`. The core module remains independent of AWT and ImageIO.

See [docs/installation.md](docs/installation.md) for Maven dependencies,
model layout, BGR/ImageIO usage, lifecycle, and concurrency guidance.

## Scope boundaries

The currently verified contract is the committed dynamic-shape FP32 PP-OCRv6
Tiny model set. The runtime does not claim arbitrary ONNX topology
compatibility, automatic model discovery, GPU, or Android support. The Vector
API backend is optional and keeps Scalar fallbacks for shapes outside its
optimized paths. Image decoding is available separately through the optional
`lw-ppocr-imageio` module.
