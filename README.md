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

The first milestone implements the Maven multi-module foundation and a defensive
LWM v0.1 loader. The loader validates the complete untrusted model byte stream
before publishing an immutable `LwmModel`.

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

The current code does not claim full OCR inference yet. REC graph execution,
preprocessing, CTC decoding, CLS, DET, DB postprocess, crop, and the public OCR
facade will be added only after the model contract and loader are stable.
