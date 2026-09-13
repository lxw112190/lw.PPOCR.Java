# lw.PPOCR.Java

[中文说明](README.zh-CN.md)

A lightweight pure-Java PP-OCRv6 inference runtime with no native
dependencies. 轻量级纯 Java PP-OCRv6 推理运行时，无原生依赖。

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

CLS and REC expose independent optional parallelism while preserving input and
reading order. Applications opt in through
`PaddleOcrOptions.setClassificationParallelism` and `setRecognitionParallelism`;
both default to one. CLS distributes fixed-shape line evaluations over sessions
that share decoded model constants. REC queues the largest estimated width-group
workloads first so idle workers can immediately take the next group without
creating duplicate sessions.

For CPU-budgeted scheduling, use
`setParallelismMode(ParallelismPolicy.AUTO)` (or `setParallelism(0)`). AUTO
derives one plan from the processors visible to the JVM, caps line workers at
four, clamps workers to the detected line count, and prevents its planned REC
worker/intra-op product from exceeding that CPU budget. MANUAL remains the
default for compatibility; either individual worker setter selects it.

The optional JDK 25 Vector API backend accelerates all Conv configurations used
by the Tiny models, the DET 2x upsampling ConvTranspose path, MatMul, reductions,
activations, and binary broadcasting. It also fuses the exact five-node
`DIV -> ERF -> ADD -> MUL -> MUL` GELU expression used by the Tiny models after
validating tensor connections, shapes, constants, and exclusive intermediate
uses. Unsupported generic shapes continue to fall back to Scalar correctness.

The runtime intentionally does not parse ONNX. Model conversion remains an
offline responsibility of `lw.PPOCR.C` and its converter.

## Build

The design baseline is JDK 25. Core sources target Java 8 bytecode and standard
APIs, while `lw-ppocr-vector` targets JDK 25 and requires the incubating Vector
API at compile time and launch time. The optional module is deliberately
separate and has no role in Scalar correctness.

```text
mvn verify
```

## Release

`0.1.0` is the first pre-1.0 release. Tagged builds produce a release-candidate
bundle containing the three runtime JARs, PP-OCRv6 Tiny LWM models, dictionary,
sample image, documentation, and license notices. Each ZIP has a SHA-256
sidecar, and CI runs full OCR from the extracted bundle before publishing it
to GitHub Releases and retaining the same files as an Actions artifact.

The artifacts are not published to Maven Central yet. Install them into the
local Maven repository with `mvn clean install`, or use the JARs from the
tagged GitHub Release. See [docs/releasing.md](docs/releasing.md) for the
maintainer checklist and [CHANGELOG.md](CHANGELOG.md) for release notes.

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
OCR results for Scalar, Vector, Vector with four CLS and REC workers, and AUTO
plans under constrained processor counts. Full OCR
JSON separates stage timing, GC activity, model memory, retained heap, and
transient heap. Schema 3 measures wall time without the operator profiler and
runs one separate warmed diagnostic invocation. Aggregate `operators` remain
available, while `stage_operators` separates DET, CLS, and REC; parallel operator
time is the sum across participating threads. `stage_hot_nodes` reports the
slowest resolved graph nodes and their tensor shapes for targeted tuning.
Focused benchmarks track the `60 x 80` by `80 x 6906` REC projection MatMul and
the fused REC terminal path (`MatMul + bias + Softmax + ArgMax`). Supported REC
graphs use the fused path by default, retaining one class row plus compact CTC
ids/scores instead of the dense `[T,C]` output. Set
`-Dlwppocr.disableProjectionFusion=true` to force the compatibility path.
Focused benchmarks also track
the `[1,24,24,480]` to `[1,48,12,240]` REC stride-two Conv without
full-pipeline scheduling noise. `[1,64,80,80]` and `[1,64,128,128]` to
16-channel benchmarks track the hot DET stride-one Conv at detector limits 320
and 960. A `[1,16,160,160]` same-size benchmark tracks its 2x2 stride-one
MaxPool under the same policy. The DET
stride-two benchmark covers both the `[1,3,320,320]` stem and the
`[1,32,160,160]` and `[1,32,256,256]` downsample layers, each with 16 output
channels, including detector limits 320 and 960. Two additional
benchmarks cover the same-size 2x2 DET layers with `16 -> 8` and `8 -> 16`
channels. A same-run Scalar and Vector benchmark also covers the repeated 5x5
depthwise CLS layer (`[1,64,5,80]`), so runner-wide load changes can be
separated from backend speedups.
Performance output is a development signal only; v0.x does not use it as a release gate.

Prepared inference sessions, preprocessing arrays, DB geometry scratch space,
and per-line perspective-crop pixel buffers are reused across synchronous OCR
calls. A `PaddleOcr` instance is therefore intended for one caller at a time;
use separate instances or `OcrWorkerPool` for concurrent requests.

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

## License

The Java source is available under the [MIT License](LICENSE). The committed
PP-OCRv6 Tiny model assets and derived Golden fixtures are redistributed under
Apache License 2.0; see [THIRD-PARTY-NOTICES.md](THIRD-PARTY-NOTICES.md).

## 联系与支持

- 作者：天天代码码天天
- QQ：819069052
- QQ Group: 天天代码码天天 | 群号: 264292622
- 项目地址：<https://github.com/lxw112190/lw.PPOCR.Java>

如果项目对你有帮助，可以扫码支持维护：

<img src="docs/assets/sponsor.jpg" alt="捐赠二维码" width="240">
