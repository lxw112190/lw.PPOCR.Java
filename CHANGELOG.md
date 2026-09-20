# Changelog

All notable changes to this project are documented in this file.

## [Unreleased]

### Download

- Tagged Release ZIPs contain the runtime JARs, PP-OCRv6 Tiny DET/CLS/REC LWM
  models, recognition dictionary, sample image, documentation, license notices,
  and SHA-256 manifests. No separate model download is required.

### Added

- REC terminal pattern detection and fused projection, bias, winning Softmax
  probability, ArgMax, and compact CTC decoding for Scalar and Vector backends.
- Dense compatibility fallback controlled by
  `-Dlwppocr.disableProjectionFusion=true`.
- Dynamic-width parity coverage at widths 192, 320, 480, 640, and 960, plus a
  focused dense-versus-fused CI benchmark.
- Opt-in AUTO CPU budgeting with deterministic 1/2/4/8-core worker plans and
  retained MANUAL controls for advanced callers.
- Full OCR allocation reporting through HotSpot thread counters, with an
  uploaded JFR recording and top allocation-class summary in Linux CI.
- A ten-iteration allocation warmup that separates steady-state allocation
  from Vector API C2 compilation and escape-analysis startup behavior.
- A reproducible DET graph/input dump and standalone DB postprocess dump for
  direct comparison with reference runtimes.
- An immutable compiled-model cache that shares decoded constants, node index
  arrays, and read-only parameter views across shape-specialized sessions.
- Model-wide tensor consumer counts and last-use indexes for fusion-safety and
  future workspace-alias planning, reused by REC terminal-fusion detection.
- Shape-specialized `PreparedNode[]` execution metadata with direct indexed
  plans, removing `IdentityHashMap` lookups from the inference hot loop.
- Cross-platform Vector projection parity now uses a numerical tolerance for
  softmax probabilities while retaining exact class-ID comparison.
- A first-page Quick Start, dedicated model guide, and direct model-download
  links in both English and Chinese project home pages.
- Structured bug, feature, and question Issue forms with model and installation
  guidance shown before a question is submitted.
- A Release-layout verifier that checks the documented model path, required
  files, model manifest, and SHA-256 values before packaging.
- An external metadata-backed dataset runner and CER evaluator for comparing
  the Java pipeline with the local `lw.PPOCR.C` reference corpus.

### Changed

- Vector split microkernels now keep their JDK 25 `VectorSpecies` specialization local
  to each hot compilation unit, including static stride-two gather indexes, restoring
  stable JIT specialization without changing the Scalar correctness path.
- Dynamic REC width sessions now share one model-wide prepared projection matrix;
  retained packed projection weights are reported once instead of once per width bucket.
- Linux performance CI now fails on conservative latency, allocation, GC, workspace,
  or retained-weight regressions after publishing the focused benchmark inputs.
- The Vector DET path now uses a bounded NHWC-friendly channel-tiled 3x3 kernel for
  the large 16-channel feature maps while keeping the public NCHW tensor contract
  and Scalar fallback for unsupported shapes.
- DB postprocess now follows the C/Paddle minimum-side filtering stages before
  scoring, after unclip, and after source-coordinate restoration, with float
  geometry arithmetic for closer cross-runtime parity.
- Fixed the dynamic DET cache's height/width argument inversion so non-square
  images keep their declared `[1,3,height,width]` session shape.
- REC sessions no longer retain the dense `[T,C]` probability matrix when the
  supported terminal pattern is active.
- Synchronous OCR calls reuse line, crop, classification, recognition, rotation,
  REC width-group, and task-future staging storage at their high-water sizes.
- Release bundles now include `QUICKSTART.md` at the archive root.

## [0.1.0] - 2026-09-13

### Added

- Complete pure-Java PP-OCRv6 Tiny DET, CLS, and REC pipeline.
- Defensive LWM v0.1 loader with shape, graph, bounds, and checksum validation.
- Scalar correctness backend targeting the Java 8 API and bytecode contract.
- Optional JDK 25 Vector API backend with optimized Conv, ConvTranspose,
  MatMul, activation, reduction, pooling, and binary-broadcast paths.
- Dynamic DET and REC sessions, reusable workspaces, optional CLS/REC
  parallelism, DB postprocess, perspective crop, CTC decode, and reading order.
- ImageIO adapter for `Path`, `InputStream`, and `BufferedImage` inputs.
- Real-model DET, CLS, REC, and 16-line full OCR Golden tests pinned to
  `lw.PPOCR.C` commit `9b31f1b`.
- Linux performance reporting for model loading, preprocessing, DB postprocess,
  focused kernels, full OCR latency, JVM heap, GC, and hot graph nodes.
- JDK 25 CI on Linux, Windows, and macOS.

### Compatibility and limits

- The verified model contract is dynamic-shape FP32 PP-OCRv6 Tiny in LWM v0.1.
- Core and ImageIO artifacts use the Java 8 API/bytecode contract; building the
  full reactor and using the optional Vector backend require JDK 25.
- CPU inference only. ONNX parsing, GPU, Android, and arbitrary graph topology
  are outside the `0.1.0` support contract.
- This is a pre-1.0 release; public APIs may still evolve in later minor
  versions.

[0.1.0]: https://github.com/lxw112190/lw.PPOCR.Java/releases/tag/v0.1.0
