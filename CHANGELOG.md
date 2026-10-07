# Changelog

All notable changes to this project are documented in this file.

## [Unreleased]

### Fixed

- Outlined dense stride-two Vector tiles into a frequently invoked, array-only
  microkernel, preserving Scalar reduction order and padding semantics.
- DET stem performance validation requires at least 100 warmup calls and one
  second before sampling; chronological samples and runner metadata are recorded.
  The same fixture, checksum and strict 5 ms gate remain enforced.
  See docs/det-stem-ci-stability-20261007.md for evidence and remaining limitations.

## [0.3.1] - 2026-10-07

### Fixed

- ONNX validation on macOS now keeps JVM argument arrays nonempty under Bash
  nounset, preserving the heap limit and all Scalar/Vector/FMA validation modes.

### Changed

- Register-resident 7x7/1x7/7x1 dense and paired 7x7/9x9 depthwise Vector
  kernels preserve Scalar reduction and padding semantics without per-call
  buffers or weight copies. Added bitwise, channel-shard, fixed-species and
  hot-allocation tests; unsupported geometry and explicit switches retain fallback.
- Parallel prepared-Conv post-ops and paired/parallel non-overlapping 2x2
  ConvTranspose now share the existing bounded operator pool. Channel alignment
  preserves activation tail boundaries, and all shards join before reuse/failure.
- Opt-in large-pointwise FMA now uses six output channels and twelve accumulators,
  retaining ordered reduction, 32-pixel scratch, canonical weights and a four-channel
  fallback. No additional model-weight packing or new numerical approximation.
- FMA-only pointwise panels cover >=64 input/output channels, with independent
  Math.fma and real ONNX/ORT validation; default non-FMA eligibility is unchanged.
  `lwppocr.disableExtendedFmaPointwise` restores the previous >=256 range.
- Optional `lwppocr.smallFmaPointwise=true` further extends FMA coverage to >=32;
  it is separately opt-in, validated against real graph/CTC/CLS outputs.
- Fixed DET ConvTranspose dynamic shuffle-mask allocations with equivalent cached
  single-source shuffles/blend masks, including the legacy fallback. Added a
  real-size hot allocation regression without relaxing existing CI thresholds.
- Added preferred/128/512-bit shard, failure-recovery and FMA fallback tests.
  Benchmark JSON records optimization switches; original CI gates are unchanged.

- Large Vector 1x1 Conv now packs bounded 32-pixel input panels into session-reused
  per-worker scratch. It preserves channel reduction order, disjoint spatial
  shards, canonical weights, and Scalar fallbacks; no per-call buffers or default FMA.
- Full OCR benchmark accepts an explicit ONNX model root and Tiny/Small/Medium
  variant, reports the model identity, and retains the existing Tiny LWM defaults.
- Added large-channel kernel diagnostics and fixed 128/512-bit correctness tests
  to CI without relaxing the existing performance/allocation gates. See
  docs/medium-optimization-20261007.md for the controlled local comparison.

### Added

- Opt-in `-Dlwppocr.vectorFma=true` fused FP32 large-pointwise kernels, with
  Math.fma reference tests. Default reduction remains Scalar-bit-identical.
- An isolated, hash-locked Windows Java/C# CPU comparison runner for all three
  models, sharing decoded BGR pixels and REC buckets; it reports text/crop
  differences and process memory independently of CI performance gates.

### Compatibility and measurement scope

- Public OCR APIs, Tiny LWM bundle, reviewed Tiny/Small/Medium FP32 ONNX model
  set and width buckets remain compatible with 0.3.0. Core/ImageIO retain Java 8
  bytecode; full builds and the optional Vector backend require JDK 25.
- FMA and >=32-channel FMA panels remain opt-in and change rounding. Validate
  your actual models/dataset before enabling them.
- The local CPU8 two-process sample comparison measured Medium about 9% lower
  latency than pinned C# commit 3e4192f; Tiny/Small were still slower. Crops are
  not identical and no dataset CER or universal speed advantage is claimed.
  See docs/spatial-convolution-optimization-20261007.md for the workload and evidence.
- The original latency/allocation gates remain unchanged; no models, benchmark
  logs, C# sources or test-only ONNX Runtime dependencies are added to the ZIP.

## [0.3.0] - 2026-10-04

### Added

- Pure Java, bounded ONNX protobuf importer and content-based ModelLoader,
  sharing the existing IR and Scalar/Vector executors without native dependencies.
- Direct FP32 PP-OCRv6 Small/Medium loading with the shared CLS and matching
  18,710-class dictionary, preserving dynamic REC width buckets.
- Hash-locked model acquisition, independent ORT graph-output checks, sample
  pipeline text/score/rotation verification, and three-platform CI coverage.
- Release ZIP acquisition-tool validation and manual release re-runs that
  validate the selected tag consistently across every prerequisite job.

### Fixed

- Spatial Conv boundary panels now use independent single-accumulator Vector
  loops, avoiding the two-CPU JDK 25 boxing/allocation regression without changing
  padding semantics or accumulation order. CI checks two fresh low-CPU JVMs
  against the existing 1 MB/OCR and GC limits and uploads evidence even on failure.
- Slice execution now honors ends; bounded Q/K/V slices cannot copy an input
  tail past their output workspace. Prepared plans avoid per-run Slice allocation.
- MatMul shape/execution now support broadcast matrix batches (rank >= 2).
- Multi-axis Unsqueeze uses output-axis semantics, with duplicate-axis checks.

### Compatibility

- Tiny LWM APIs and the bundled model layout remain compatible with v0.2.1.
- The ZIP includes the SHA-256-locked ONNX manifest and optional standard-library
  Python downloader; Small/Medium model binaries are separate downloads.
- ONNX support is a reviewed PP-OCR subset, not arbitrary ONNX compatibility.
- C sample text differences are retained separately from independently
  verified Java pipeline Goldens; see docs/onnx-models.md.

## [0.2.1] - 2026-09-27

### Changed

- Connected AUTO DET/REC intra-op budgets to the prepared graph executor, with
  bounded shared workers and disjoint convolution/matrix shards.
- Added prepared physical Conv instructions and eligible BN, bias, residual,
  and activation fusion, sharing one plan with the workspace allocator.
- Reused REC projection weight panels across larger row tiles and split the
  MatMul hot loop to reduce Vector API compilation-stage allocation.
- Added prepared spatial convolution input panels and output-channel packed
  weights. Extra model-wide packing is bounded to 80 KiB; weights exceeding
  the cache budget use session-reused scratch rather than per-call arrays.
- Included spatial packing in the total prepared-weight benchmark metric,
  with separate spatial weight/scratch diagnostics.

### Added

- Reproducible local end-to-end OCR measurements, environment/model hashes,
  frozen source snapshots, cold-start and process-peak memory records.
- Regression coverage for fusion liveness, parallel tails, spatial padding,
  dilation, nonfinite weights, cache budgets, and reused scratch.

### Compatibility

- Existing OCR APIs, Tiny models, dictionary and detection thresholds remain
  unchanged. The graph remains NCHW; this is not a graph-wide NHWC migration.
- Core remains Java 8 bytecode; the optional Vector backend requires JDK 25.
- Release publication remains gated by three-platform tests and the existing
  performance checks. Local measurements do not prove dataset CER or CI success.

## [0.2.0] - 2026-09-20

### Highlights

- Significantly reduced steady-state heap usage and OCR allocation through
  zero-copy execution, reusable PP-OCR staging buffers, workspace aliasing,
  lazy constants, and shared prepared REC projection weights.
- Added fused REC projection, winning Softmax probability, ArgMax, and compact
  CTC decoding, avoiding the retained dense `[T,C]` probability matrix on
  supported PP-OCRv6 Tiny REC graphs.
- Added AUTO CPU budgeting for CLS/REC workers while preserving MANUAL
  controls for compatibility.
- Hardened and specialized the JDK 25 Vector API backend, including REC
  stride-two convolution and fixed-species pointwise tuning for 256-bit and
  512-bit Vector API hosts.
- Added steady-state allocation/JFR diagnostics and conservative CI performance
  regression gates for latency, allocation, GC, workspace efficiency, and
  retained prepared weights.

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
- A REC hot-shape Vector benchmark matrix covering pointwise and stride-two
  convolution widths, including steady-state allocation per operation.
- REC hot-shape measurements now prepare all selected shapes and complete a
  global warmup before allocation sampling, with pointwise and stride-two
  filters for focused kernel experiments.
- Added the experimental \`lwppocr.vectorPointwiseBlock\` property for
  pointwise block 4/8/12 comparisons; the default remains block 12.
- Added fixed-species pointwise block matrices, Vector API capability reporting,
  and an eligibility-gated comparison summary for 256-bit and (when supported)
  512-bit CI hosts; experimental results never change the production default.

### Changed

- Isolated the Vector API 1x1 convolution hot path in `VectorPointwiseKernel`
  while preserving its NCHW layout, output blocking, dispatch behavior, and
  numerical results.
- Vector split microkernels now keep their JDK 25 `VectorSpecies` specialization local
  to each hot compilation unit, including static stride-two gather indexes, restoring
  stable JIT specialization without changing the Scalar correctness path.
- Dynamic REC width sessions now share one model-wide prepared projection matrix;
  retained packed projection weights are reported once instead of once per width bucket.
- The first Tiny REC 3x3 stride-two convolution now has a strict fixed-shape NCHW
  Vector kernel with scalar border handling and no steady-state per-operation arrays.
- Linux performance CI now fails on conservative latency, allocation, GC, workspace,
  or retained-weight regressions after publishing the focused benchmark inputs.
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

[0.2.0]: https://github.com/lxw112190/lw.PPOCR.Java/releases/tag/v0.2.0
[0.2.1]: https://github.com/lxw112190/lw.PPOCR.Java/releases/tag/v0.2.1
[0.3.0]: https://github.com/lxw112190/lw.PPOCR.Java/releases/tag/v0.3.0
[0.3.1]: https://github.com/lxw112190/lw.PPOCR.Java/releases/tag/v0.3.1
[0.1.0]: https://github.com/lxw112190/lw.PPOCR.Java/releases/tag/v0.1.0
