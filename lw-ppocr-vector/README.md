# lw-ppocr-vector

Optional JDK 25 Vector API backend for the pure-Java PP-OCR runtime. It
accelerates standard binary broadcasting, activations, reductions, MatMul, all
Conv configurations used by the PP-OCRv6 Tiny models, and their 2x DET
ConvTranspose path. Shapes outside the optimized paths fall back to Scalar.

Add this module beside `lw-ppocr-core`, create a `VectorBackend`, and pass it
to the DET, CLS, and REC constructors. Because the Vector API is incubating,
both compilation and launch require:

```text
--add-modules jdk.incubator.vector
```

The module has an end-to-end Golden test against the same pinned C result as
the Scalar path. Keeping it separate means applications that only use the
Java 8-targeted core do not load or depend on incubator API classes.
