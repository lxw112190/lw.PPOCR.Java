# lw-ppocr-vector

Optional JDK 25 Vector API backend for the pure-Java PP-OCR runtime. It
accelerates supported elementwise, MatMul, and pointwise-convolution kernels
and falls back to the Scalar backend for the remaining operators.

Add this module beside `lw-ppocr-core`, create a `VectorBackend`, and pass it
to the DET, CLS, and REC constructors. Because the Vector API is incubating,
both compilation and launch require:

```text
--add-modules jdk.incubator.vector
```

The module has an end-to-end Golden test against the same pinned C result as
the Scalar path. Keeping it separate means applications that only use the
Java 8-targeted core do not load or depend on incubator API classes.
