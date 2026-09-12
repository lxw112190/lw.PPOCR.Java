package io.github.lxw112190.ppocr.kernels;

/** Scalar/SIMD-independent kernel contract used by the graph executor. */
public interface KernelBackend {
    void add(float[] left, int leftOffset, float[] right, int rightOffset,
             float[] output, int outputOffset, int length);

    void mul(float[] left, int leftOffset, float[] right, int rightOffset,
             float[] output, int outputOffset, int length);

    void div(float[] left, int leftOffset, float[] right, int rightOffset,
             float[] output, int outputOffset, int length);

    void sub(float[] left, int leftOffset, float[] right, int rightOffset,
             float[] output, int outputOffset, int length);

    void relu(float[] input, int inputOffset, float[] output, int outputOffset, int length);

    void matMul(float[] left, int leftOffset, float[] right, int rightOffset,
                float[] output, int outputOffset, int rows, int inner, int columns);
}
