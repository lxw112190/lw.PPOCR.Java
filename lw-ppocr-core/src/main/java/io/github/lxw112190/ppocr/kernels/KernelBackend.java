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

    void sigmoid(float[] input, int inputOffset, float[] output, int outputOffset, int length);

    void erf(float[] input, int inputOffset, float[] output, int outputOffset, int length);

    void hardSigmoid(float[] input, int inputOffset, float[] output, int outputOffset,
                     int length, float alpha, float beta);

    void sqrt(float[] input, int inputOffset, float[] output, int outputOffset, int length);

    void pow(float[] left, int leftOffset, float[] right, int rightOffset,
             float[] output, int outputOffset, int length);

    void reduceMean(float[] input, int inputOffset, float[] output, int outputOffset,
                    int[] inputDimensions, int[] axes, boolean keepDimensions);

    void concat(float[][] inputs, int[] inputOffsets, float[] output, int outputOffset,
                int[] inputDimensions, int axis, int[] axisSizes);

    void slice(float[] input, int inputOffset, float[] output, int outputOffset,
               int[] inputDimensions, int[] starts, int[] axes, int[] steps);

    void matMul(float[] left, int leftOffset, float[] right, int rightOffset,
                float[] output, int outputOffset, int rows, int inner, int columns);

    void conv(float[] input, int inputOffset, float[] weights, int weightOffset,
              float[] bias, int biasOffset, float[] output, int outputOffset,
              int batch, int channels, int height, int width, int outputChannels,
              int kernelHeight, int kernelWidth, int strideHeight, int strideWidth,
              int dilationHeight, int dilationWidth, int padTop, int padLeft,
              int groups, int outputHeight, int outputWidth);

    void softmax(float[] input, int inputOffset, float[] output, int outputOffset,
                 int outer, int axisLength, int inner);
}
