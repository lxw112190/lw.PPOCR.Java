package io.github.lxw112190.ppocr.kernels;

/** Readable reference kernels; optimized backends must preserve their semantics. */
public final class ScalarBackend implements KernelBackend {
    @Override
    public void add(float[] left, int leftOffset, float[] right, int rightOffset,
                    float[] output, int outputOffset, int length) {
        binary(left, leftOffset, right, rightOffset, output, outputOffset, length, 0);
    }

    @Override
    public void mul(float[] left, int leftOffset, float[] right, int rightOffset,
                    float[] output, int outputOffset, int length) {
        binary(left, leftOffset, right, rightOffset, output, outputOffset, length, 1);
    }

    @Override
    public void div(float[] left, int leftOffset, float[] right, int rightOffset,
                    float[] output, int outputOffset, int length) {
        binary(left, leftOffset, right, rightOffset, output, outputOffset, length, 2);
    }

    @Override
    public void sub(float[] left, int leftOffset, float[] right, int rightOffset,
                    float[] output, int outputOffset, int length) {
        binary(left, leftOffset, right, rightOffset, output, outputOffset, length, 3);
    }

    @Override
    public void relu(float[] input, int inputOffset, float[] output, int outputOffset, int length) {
        for (int i = 0; i < length; i++) {
            output[outputOffset + i] = Math.max(0.0f, input[inputOffset + i]);
        }
    }

    @Override
    public void matMul(float[] left, int leftOffset, float[] right, int rightOffset,
                       float[] output, int outputOffset, int rows, int inner, int columns) {
        for (int row = 0; row < rows; row++) {
            for (int column = 0; column < columns; column++) {
                float sum = 0.0f;
                for (int k = 0; k < inner; k++) {
                    sum += left[leftOffset + row * inner + k] * right[rightOffset + k * columns + column];
                }
                output[outputOffset + row * columns + column] = sum;
            }
        }
    }

    private static void binary(float[] left, int leftOffset, float[] right, int rightOffset,
                               float[] output, int outputOffset, int length, int operation) {
        for (int i = 0; i < length; i++) {
            float a = left[leftOffset + i];
            float b = right[rightOffset + i];
            switch (operation) {
                case 0: output[outputOffset + i] = a + b; break;
                case 1: output[outputOffset + i] = a * b; break;
                case 2: output[outputOffset + i] = a / b; break;
                case 3: output[outputOffset + i] = a - b; break;
                default: throw new AssertionError("unknown scalar operation");
            }
        }
    }
}
