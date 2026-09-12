package io.github.lxw112190.ppocr.vector;

import io.github.lxw112190.ppocr.kernels.BinaryOp;
import io.github.lxw112190.ppocr.kernels.ScalarBackend;
import io.github.lxw112190.ppocr.runtime.BinaryPlan;
import io.github.lxw112190.ppocr.runtime.TensorShape;
import org.junit.Assert;
import org.junit.Test;

public final class VectorBackendTest {
    private final VectorBackend vector = new VectorBackend();
    private final ScalarBackend scalar = new ScalarBackend();

    @Test
    public void matchesScalarElementwiseAndMatmul() {
        float[] left = new float[37];
        float[] right = new float[37];
        for (int i = 0; i < left.length; i++) {
            left[i] = i * 0.25f - 3.0f;
            right[i] = i * 0.125f + 1.0f;
        }
        float[] expected = new float[37];
        float[] actual = new float[37];
        scalar.add(left, 0, right, 0, expected, 0, left.length);
        vector.add(left, 0, right, 0, actual, 0, left.length);
        Assert.assertArrayEquals(expected, actual, 0.0f);
        scalar.mul(left, 0, right, 0, expected, 0, left.length);
        vector.mul(left, 0, right, 0, actual, 0, left.length);
        Assert.assertArrayEquals(expected, actual, 0.0f);

        float[] matrixLeft = new float[12];
        float[] matrixRight = new float[78];
        for (int i = 0; i < matrixLeft.length; i++) matrixLeft[i] = i * 0.2f - 1.0f;
        for (int i = 0; i < matrixRight.length; i++) matrixRight[i] = i * 0.03f - 0.5f;
        float[] matrixExpected = new float[26];
        float[] matrixActual = new float[26];
        scalar.matMul(matrixLeft, 0, matrixRight, 0, matrixExpected, 0, 2, 6, 13);
        vector.matMul(matrixLeft, 0, matrixRight, 0, matrixActual, 0, 2, 6, 13);
        Assert.assertArrayEquals(matrixExpected, matrixActual, 0.000001f);
    }

    @Test
    public void matchesScalarErfApproximation() {
        float[] input = values(37, 0.16666667f, -3.0f);
        float[] expected = new float[input.length];
        float[] actual = new float[input.length];
        scalar.erf(input, 0, expected, 0, input.length);
        vector.erf(input, 0, actual, 0, input.length);
        Assert.assertArrayEquals(expected, actual, 0.000001f);
    }

    @Test
    public void matchesScalarContiguousSoftmax() {
        float[] input = values(2 * 37, 0.03125f, -1.0f);
        float[] expected = new float[input.length];
        float[] actual = new float[input.length];
        scalar.softmax(input, 0, expected, 0, 2, 37, 1);
        vector.softmax(input, 0, actual, 0, 2, 37, 1);
        Assert.assertArrayEquals(expected, actual, 0.000001f);
    }

    @Test
    public void matchesScalarContiguousReduceMean() {
        int[] dimensions = {2, 3, 4, 5};
        int[] axes = {2, 3};
        float[] input = values(2 * 3 * 4 * 5, 0.015625f, -0.75f);
        float[] expected = new float[2 * 3];
        float[] actual = new float[expected.length];
        scalar.reduceMean(input, 0, expected, 0, dimensions, axes, true);
        vector.reduceMean(input, 0, actual, 0, dimensions, axes, true);
        Assert.assertArrayEquals(expected, actual, 0.000001f);
    }

    @Test
    public void matchesScalarPointwiseConvolution() {
        float[] input = new float[3 * 23];
        float[] weights = new float[8 * 3];
        float[] bias = new float[8];
        for (int i = 0; i < input.length; i++) input[i] = (i % 11 - 5) * 0.125f;
        for (int i = 0; i < weights.length; i++) weights[i] = (i % 7 - 3) * 0.0625f;
        for (int i = 0; i < bias.length; i++) bias[i] = i * 0.01f;
        float[] expected = new float[8 * 23];
        float[] actual = new float[8 * 23];
        scalar.conv(input, 0, weights, 0, bias, 0, expected, 0,
                1, 3, 1, 23, 8, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 23);
        vector.conv(input, 0, weights, 0, bias, 0, actual, 0,
                1, 3, 1, 23, 8, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 23);
        Assert.assertArrayEquals(expected, actual, 0.000001f);
    }

    @Test
    public void matchesScalarDepthwiseConvolution() {
        float[] input = values(2 * 4 * 7, 0.015625f, -0.75f);
        float[] weights = values(2 * 3 * 3, 0.03125f, -0.25f);
        float[] bias = {0.125f, -0.25f};
        float[] expected = new float[2 * 4 * 7];
        float[] actual = new float[expected.length];
        scalar.conv(input, 0, weights, 0, bias, 0, expected, 0,
                1, 2, 4, 7, 2, 3, 3, 1, 1, 1, 1,
                1, 1, 1, 1, 2, 4, 7);
        vector.conv(input, 0, weights, 0, bias, 0, actual, 0,
                1, 2, 4, 7, 2, 3, 3, 1, 1, 1, 1,
                1, 1, 1, 1, 2, 4, 7);
        Assert.assertArrayEquals(expected, actual, 0.000001f);
    }

    @Test
    public void matchesScalarBroadcastVariants() {
        assertBroadcast(BinaryOp.ADD, new int[] {2, 3, 5}, new int[] {1}, new int[] {2, 3, 5});
        assertBroadcast(BinaryOp.SUB, new int[] {1}, new int[] {2, 3, 5}, new int[] {2, 3, 5});
        assertBroadcast(BinaryOp.MUL, new int[] {2, 3, 5}, new int[] {1, 3, 1}, new int[] {2, 3, 5});
        assertBroadcast(BinaryOp.DIV, new int[] {2, 1, 5}, new int[] {1, 3, 1}, new int[] {2, 3, 5});
        assertBroadcast(BinaryOp.SUB, new int[] {2, 3, 1}, new int[] {1, 1, 5}, new int[] {2, 3, 5});
    }

    private void assertBroadcast(BinaryOp operation, int[] leftShape, int[] rightShape,
                                 int[] outputShape) {
        float[] left = values(length(leftShape), 0.17f, 0.5f);
        float[] right = values(length(rightShape), 0.11f, 1.25f);
        float[] expected = new float[length(outputShape)];
        float[] actual = new float[expected.length];
        BinaryPlan plan = new BinaryPlan(new TensorShape(leftShape), new TensorShape(rightShape),
                new TensorShape(outputShape));
        scalar.binary(operation, left, 0, right, 0, expected, 0, plan);
        vector.binary(operation, left, 0, right, 0, actual, 0, plan);
        Assert.assertArrayEquals(operation.name(), expected, actual, 0.000001f);
    }

    private static int length(int[] shape) {
        int result = 1;
        for (int dimension : shape) result *= dimension;
        return result;
    }

    private static float[] values(int length, float step, float base) {
        float[] result = new float[length];
        for (int i = 0; i < length; i++) result[i] = base + i * step;
        return result;
    }
}
