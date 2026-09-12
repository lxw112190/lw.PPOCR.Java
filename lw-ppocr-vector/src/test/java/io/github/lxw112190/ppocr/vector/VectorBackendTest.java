package io.github.lxw112190.ppocr.vector;

import io.github.lxw112190.ppocr.kernels.BinaryOp;
import io.github.lxw112190.ppocr.kernels.ScalarBackend;
import io.github.lxw112190.ppocr.runtime.BinaryPlan;
import io.github.lxw112190.ppocr.runtime.TensorShape;
import java.util.Arrays;
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
        int matrixColumns = 137;
        float[] matrixRight = new float[6 * matrixColumns];
        for (int i = 0; i < matrixLeft.length; i++) matrixLeft[i] = i * 0.2f - 1.0f;
        for (int i = 0; i < matrixRight.length; i++) matrixRight[i] = i * 0.03f - 0.5f;
        float[] matrixExpected = new float[2 * matrixColumns];
        float[] matrixActual = new float[2 * matrixColumns];
        scalar.matMul(matrixLeft, 0, matrixRight, 0, matrixExpected, 0,
                2, 6, matrixColumns);
        vector.matMul(matrixLeft, 0, matrixRight, 0, matrixActual, 0,
                2, 6, matrixColumns);
        Assert.assertArrayEquals(matrixExpected, matrixActual, 0.000001f);
    }

    @Test
    public void matchesScalarFourRowBlockedMatmulWithOffsetsAndTails() {
        int rows = 5;
        int inner = 7;
        int columns = 139;
        int leftOffset = 3;
        int rightOffset = 5;
        int outputOffset = 7;
        float[] left = values(leftOffset + rows * inner + 2, 0.03125f, -0.75f);
        float[] right = values(rightOffset + inner * columns + 2, 0.0078125f, -0.5f);
        float[] expected = new float[outputOffset + rows * columns + 2];
        float[] actual = new float[expected.length];
        Arrays.fill(expected, -17.0f);
        Arrays.fill(actual, -17.0f);

        scalar.matMul(left, leftOffset, right, rightOffset, expected, outputOffset,
                rows, inner, columns);
        vector.matMul(left, leftOffset, right, rightOffset, actual, outputOffset,
                rows, inner, columns);

        Assert.assertArrayEquals(expected, actual, 0.0f);
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
    public void fusedGeluMatchesFiveVectorOperationsWithOffsetsAndTail() {
        int length = 37;
        int inputOffset = 3;
        int outputOffset = 5;
        float divisor = 1.4142135f;
        float addend = 1.0f;
        float multiplier = 0.5f;
        float[] input = values(inputOffset + length + 2, 0.125f, -2.5f);
        float[] normalized = new float[length];
        float[] erf = new float[length];
        float[] shifted = new float[length];
        float[] product = new float[length];
        float[] expected = new float[outputOffset + length + 2];
        float[] actual = new float[expected.length];
        Arrays.fill(expected, -17.0f);
        Arrays.fill(actual, -17.0f);
        BinaryPlan scalar = new BinaryPlan(new TensorShape(length), new TensorShape(1),
                new TensorShape(length));

        vector.binary(BinaryOp.DIV, input, inputOffset, new float[] {divisor}, 0,
                normalized, 0, scalar);
        vector.erf(normalized, 0, erf, 0, length);
        vector.binary(BinaryOp.ADD, erf, 0, new float[] {addend}, 0,
                shifted, 0, scalar);
        vector.mul(input, inputOffset, shifted, 0, product, 0, length);
        vector.binary(BinaryOp.MUL, product, 0, new float[] {multiplier}, 0,
                expected, outputOffset, scalar);
        vector.gelu(input, inputOffset, actual, outputOffset, length,
                divisor, addend, multiplier);

        Assert.assertArrayEquals(expected, actual, 0.0f);
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
    public void matchesScalarGroupedPointwiseConvolutionWithOffsets() {
        int batch = 2;
        int channels = 6;
        int plane = 23;
        int groups = 2;
        int outputChannels = 34;
        int inputOffset = 5;
        int weightOffset = 7;
        int biasOffset = 3;
        int outputOffset = 11;
        float[] input = values(inputOffset + batch * channels * plane + 4,
                0.001953125f, -0.375f);
        float[] weights = values(weightOffset + outputChannels * channels / groups + 4,
                0.00390625f, -0.25f);
        float[] bias = values(biasOffset + outputChannels + 4, 0.015625f, -0.125f);
        float[] expected = new float[outputOffset + batch * outputChannels * plane + 7];
        float[] actual = new float[expected.length];
        Arrays.fill(expected, -9.0f);
        Arrays.fill(actual, -9.0f);
        scalar.conv(input, inputOffset, weights, weightOffset, bias, biasOffset,
                expected, outputOffset, batch, channels, 1, plane, outputChannels,
                1, 1, 1, 1, 1, 1, 0, 0, groups, 1, plane);
        vector.conv(input, inputOffset, weights, weightOffset, bias, biasOffset,
                actual, outputOffset, batch, channels, 1, plane, outputChannels,
                1, 1, 1, 1, 1, 1, 0, 0, groups, 1, plane);
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
    public void matchesScalarGroupedGeneralConvolution() {
        int outputChannels = 18;
        float[] input = values(4 * 5 * 13, 0.00390625f, -0.5f);
        float[] weights = values(outputChannels * 2 * 3 * 3, 0.0078125f, -0.25f);
        float[] bias = values(outputChannels, 0.03125f, -0.0625f);
        float[] expected = new float[outputChannels * 5 * 13];
        float[] actual = new float[expected.length];
        scalar.conv(input, 0, weights, 0, bias, 0, expected, 0,
                1, 4, 5, 13, outputChannels, 3, 3, 1, 1, 1, 1,
                1, 1, 1, 1, 2, 5, 13);
        vector.conv(input, 0, weights, 0, bias, 0, actual, 0,
                1, 4, 5, 13, outputChannels, 3, 3, 1, 1, 1, 1,
                1, 1, 1, 1, 2, 5, 13);
        Assert.assertArrayEquals(expected, actual, 0.000001f);
    }

    @Test
    public void matchesScalarSameSizeTwoByTwoConvolutionWithOffsetsAndTails() {
        int batch = 2;
        int channels = 5;
        int height = 5;
        int width = 37;
        int outputChannels = 16;
        int inputOffset = 5;
        int weightOffset = 7;
        int biasOffset = 3;
        int outputOffset = 11;
        float[] input = values(inputOffset + batch * channels * height * width + 3,
                0.001953125f, -0.375f);
        float[] weights = values(weightOffset + outputChannels * channels * 4 + 3,
                0.0078125f, -0.25f);
        float[] bias = values(biasOffset + outputChannels + 3, 0.03125f, -0.0625f);
        float[] expected = new float[outputOffset
                + batch * outputChannels * height * width + 3];
        float[] actual = new float[expected.length];
        Arrays.fill(expected, -17.0f);
        Arrays.fill(actual, -17.0f);

        scalar.conv(input, inputOffset, weights, weightOffset, bias, biasOffset,
                expected, outputOffset, batch, channels, height, width, outputChannels,
                2, 2, 1, 1, 1, 1, 0, 0, 1, 1,
                1, height, width);
        vector.conv(input, inputOffset, weights, weightOffset, bias, biasOffset,
                actual, outputOffset, batch, channels, height, width, outputChannels,
                2, 2, 1, 1, 1, 1, 0, 0, 1, 1,
                1, height, width);

        Assert.assertArrayEquals(expected, actual, 0.0f);
    }

    @Test
    public void matchesScalarStrideTwoConvolution() {
        int outputChannels = 34;
        float[] input = values(4 * 7 * 15, 0.001953125f, -0.375f);
        float[] weights = values(outputChannels * 2 * 3 * 3, 0.0078125f, -0.25f);
        float[] bias = values(outputChannels, 0.03125f, -0.0625f);
        float[] expected = new float[outputChannels * 4 * 8];
        float[] actual = new float[expected.length];
        scalar.conv(input, 0, weights, 0, bias, 0, expected, 0,
                1, 4, 7, 15, outputChannels, 3, 3, 2, 2, 1, 1,
                1, 1, 1, 1, 2, 4, 8);
        vector.conv(input, 0, weights, 0, bias, 0, actual, 0,
                1, 4, 7, 15, outputChannels, 3, 3, 2, 2, 1, 1,
                1, 1, 1, 1, 2, 4, 8);
        Assert.assertArrayEquals(expected, actual, 0.000001f);
    }

    @Test
    public void matchesScalarLowChannelStrideTwoConvolutionWithOffsetsAndSpatialTails() {
        int batch = 2;
        int channels = 3;
        int height = 7;
        int width = 37;
        int outputChannels = 16;
        int outputHeight = (height + 1) / 2;
        int outputWidth = (width + 1) / 2;
        int inputOffset = 5;
        int weightOffset = 7;
        int biasOffset = 3;
        int outputOffset = 11;
        float[] input = values(inputOffset + batch * channels * height * width + 3,
                0.001953125f, -0.375f);
        float[] weights = values(weightOffset + outputChannels * channels * 9 + 3,
                0.0078125f, -0.25f);
        float[] bias = values(biasOffset + outputChannels + 3, 0.03125f, -0.0625f);
        float[] expected = new float[outputOffset
                + batch * outputChannels * outputHeight * outputWidth + 3];
        float[] actual = new float[expected.length];
        Arrays.fill(expected, -17.0f);
        Arrays.fill(actual, -17.0f);

        scalar.conv(input, inputOffset, weights, weightOffset, bias, biasOffset,
                expected, outputOffset, batch, channels, height, width, outputChannels,
                3, 3, 2, 2, 1, 1, 1, 1, 1, 1,
                1, outputHeight, outputWidth);
        vector.conv(input, inputOffset, weights, weightOffset, bias, biasOffset,
                actual, outputOffset, batch, channels, height, width, outputChannels,
                3, 3, 2, 2, 1, 1, 1, 1, 1, 1,
                1, outputHeight, outputWidth);

        Assert.assertArrayEquals(expected, actual, 0.0f);
    }

    @Test
    public void matchesScalarTwoByTwoStrideOneMaxPoolWithOffsetsAndTails() {
        int batch = 2;
        int channels = 3;
        int height = 5;
        int width = 37;
        int inputOffset = 5;
        int outputOffset = 7;
        float[] input = values(inputOffset + batch * channels * height * width + 3,
                0.03125f, -4.0f);
        for (int i = inputOffset; i < input.length; i += 11) input[i] = -input[i];
        input[inputOffset + 17] = Float.NaN;
        input[inputOffset + 18] = Float.POSITIVE_INFINITY;
        input[inputOffset + 36] = -0.0f;
        input[inputOffset + 37] = 0.0f;
        float[] expected = new float[outputOffset + batch * channels * height * width + 3];
        float[] actual = new float[expected.length];
        Arrays.fill(expected, -17.0f);
        Arrays.fill(actual, -17.0f);

        scalar.pool(input, inputOffset, expected, outputOffset,
                batch, channels, height, width, 2, 2, 1, 1,
                0, 0, 1, 1, height, width, true, false);
        vector.pool(input, inputOffset, actual, outputOffset,
                batch, channels, height, width, 2, 2, 1, 1,
                0, 0, 1, 1, height, width, true, false);

        Assert.assertArrayEquals(expected, actual, 0.0f);
        for (int i = 0; i < expected.length; i++) {
            Assert.assertEquals("float bits at " + i, Float.floatToIntBits(expected[i]),
                    Float.floatToIntBits(actual[i]));
        }
    }

    @Test
    public void fallsBackForOtherPoolConfigurations() {
        float[] input = values(2 * 5 * 7, 0.0625f, -1.0f);
        float[] expected = new float[2 * 3 * 4];
        float[] actual = new float[expected.length];
        scalar.pool(input, 0, expected, 0, 1, 2, 5, 7,
                2, 2, 2, 2, 0, 0, 0, 0, 3, 4, false, true);
        vector.pool(input, 0, actual, 0, 1, 2, 5, 7,
                2, 2, 2, 2, 0, 0, 0, 0, 3, 4, false, true);
        Assert.assertArrayEquals(expected, actual, 0.0f);
    }

    @Test
    public void matchesScalarTwoByTwoTransposeConvolution() {
        int batch = 2;
        int inputChannels = 3;
        int inputHeight = 3;
        int inputWidth = 17;
        int outputChannels = 5;
        int outputHeight = inputHeight * 2;
        int outputWidth = inputWidth * 2;
        float[] input = values(batch * inputChannels * inputHeight * inputWidth,
                0.001953125f, -0.25f);
        float[] weights = values(inputChannels * outputChannels * 4, 0.0078125f, -0.125f);
        float[] bias = values(outputChannels, 0.03125f, -0.0625f);
        float[] expected = new float[batch * outputChannels * outputHeight * outputWidth];
        float[] actual = new float[expected.length];
        scalar.convTranspose(input, 0, weights, 0, bias, 0, expected, 0,
                batch, inputChannels, inputHeight, inputWidth, outputChannels,
                2, 2, 2, 2, 1, 1, 0, 0, 1, outputHeight, outputWidth);
        vector.convTranspose(input, 0, weights, 0, bias, 0, actual, 0,
                batch, inputChannels, inputHeight, inputWidth, outputChannels,
                2, 2, 2, 2, 1, 1, 0, 0, 1, outputHeight, outputWidth);
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
