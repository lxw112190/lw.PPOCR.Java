package io.github.lxw112190.ppocr.vector;

import io.github.lxw112190.ppocr.kernels.ScalarBackend;
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
}
