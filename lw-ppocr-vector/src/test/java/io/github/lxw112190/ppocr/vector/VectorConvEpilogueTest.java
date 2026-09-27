package io.github.lxw112190.ppocr.vector;

import io.github.lxw112190.ppocr.kernels.ConvEpilogue;
import java.util.Arrays;
import org.junit.Assert;
import org.junit.Test;

public final class VectorConvEpilogueTest {
    @Test public void pointwiseEpiloguesPreserveUnfusedBitsWithOffsetsGroupsAndTails() {
        for (int activation : new int[] {0, 1, 2, 3, 4}) {
            for (int plane : new int[] {32, 33}) {
                assertEpilogue(activation, plane, 1);
                assertEpilogue(activation, plane, 3);
            }
        }
    }

    private static void assertEpilogue(int activation, int plane, int groups) {
        int batch = 2, channels = 24, outputChannels = 36, length = batch * outputChannels * plane;
        float[] input = values(3 + batch * channels * plane), weights = values(5 + outputChannels * channels / groups);
        float[] bias = values(2 + outputChannels), residual = values(11 + length);
        float[] expected = new float[7 + length + 3], actual = new float[expected.length];
        Arrays.fill(expected, -17); Arrays.fill(actual, -17);
        float[] mean = values(outputChannels), factor = values(outputChannels), bnBias = values(outputChannels);
        float[] postBias = values(outputChannels);
        ConvEpilogue epilogue = new ConvEpilogue(mean, factor, bnBias, postBias, false,
                activation, 1.41421356f, 1, 0.5f, 1.0f / 6, 0.5f);
        VectorBackend backend = new VectorBackend();
        backend.conv(input, 3, weights, 5, bias, 2, expected, 7, batch, channels, 1, plane,
                outputChannels, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, groups, 1, plane);
        for (int n = 0; n < batch; n++) for (int c = 0; c < outputChannels; c++) {
            for (int i = 0; i < plane; i++) {
                int index = (n * outputChannels + c) * plane + i;
                expected[7 + index] = (expected[7 + index] - mean[c]) * factor[c] + bnBias[c];
                expected[7 + index] += postBias[c];
            }
        }
        backend.add(expected, 7, residual, 11, expected, 7, length);
        switch (activation) {
            case 1: backend.relu(expected, 7, expected, 7, length); break;
            case 2: backend.gelu(expected, 7, expected, 7, length, epilogue.divisor, 1, 0.5f); break;
            case 3:
                float[] gates = new float[length];
                backend.hardSigmoid(expected, 7, gates, 0, length, epilogue.alpha, epilogue.beta);
                backend.mul(expected, 7, gates, 0, expected, 7, length); break;
            case 4: backend.hardSigmoid(expected, 7, expected, 7, length, epilogue.alpha, epilogue.beta); break;
            default: break;
        }
        backend.convEpilogue(input, 3, weights, 5, bias, 2, actual, 7, batch, channels, 1, plane,
                outputChannels, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, groups, 1, plane,
                epilogue, 0, residual, 11);
        Assert.assertArrayEquals("activation=" + activation + ",plane=" + plane + ",groups=" + groups,
                expected, actual, 0);
    }

    private static float[] values(int length) {
        float[] result = new float[length];
        for (int i = 0; i < length; i++) result[i] = (i % 97 - 48) * 0.013f;
        return result;
    }
}
