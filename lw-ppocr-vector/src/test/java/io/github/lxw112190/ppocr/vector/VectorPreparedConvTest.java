package io.github.lxw112190.ppocr.vector;

import io.github.lxw112190.ppocr.kernels.PreparedConvBackend;
import io.github.lxw112190.ppocr.kernels.ScalarBackend;
import java.util.Arrays;
import org.junit.Test;
import static org.junit.Assert.*;

public final class VectorPreparedConvTest {
    @Test public void panelsPreserveOffsetsPaddingDilationAndRowShards() {
        int oc = Math.max(16, VectorSupport.F32.length());
        for (int channels : new int[] {3, 24}) for (int kernel : new int[] {2, 3, 5}) {
            for (int stride : new int[] {1, 2}) {
                VectorBackend backend = new VectorBackend();
                int height = 11, width = 13, dilation = kernel == 2 ? 2 : 1;
                int pad = kernel / 2, oh = (height + 2 * pad - dilation * (kernel - 1) - 1) / stride + 1;
                int ow = (width + 2 * pad - dilation * (kernel - 1) - 1) / stride + 1;
                int[] p = {1, channels, height, width, oc, kernel, kernel, stride, stride,
                        dilation, dilation, pad, pad, pad, pad, 1, oh, ow};
                float[] input = values(channels * height * width + 7);
                float[] weights = values(oc * channels * kernel * kernel + 9);
                float[] bias = values(oc + 5);
                float[] actual = new float[oc * oh * ow + 11];
                float[] expected = new float[actual.length];
                Arrays.fill(actual, -31); Arrays.fill(expected, -31);
                new ScalarBackend().conv(input, 7, weights, 9, bias, 5, expected, 11,
                        1, channels, height, width, oc, kernel, kernel, stride, stride,
                        dilation, dilation, pad, pad, pad, pad, 1, oh, ow);
                PreparedConvBackend.Kernel plan = backend.prepareConv(weights, 9, p);
                assertNotNull(plan);
                float[] scratch = new float[plan.scratchFloats()];
                plan.runRows(input, 7, bias, 5, actual, 11, scratch, 0, oh / 2);
                plan.runRows(input, 7, bias, 5, actual, 11, scratch, oh / 2, oh);
                assertArrayEquals(expected, actual, 0.0f);
                long bytes = backend.preparedConvWeightBytes();
                assertNotNull(backend.prepareConv(weights, 9, p));
                assertEquals(bytes, backend.preparedConvWeightBytes());
            }
        }
    }

    @Test public void additionalPackingIsModelWideAndBudgetBounded() {
        VectorBackend backend = new VectorBackend();
        int[] p = {1, 32, 16, 16, 32, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 16, 16};
        float[][] source = {new float[32 * 32 * 9], new float[32 * 32 * 9], new float[32 * 32 * 9]};
        assertNotNull(backend.prepareConv(source[0], 0, p));
        assertNotNull(backend.prepareConv(source[1], 0, p));
        Arrays.fill(source[2], 0.0625f);
        PreparedConvBackend.Kernel scratchPlan = backend.prepareConv(source[2], 0, p);
        assertNotNull(scratchPlan);
        float[] input = values(32 * 16 * 16);
        float[] expected = new float[32 * 16 * 16], actual = new float[expected.length];
        new ScalarBackend().conv(input, 0, source[2], 0, null, 0, expected, 0,
                1, 32, 16, 16, 32, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 16, 16);
        float[] scratch = new float[scratchPlan.scratchFloats()];
        for (int repeat = 0; repeat < 3; repeat++) {
            scratchPlan.runRows(input, 0, null, 0, actual, 0, scratch, 0, 16);
            assertArrayEquals(expected, actual, 0);
        }
        assertTrue(backend.preparedConvWeightBytes() <= 80L * 1024);
        assertNotNull(backend.prepareConv(source[0], 0, p));
        assertEquals(2L * source[0].length * 4, backend.preparedConvWeightBytes());
    }

    @Test public void boundaryTilesPreserveAllTailLengthsAndBiasOffsets() {
        int oc = Math.max(16, VectorSupport.F32.length());
        int channels = 3, height = 5;
        for (int width = 1; width <= 8; width++) for (boolean withBias : new boolean[] {false, true}) {
            VectorBackend backend = new VectorBackend();
            int[] p = {1, channels, height, width, oc, 3, 3, 1, 1,
                    1, 1, 1, 1, 1, 1, 1, height, width};
            float[] input = values(channels * height * width + 7);
            float[] weights = values(oc * channels * 9 + 9);
            float[] bias = withBias ? values(oc + 5) : null;
            float[] actual = new float[oc * height * width + 22];
            float[] expected = new float[actual.length];
            Arrays.fill(actual, -31); Arrays.fill(expected, -31);
            new ScalarBackend().conv(input, 7, weights, 9, bias, 5, expected, 11,
                    1, channels, height, width, oc, 3, 3, 1, 1,
                    1, 1, 1, 1, 1, 1, 1, height, width);
            PreparedConvBackend.Kernel plan = backend.prepareConv(weights, 9, p);
            assertNotNull(plan);
            float[] scratch = new float[plan.scratchFloats()];
            // Reuse the plan and scratch, including row shards and both array guards.
            for (int repeat = 0; repeat < 3; repeat++) {
                plan.runRows(input, 7, bias, 5, actual, 11, scratch, 0, 2);
                plan.runRows(input, 7, bias, 5, actual, 11, scratch, 2, height);
                assertArrayEquals(expected, actual, 0.0f);
            }
        }
    }

    @Test public void paddingDoesNotMultiplyInvalidPixelsByNonfiniteWeights() {
        VectorBackend backend = new VectorBackend();
        int oc = Math.max(16, VectorSupport.F32.length());
        float[] weights = new float[oc * 9];
        for (int c = 0; c < oc; c++) { weights[c * 9] = Float.POSITIVE_INFINITY; weights[c * 9 + 4] = 2; }
        for (int width = 1; width <= 5; width++) {
            int[] p = {1, 1, 1, width, oc, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, width};
            PreparedConvBackend.Kernel plan = backend.prepareConv(weights, 0, p);
            float[] input = new float[width]; Arrays.fill(input, 3);
            float[] output = new float[oc * width];
            plan.runRows(input, 0, null, 0, output, 0, new float[plan.scratchFloats()], 0, 1);
            for (float value : output) assertEquals(6, value, 0);
        }
    }

    private static float[] values(int length) {
        float[] values = new float[length];
        for (int i = 0; i < length; i++) values[i] = (i % 17 - 8) * 0.03125f;
        return values;
    }
}
