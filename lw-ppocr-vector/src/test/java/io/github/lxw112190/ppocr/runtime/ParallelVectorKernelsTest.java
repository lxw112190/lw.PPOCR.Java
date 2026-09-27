package io.github.lxw112190.ppocr.runtime;

import io.github.lxw112190.ppocr.vector.VectorBackend;
import io.github.lxw112190.ppocr.kernels.ConvEpilogue;
import java.util.Arrays;
import org.junit.Assert;
import org.junit.Test;

public final class ParallelVectorKernelsTest {
    @Test public void fusedGeluShardsPreserveFlatActivationTail() {
        int channels = 512, oc = 40, plane = 133;
        float[] input = values(5 + channels * plane), weights = values(7 + channels * oc);
        float[] expected = new float[11 + oc * plane + 3], actual = new float[expected.length];
        VectorBackend backend = new VectorBackend();
        ConvEpilogue ep = new ConvEpilogue(null, null, null, null, false,
                ConvEpilogue.GELU, 1.41421356f, 1, 0.5f, 0, 0);
        backend.convEpilogue(input, 5, weights, 7, null, 0, expected, 11, 1, channels, 1, plane,
                oc, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, plane, ep, 0, null, 0);
        ParallelKernels parallel = new ParallelKernels(backend);
        parallel.setParallelism(4);
        parallel.conv(input, 5, weights, 7, null, 0, actual, 11, 1, channels, 1, plane,
                oc, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, plane, ep, null, 0);
        Assert.assertArrayEquals(expected, actual, 0);
    }

    @Test public void convolutionShardsMatchSerialVectorBits() {
        assertConv(24, 48, 24, 320, 3, 2, 1);
        assertConv(32, 32, 96, 103, 3, 2, 1);
        assertConv(24, 40, 80, 103, 1, 1, 1);
        assertConv(24, 24, 100, 103, 3, 1, 24);
        assertConv(24, 40, 80, 103, 3, 1, 1);
    }

    private static void assertConv(int channels, int oc, int h, int w, int kernel, int stride, int groups) {
        int pad = kernel == 1 ? 0 : 1, oh = (h + stride - 1) / stride, ow = (w + stride - 1) / stride;
        float[] input = values(5 + channels * h * w);
        float[] weights = values(7 + oc * (channels / groups) * kernel * kernel);
        float[] bias = values(3 + oc);
        float[] expected = new float[11 + oc * oh * ow + 3], actual = new float[expected.length];
        Arrays.fill(expected, -17); Arrays.fill(actual, -17);
        VectorBackend backend = new VectorBackend();
        backend.conv(input, 5, weights, 7, bias, 3, expected, 11, 1, channels, h, w,
                oc, kernel, kernel, stride, stride, 1, 1, pad, pad, pad, pad, groups, oh, ow);
        ParallelKernels parallel = new ParallelKernels(backend);
        parallel.setParallelism(4);
        parallel.conv(input, 5, weights, 7, bias, 3, actual, 11, 1, channels, h, w,
                oc, kernel, kernel, stride, stride, 1, 1, pad, pad, pad, pad, groups, oh, ow);
        Assert.assertArrayEquals(expected, actual, 0);
    }

    private static float[] values(int length) {
        float[] result = new float[length];
        for (int i = 0; i < length; i++) result[i] = (i % 97 - 48) * 0.003f;
        return result;
    }
}
