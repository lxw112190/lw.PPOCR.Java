package io.github.lxw112190.ppocr.runtime;

import io.github.lxw112190.ppocr.kernels.ScalarBackend;
import java.util.Arrays;
import java.util.concurrent.atomic.AtomicInteger;
import org.junit.Assert;
import org.junit.Test;

public final class ParallelKernelsTest {
    @Test public void joinsFailuresAndCanBeReused() {
        OperatorParallelExecutor executor = new OperatorParallelExecutor();
        final AtomicInteger completed = new AtomicInteger();
        try {
            executor.run(4, new OperatorParallelExecutor.Action() {
                public void run(int shard, int shards) {
                    completed.incrementAndGet();
                    if (shard == 2) throw new IllegalArgumentException("test failure");
                }
            });
            Assert.fail("failure must reach caller");
        } catch (IllegalArgumentException expected) {
            Assert.assertEquals("test failure", expected.getMessage());
        }
        Assert.assertEquals(4, completed.get());
        executor.run(2, new OperatorParallelExecutor.Action() {
            public void run(int shard, int shards) { completed.incrementAndGet(); }
        });
        Assert.assertEquals(6, completed.get());
    }

    @Test public void denseAndDepthwiseShardsPreserveOffsetsAndTails() {
        for (int groups : new int[] {1, 24}) {
            int channels = 24, oc = groups == 1 ? 40 : 24, h = 100, w = 103;
            float[] a = values(5 + channels * h * w);
            float[] weights = values(7 + oc * (channels / groups) * 9);
            float[] bias = values(3 + oc);
            float[] expected = new float[11 + oc * h * w + 3];
            float[] actual = new float[expected.length];
            Arrays.fill(expected, -17); Arrays.fill(actual, -17);
            ScalarBackend scalar = new ScalarBackend();
            scalar.conv(a, 5, weights, 7, bias, 3, expected, 11, 1, channels,
                    h, w, oc, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, groups, h, w);
            ParallelKernels parallel = new ParallelKernels(scalar);
            parallel.setParallelism(4);
            for (int repeat = 0; repeat < 2; repeat++) {
                parallel.conv(a, 5, weights, 7, bias, 3, actual, 11, 1, channels,
                        h, w, oc, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, groups, h, w);
                Assert.assertArrayEquals(expected, actual, 0);
            }
        }
    }

    @Test public void matrixRowShardsPreserveReductionOrder() {
        int rows = 35, inner = 192, columns = 321;
        float[] a = values(5 + rows * inner), b = values(7 + inner * columns);
        float[] expected = new float[11 + rows * columns + 3], actual = new float[expected.length];
        ScalarBackend scalar = new ScalarBackend();
        scalar.matMul(a, 5, b, 7, expected, 11, rows, inner, columns);
        ParallelKernels parallel = new ParallelKernels(scalar);
        parallel.setParallelism(4);
        parallel.matMul(a, 5, b, 7, actual, 11, rows, inner, columns);
        Assert.assertArrayEquals(expected, actual, 0);
    }

    private static float[] values(int length) {
        float[] result = new float[length];
        for (int i = 0; i < length; i++) result[i] = (i % 97 - 48) * 0.001f;
        return result;
    }
}
