package io.github.lxw112190.ppocr.kernels;

import org.junit.Assert;
import org.junit.Test;

public final class ScalarBackendTest {
    private final ScalarBackend backend = new ScalarBackend();

    @Test
    public void computesOneByOneConvolution() {
        float[] output = new float[4];
        backend.conv(new float[] {1, 2, 3, 4}, 0,
                new float[] {2}, 0,
                new float[] {1}, 0,
                output, 0,
                1, 1, 2, 2, 1,
                1, 1,
                1, 1,
                1, 1,
                0, 0,
                1, 2, 2);
        Assert.assertArrayEquals(new float[] {3, 5, 7, 9}, output, 0.0f);
    }

    @Test
    public void computesStableSoftmax() {
        float[] output = new float[3];
        backend.softmax(new float[] {1000, 1001, 1002}, 0, output, 0, 1, 3, 1);
        Assert.assertEquals(1.0f, output[0] + output[1] + output[2], 0.00001f);
        Assert.assertTrue(output[2] > output[1]);
        Assert.assertTrue(output[1] > output[0]);
    }

    @Test
    public void computesReduceMeanConcatAndSlice() {
        float[] reduced = new float[2];
        backend.reduceMean(new float[] {1, 3, 5, 7}, 0, reduced, 0,
                new int[] {2, 2}, new int[] {1}, false);
        Assert.assertArrayEquals(new float[] {2, 6}, reduced, 0.0f);

        float[] concatenated = new float[4];
        backend.concat(new float[][] {{1, 2}, {3, 4}}, new int[] {0, 0}, concatenated, 0,
                new int[] {1, 2}, 1, new int[] {2, 2});
        Assert.assertArrayEquals(new float[] {1, 2, 3, 4}, concatenated, 0.0f);

        float[] sliced = new float[4];
        backend.slice(new float[] {1, 2, 3, 4, 5, 6}, 0, sliced, 0,
                new int[] {2, 3}, new int[] {1}, new int[] {1}, new int[] {1});
        Assert.assertArrayEquals(new float[] {2, 3, 5, 6}, sliced, 0.0f);
    }
}
