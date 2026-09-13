package io.github.lxw112190.ppocr.kernels;

import io.github.lxw112190.ppocr.runtime.BinaryPlan;
import io.github.lxw112190.ppocr.runtime.TensorShape;
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
    public void computesMultiChannelPointwiseConvolution() {
        float[] output = new float[8];
        backend.conv(new float[] {1, 2, 3, 4, 5, 6, 7, 8}, 0,
                new float[] {1, 2, -1, 0.5f}, 0,
                new float[] {0, 1}, 0, output, 0,
                1, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1,
                0, 0, 1, 2, 2);
        Assert.assertArrayEquals(new float[] {11, 14, 17, 20, 2.5f, 2, 1.5f, 1},
                output, 0.0f);
    }

    @Test
    public void computesPaddedDepthwiseConvolution() {
        float[] output = new float[8];
        float[] weights = {
                1, 1, 1, 1, 1, 1, 1, 1, 1,
                2, 2, 2, 2, 2, 2, 2, 2, 2
        };
        backend.conv(new float[] {1, 2, 3, 4, 5, 6, 7, 8}, 0,
                weights, 0, new float[] {0, 1}, 0, output, 0,
                1, 2, 2, 2, 2, 3, 3, 1, 1, 1, 1,
                1, 1, 2, 2, 2);
        Assert.assertArrayEquals(new float[] {10, 10, 10, 10, 53, 53, 53, 53},
                output, 0.0f);
    }

    @Test
    public void computesMatrixMultiplicationWithContiguousRows() {
        float[] output = new float[4];
        backend.matMul(new float[] {1, 2, 3, 4, 5, 6}, 0,
                new float[] {7, 8, 9, 10, 11, 12}, 0,
                output, 0, 2, 3, 2);
        Assert.assertArrayEquals(new float[] {58, 64, 139, 154}, output, 0.0f);
    }

    @Test
    public void fusesProjectionArgMaxAndWinningSoftmaxProbability() {
        int rows = 2;
        int inner = 3;
        int columns = 4;
        float[] activations = {1, 2, -1, -2, 1, 3};
        float[] weights = {
                0.5f, -1.0f, 0.25f, 2.0f,
                1.5f, 0.5f, -0.5f, 0.25f,
                -0.5f, 1.0f, 0.75f, -1.0f
        };
        float[] bias = {0.1f, -0.2f, 0.3f, 0.4f};
        float[] dense = new float[rows * columns];
        backend.matMul(activations, 0, weights, 0, dense, 0, rows, inner, columns);
        for (int row = 0; row < rows; row++) {
            for (int column = 0; column < columns; column++) dense[row * columns + column] += bias[column];
        }
        backend.softmax(dense, 0, dense, 0, rows, columns, 1);

        int[] ids = new int[rows];
        float[] logits = new float[rows];
        float[] probabilities = new float[rows];
        backend.projectionArgMax(activations, 0, weights, 0, bias, 0,
                rows, inner, columns, ids, logits, probabilities,
                new float[Math.min(rows, 4) * columns]);
        for (int row = 0; row < rows; row++) {
            int expected = 0;
            for (int column = 1; column < columns; column++) {
                if (dense[row * columns + column] > dense[row * columns + expected]) expected = column;
            }
            Assert.assertEquals(expected, ids[row]);
            Assert.assertEquals(dense[row * columns + expected], probabilities[row], 0.0f);
        }
    }

    @Test
    public void computesLeftAndRightScalarBroadcasts() {
        TensorShape vector = new TensorShape(4);
        TensorShape scalar = new TensorShape(1);
        float[] output = new float[4];
        backend.binary(BinaryOp.MUL, new float[] {1, 2, 3, 4}, 0,
                new float[] {2}, 0, output, 0, new BinaryPlan(vector, scalar, vector));
        Assert.assertArrayEquals(new float[] {2, 4, 6, 8}, output, 0.0f);

        backend.binary(BinaryOp.SUB, new float[] {10}, 0,
                new float[] {1, 2, 3, 4}, 0, output, 0,
                new BinaryPlan(scalar, vector, vector));
        Assert.assertArrayEquals(new float[] {9, 8, 7, 6}, output, 0.0f);
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

    @Test
    public void computesPoolResizeAndConvTranspose() {
        float[] pooled = new float[1];
        backend.pool(new float[] {1, 2, 3, 4}, 0, pooled, 0,
                1, 1, 2, 2, 2, 2, 1, 1, 0, 0, 1, 1, true, false);
        Assert.assertArrayEquals(new float[] {4}, pooled, 0.0f);

        float[] resized = new float[16];
        backend.resizeNearest(new float[] {1, 2, 3, 4}, 0, resized, 0,
                1, 1, 2, 2, 4, 4, 2, 2);
        Assert.assertArrayEquals(new float[] {
                1, 1, 2, 2, 1, 1, 2, 2, 3, 3, 4, 4, 3, 3, 4, 4
        }, resized, 0.0f);

        float[] transposed = new float[4];
        backend.convTranspose(new float[] {1}, 0, new float[] {2, 3, 4, 5}, 0,
                new float[] {1}, 0, transposed, 0,
                1, 1, 1, 1, 1, 2, 2, 2, 2, 1, 1, 0, 0, 1, 2, 2);
        Assert.assertArrayEquals(new float[] {3, 4, 5, 6}, transposed, 0.0f);
    }

    @Test
    public void computesBatchNormalizationPerChannel() {
        float[] output = new float[4];
        backend.batchNormalization(new float[] {1, 3, 5, 7}, 0,
                new float[] {1, 1}, 0, new float[] {0, 0}, 0,
                new float[] {1, 2}, 0, new float[] {0, 0}, 0, 1.0f,
                output, 0, new int[] {1, 2, 2});
        Assert.assertArrayEquals(new float[] {0, 2, 3, 5}, output, 0.00001f);
    }
}
