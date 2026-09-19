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
    public void fusedGeluMatchesFiveScalarOperationsWithOffsetsAndTail() {
        int length = 11;
        int inputOffset = 3;
        int outputOffset = 5;
        float divisor = 1.4142135f;
        float addend = 1.0f;
        float multiplier = 0.5f;
        float[] input = new float[inputOffset + length + 2];
        for (int i = 0; i < input.length; i++) input[i] = i * 0.125f - 2.5f;
        float[] normalized = new float[length];
        float[] erf = new float[length];
        float[] shifted = new float[length];
        float[] product = new float[length];
        float[] expected = new float[outputOffset + length + 2];
        float[] actual = new float[expected.length];
        BinaryPlan scalar = new BinaryPlan(new TensorShape(length), new TensorShape(1),
                new TensorShape(length));

        backend.binary(BinaryOp.DIV, input, inputOffset, new float[] {divisor}, 0,
                normalized, 0, scalar);
        backend.erf(normalized, 0, erf, 0, length);
        backend.binary(BinaryOp.ADD, erf, 0, new float[] {addend}, 0,
                shifted, 0, scalar);
        backend.mul(input, inputOffset, shifted, 0, product, 0, length);
        backend.binary(BinaryOp.MUL, product, 0, new float[] {multiplier}, 0,
                expected, outputOffset, scalar);
        backend.gelu(input, inputOffset, actual, outputOffset, length,
                divisor, addend, multiplier);

        Assert.assertArrayEquals(expected, actual, 0.0f);
    }

    @Test
    public void computesBlockedUnitStridePointwiseConvolution() {
        int channels = 3;
        int height = 2;
        int width = 3;
        int outputChannels = 8;
        float[] input = new float[channels * height * width];
        float[] weights = new float[outputChannels * channels];
        float[] bias = new float[outputChannels];
        for (int i = 0; i < input.length; i++) input[i] = (i - 3) * 0.25f;
        for (int i = 0; i < weights.length; i++) weights[i] = (i % 7 - 3) * 0.125f;
        for (int i = 0; i < bias.length; i++) bias[i] = (i - 2) * 0.2f;

        float[] actual = new float[outputChannels * height * width];
        backend.conv(input, 0, weights, 0, bias, 0, actual, 0,
                1, channels, height, width, outputChannels,
                1, 1, 1, 1, 1, 1, 0, 0, 1, height, width);
        Assert.assertArrayEquals(referenceConvolution(input, weights, bias,
                channels, height, width, outputChannels, 1, 1, 0, height, width),
                actual, 0.0f);
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
    public void computesPaddedThreeByThreeConvolutionForBothStrides() {
        int channels = 2;
        int height = 4;
        int width = 5;
        int outputChannels = 4;
        float[] input = new float[channels * height * width];
        float[] weights = new float[outputChannels * channels * 9];
        float[] bias = new float[outputChannels];
        for (int i = 0; i < input.length; i++) {
            input[i] = (i % 13 - 6) * 0.25f;
        }
        for (int i = 0; i < weights.length; i++) {
            weights[i] = (i % 17 - 8) * 0.125f;
        }
        for (int i = 0; i < bias.length; i++) {
            bias[i] = (i - 1) * 0.375f;
        }

        float[] strideOne = new float[outputChannels * height * width];
        backend.conv(input, 0, weights, 0, bias, 0, strideOne, 0,
                1, channels, height, width, outputChannels,
                3, 3, 1, 1, 1, 1, 1, 1, 1, height, width);
        Assert.assertArrayEquals(referenceConvolution(input, weights, bias,
                channels, height, width, outputChannels, 3, 1, 1, height, width),
                strideOne, 0.0f);

        int strideTwoHeight = (height + 1) / 2;
        int strideTwoWidth = (width + 1) / 2;
        float[] strideTwo = new float[outputChannels * strideTwoHeight * strideTwoWidth];
        backend.conv(input, 0, weights, 0, bias, 0, strideTwo, 0,
                1, channels, height, width, outputChannels,
                3, 3, 2, 2, 1, 1, 1, 1, 1, strideTwoHeight, strideTwoWidth);
        Assert.assertArrayEquals(referenceConvolution(input, weights, bias,
                channels, height, width, outputChannels, 3, 2, 1,
                strideTwoHeight, strideTwoWidth), strideTwo, 0.0f);
    }

    @Test
    public void computesPaddedThreeByThreeConvolutionWithEightOutputChannels() {
        int channels = 3;
        int height = 5;
        int width = 6;
        int outputChannels = 8;
        float[] input = new float[channels * height * width];
        float[] weights = new float[outputChannels * channels * 9];
        float[] bias = new float[outputChannels];
        for (int i = 0; i < input.length; i++) {
            input[i] = (i % 19 - 9) * 0.125f;
        }
        for (int i = 0; i < weights.length; i++) {
            weights[i] = (i % 23 - 11) * 0.0625f;
        }
        for (int i = 0; i < bias.length; i++) {
            bias[i] = (i - 3) * 0.1875f;
        }

        float[] actual = new float[outputChannels * height * width];
        backend.conv(input, 0, weights, 0, bias, 0, actual, 0,
                1, channels, height, width, outputChannels,
                3, 3, 1, 1, 1, 1, 1, 1, 1, height, width);
        Assert.assertArrayEquals(referenceConvolution(input, weights, bias,
                channels, height, width, outputChannels, 3, 1, 1, height, width),
                actual, 0.0f);
    }

    @Test
    public void computesRightAndBottomPaddedTwoByTwoConvolution() {
        int channels = 3;
        int height = 3;
        int width = 4;
        int outputChannels = 4;
        float[] input = new float[channels * height * width];
        float[] weights = new float[outputChannels * channels * 4];
        float[] bias = new float[outputChannels];
        for (int i = 0; i < input.length; i++) {
            input[i] = (i % 9 - 4) * 0.2f;
        }
        for (int i = 0; i < weights.length; i++) {
            weights[i] = (i % 7 - 3) * 0.15f;
        }
        for (int i = 0; i < bias.length; i++) {
            bias[i] = (i - 2) * 0.25f;
        }

        float[] actual = new float[outputChannels * height * width];
        backend.conv(input, 0, weights, 0, bias, 0, actual, 0,
                1, channels, height, width, outputChannels,
                2, 2, 1, 1, 1, 1, 0, 0, 1, 1, 1, height, width);
        Assert.assertArrayEquals(referenceConvolution(input, weights, bias,
                channels, height, width, outputChannels, 2, 1, 0, height, width),
                actual, 0.0f);
    }

    private static float[] referenceConvolution(float[] input, float[] weights, float[] bias,
                                                int channels, int height, int width,
                                                int outputChannels, int kernel, int stride,
                                                int padding, int outputHeight, int outputWidth) {
        float[] output = new float[outputChannels * outputHeight * outputWidth];
        for (int outputChannel = 0; outputChannel < outputChannels; outputChannel++) {
            for (int outputY = 0; outputY < outputHeight; outputY++) {
                for (int outputX = 0; outputX < outputWidth; outputX++) {
                    float sum = bias[outputChannel];
                    for (int inputChannel = 0; inputChannel < channels; inputChannel++) {
                        for (int kernelY = 0; kernelY < kernel; kernelY++) {
                            int inputY = outputY * stride - padding + kernelY;
                            if (inputY < 0 || inputY >= height) {
                                continue;
                            }
                            for (int kernelX = 0; kernelX < kernel; kernelX++) {
                                int inputX = outputX * stride - padding + kernelX;
                                if (inputX < 0 || inputX >= width) {
                                    continue;
                                }
                                int inputIndex = (inputChannel * height + inputY) * width + inputX;
                                int weightIndex = ((outputChannel * channels + inputChannel) * kernel
                                        + kernelY) * kernel + kernelX;
                                sum += input[inputIndex] * weights[weightIndex];
                            }
                        }
                    }
                    output[(outputChannel * outputHeight + outputY) * outputWidth + outputX] = sum;
                }
            }
        }
        return output;
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

        float[] kept = new float[6];
        backend.reduceMean(new float[] {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}, 0,
                kept, 0, new int[] {2, 2, 3}, new int[] {-1, -2}, true);
        Assert.assertArrayEquals(new float[] {3.5f, 9.5f, 0, 0, 0, 0}, kept, 0.0f);

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
    public void computesStrideTwoTransposedConvolutionForMultipleOutputs() {
        float[] output = new float[2 * 4 * 4];
        backend.convTranspose(new float[] {1, 2, 3, 4}, 0,
                new float[] {1, 2, 3, 4, 10, 20, 30, 40}, 0,
                new float[] {0, 100}, 0, output, 0,
                1, 1, 2, 2, 2, 2, 2, 2, 2, 1, 1, 0, 0, 1, 4, 4);
        Assert.assertArrayEquals(new float[] {
                1, 2, 2, 4, 3, 4, 6, 8,
                3, 6, 4, 8, 9, 12, 12, 16,
                110, 120, 120, 140, 130, 140, 160, 180,
                130, 160, 140, 180, 190, 220, 220, 260
        }, output, 0.0f);
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
