package io.github.lxw112190.ppocr.vector;

import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorSpecies;

/** Fixed-shape NCHW kernel for the first 3x3 stride-two REC convolution. */
final class VectorRecConv3x3Stride2Kernel {
    private static final VectorSpecies<Float> SPECIES = VectorSupport.F32;
    private static final int[] STRIDE_TWO_INDEXES = strideIndexes();

    private VectorRecConv3x3Stride2Kernel() { }

    static boolean supports(int batch, int channels, int height, int width,
                            int outputChannels, int outputHeight, int outputWidth,
                            int kernelHeight, int kernelWidth, int strideHeight,
                            int strideWidth, int padTop, int padLeft, int padBottom,
                            int padRight, int groups) {
        return batch == 1
                && channels == 24
                && outputChannels == 48
                && height == 24
                && outputHeight == 12
                && outputWidth == (width + 1) / 2
                && kernelHeight == 3
                && kernelWidth == 3
                && strideHeight == 2
                && strideWidth == 2
                && padTop == 1
                && padLeft == 1
                && padBottom == 1
                && padRight == 1
                && groups == 1;
    }

    static void apply(float[] input, int inputOffset, float[] weights, int weightOffset,
                      float[] bias, int biasOffset, float[] output, int outputOffset,
                      int width, int outputWidth) {
        int inputPlane = 24 * width;
        int outputPlane = 12 * outputWidth;
        int lanes = SPECIES.length();
        int firstVector = outputWidth > lanes ? 1 : outputWidth;
        boolean rightPadding = (width & 1) != 0;
        int lastVector = rightPadding ? outputWidth - 1 - lanes : outputWidth - lanes;

        for (int outputChannel = 0; outputChannel < 48; outputChannel += 8) {
            int outputBase0 = outputOffset + outputChannel * outputPlane;
            int outputBase1 = outputBase0 + outputPlane;
            int outputBase2 = outputBase1 + outputPlane;
            int outputBase3 = outputBase2 + outputPlane;
            int outputBase4 = outputBase3 + outputPlane;
            int outputBase5 = outputBase4 + outputPlane;
            int outputBase6 = outputBase5 + outputPlane;
            int outputBase7 = outputBase6 + outputPlane;
            float bias0 = bias == null ? 0.0f : bias[biasOffset + outputChannel];
            float bias1 = bias == null ? 0.0f : bias[biasOffset + outputChannel + 1];
            float bias2 = bias == null ? 0.0f : bias[biasOffset + outputChannel + 2];
            float bias3 = bias == null ? 0.0f : bias[biasOffset + outputChannel + 3];
            float bias4 = bias == null ? 0.0f : bias[biasOffset + outputChannel + 4];
            float bias5 = bias == null ? 0.0f : bias[biasOffset + outputChannel + 5];
            float bias6 = bias == null ? 0.0f : bias[biasOffset + outputChannel + 6];
            float bias7 = bias == null ? 0.0f : bias[biasOffset + outputChannel + 7];

            for (int oh = 0; oh < 12; oh++) {
                int khStart = oh == 0 ? 1 : 0;
                int khEnd = 3;
                int outputRow0 = outputBase0 + oh * outputWidth;
                int outputRow1 = outputBase1 + oh * outputWidth;
                int outputRow2 = outputBase2 + oh * outputWidth;
                int outputRow3 = outputBase3 + oh * outputWidth;
                int outputRow4 = outputBase4 + oh * outputWidth;
                int outputRow5 = outputBase5 + oh * outputWidth;
                int outputRow6 = outputBase6 + oh * outputWidth;
                int outputRow7 = outputBase7 + oh * outputWidth;

                scalarPixel(input, inputOffset, weights, weightOffset, output,
                        outputRow0, outputRow1, outputRow2, outputRow3, outputRow4,
                        outputRow5, outputRow6, outputRow7, width, oh, 0, khStart,
                        khEnd, outputChannel, bias0, bias1, bias2, bias3, bias4, bias5,
                        bias6, bias7);
                for (int ow = firstVector; ow < lastVector; ow += lanes) {
                    vectorPixel(input, inputOffset, weights, weightOffset, output,
                            outputRow0, outputRow1, outputRow2, outputRow3, outputRow4,
                        outputRow5, outputRow6, outputRow7, width, oh, ow, khStart,
                        khEnd, outputChannel, bias0, bias1, bias2, bias3, bias4,
                        bias5, bias6, bias7);
                }
                if (lastVector >= firstVector) {
                    vectorPixel(input, inputOffset, weights, weightOffset, output,
                            outputRow0, outputRow1, outputRow2, outputRow3, outputRow4,
                            outputRow5, outputRow6, outputRow7, width, oh, lastVector,
                            khStart, khEnd, outputChannel, bias0, bias1, bias2, bias3,
                            bias4, bias5, bias6, bias7);
                }
                int firstScalar = outputWidth > lanes
                        ? Math.max(firstVector, lastVector + lanes) : 1;
                for (int ow = firstScalar; ow < outputWidth; ow++) {
                    if (rightPadding && ow == outputWidth - 1) {
                        scalarPixel(input, inputOffset, weights, weightOffset, output,
                                outputRow0, outputRow1, outputRow2, outputRow3, outputRow4,
                                outputRow5, outputRow6, outputRow7, width, oh, ow,
                                khStart, khEnd, outputChannel, bias0, bias1, bias2, bias3,
                                bias4, bias5, bias6, bias7);
                    } else if (ow >= firstVector) {
                        scalarPixel(input, inputOffset, weights, weightOffset, output,
                                outputRow0, outputRow1, outputRow2, outputRow3, outputRow4,
                                outputRow5, outputRow6, outputRow7, width, oh, ow,
                                khStart, khEnd, outputChannel, bias0, bias1, bias2, bias3,
                                bias4, bias5, bias6, bias7);
                    }
                }
            }
        }
    }

    private static void vectorPixel(float[] input, int inputOffset, float[] weights,
                                    int weightOffset, float[] output, int outputRow0,
                                    int outputRow1, int outputRow2, int outputRow3,
                                    int outputRow4, int outputRow5, int outputRow6,
                                    int outputRow7, int width, int oh, int ow,
                                    int khStart, int khEnd, int outputChannel,
                                    float bias0, float bias1, float bias2, float bias3,
                                    float bias4, float bias5, float bias6, float bias7) {
        FloatVector sum0 = FloatVector.broadcast(SPECIES, bias0);
        FloatVector sum1 = FloatVector.broadcast(SPECIES, bias1);
        FloatVector sum2 = FloatVector.broadcast(SPECIES, bias2);
        FloatVector sum3 = FloatVector.broadcast(SPECIES, bias3);
        FloatVector sum4 = FloatVector.broadcast(SPECIES, bias4);
        FloatVector sum5 = FloatVector.broadcast(SPECIES, bias5);
        FloatVector sum6 = FloatVector.broadcast(SPECIES, bias6);
        FloatVector sum7 = FloatVector.broadcast(SPECIES, bias7);
        for (int ic = 0; ic < 24; ic++) {
            int inputBase = inputOffset + ic * 24 * width;
            int kernelBase = ic * 9;
            for (int kh = khStart; kh < khEnd; kh++) {
                int inputRow = inputBase + (oh * 2 - 1 + kh) * width + ow * 2 - 1;
                for (int kw = 0; kw < 3; kw++) {
                    FloatVector sample = FloatVector.fromArray(
                            SPECIES, input, inputRow + kw, STRIDE_TWO_INDEXES, 0);
                    int kernelIndex = kernelBase + kh * 3 + kw;
                    sum0 = sum0.add(sample.mul(weights[weightOffset
                            + outputChannel * 24 * 9 + kernelIndex]));
                    sum1 = sum1.add(sample.mul(weights[weightOffset
                            + (outputChannel + 1) * 24 * 9 + kernelIndex]));
                    sum2 = sum2.add(sample.mul(weights[weightOffset
                            + (outputChannel + 2) * 24 * 9 + kernelIndex]));
                    sum3 = sum3.add(sample.mul(weights[weightOffset
                            + (outputChannel + 3) * 24 * 9 + kernelIndex]));
                    sum4 = sum4.add(sample.mul(weights[weightOffset
                            + (outputChannel + 4) * 24 * 9 + kernelIndex]));
                    sum5 = sum5.add(sample.mul(weights[weightOffset
                            + (outputChannel + 5) * 24 * 9 + kernelIndex]));
                    sum6 = sum6.add(sample.mul(weights[weightOffset
                            + (outputChannel + 6) * 24 * 9 + kernelIndex]));
                    sum7 = sum7.add(sample.mul(weights[weightOffset
                            + (outputChannel + 7) * 24 * 9 + kernelIndex]));
                }
            }
        }
        sum0.intoArray(output, outputRow0 + ow);
        sum1.intoArray(output, outputRow1 + ow);
        sum2.intoArray(output, outputRow2 + ow);
        sum3.intoArray(output, outputRow3 + ow);
        sum4.intoArray(output, outputRow4 + ow);
        sum5.intoArray(output, outputRow5 + ow);
        sum6.intoArray(output, outputRow6 + ow);
        sum7.intoArray(output, outputRow7 + ow);
    }

    private static void scalarPixel(float[] input, int inputOffset, float[] weights,
                                    int weightOffset, float[] output, int outputRow0,
                                    int outputRow1, int outputRow2, int outputRow3,
                                    int outputRow4, int outputRow5, int outputRow6,
                                    int outputRow7, int width, int oh, int ow,
                                    int khStart, int khEnd, int outputChannel,
                                    float bias0, float bias1, float bias2, float bias3,
                                    float bias4, float bias5, float bias6, float bias7) {
        float value0 = bias0;
        float value1 = bias1;
        float value2 = bias2;
        float value3 = bias3;
        float value4 = bias4;
        float value5 = bias5;
        float value6 = bias6;
        float value7 = bias7;
        int kwStart = ow == 0 ? 1 : 0;
        int kwEnd = ow * 2 + 1 >= width ? 2 : 3;
        for (int ic = 0; ic < 24; ic++) {
            int inputBase = inputOffset + ic * 24 * width;
            int kernelBase = ic * 9;
            for (int kh = khStart; kh < khEnd; kh++) {
                int inputRow = inputBase + (oh * 2 - 1 + kh) * width + ow * 2 - 1;
                for (int kw = kwStart; kw < kwEnd; kw++) {
                    int kernelIndex = kernelBase + kh * 3 + kw;
                    float sample = input[inputRow + kw];
                    value0 += sample * weights[weightOffset
                            + outputChannel * 24 * 9 + kernelIndex];
                    value1 += sample * weights[weightOffset
                            + (outputChannel + 1) * 24 * 9 + kernelIndex];
                    value2 += sample * weights[weightOffset
                            + (outputChannel + 2) * 24 * 9 + kernelIndex];
                    value3 += sample * weights[weightOffset
                            + (outputChannel + 3) * 24 * 9 + kernelIndex];
                    value4 += sample * weights[weightOffset
                            + (outputChannel + 4) * 24 * 9 + kernelIndex];
                    value5 += sample * weights[weightOffset
                            + (outputChannel + 5) * 24 * 9 + kernelIndex];
                    value6 += sample * weights[weightOffset
                            + (outputChannel + 6) * 24 * 9 + kernelIndex];
                    value7 += sample * weights[weightOffset
                            + (outputChannel + 7) * 24 * 9 + kernelIndex];
                }
            }
        }
        output[outputRow0 + ow] = value0;
        output[outputRow1 + ow] = value1;
        output[outputRow2 + ow] = value2;
        output[outputRow3 + ow] = value3;
        output[outputRow4 + ow] = value4;
        output[outputRow5 + ow] = value5;
        output[outputRow6 + ow] = value6;
        output[outputRow7 + ow] = value7;
    }

    private static int[] strideIndexes() {
        int[] indexes = new int[SPECIES.length()];
        for (int i = 0; i < indexes.length; i++) indexes[i] = i * 2;
        return indexes;
    }
}
