package io.github.lxw112190.ppocr.vector;

import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorSpecies;

/** Vector microkernel for the common 3x3, same-padding, stride-two convolution. */
final class VectorConv3x3Stride2Kernel {
    private VectorConv3x3Stride2Kernel() { }

    static void apply(float[] input, int inputOffset, float[] weights, int weightOffset,
                      float[] bias, int biasOffset, float[] output, int outputOffset,
                      int batch, int channels, int height, int width, int outputChannels,
                      int outputHeight, int outputWidth, VectorSpecies<Float> species,
                      int[] strideIndexes) {
        int inputPlane = height * width;
        int outputPlane = outputHeight * outputWidth;
        int fullColumnEnd = width / 2;
        int vectorStart = fullColumnEnd > species.length() ? 1 : fullColumnEnd;
        int lastVectorStart = fullColumnEnd - species.length();
        for (int n = 0; n < batch; n++) {
            for (int outputChannel = 0; outputChannel < outputChannels; outputChannel += 8) {
                int outputBase0 = outputOffset
                        + (n * outputChannels + outputChannel) * outputPlane;
                int outputBase1 = outputBase0 + outputPlane;
                int outputBase2 = outputBase1 + outputPlane;
                int outputBase3 = outputBase2 + outputPlane;
                int outputBase4 = outputBase3 + outputPlane;
                int outputBase5 = outputBase4 + outputPlane;
                int outputBase6 = outputBase5 + outputPlane;
                int outputBase7 = outputBase6 + outputPlane;
                int weightBase0 = weightOffset + outputChannel * channels * 9;
                int weightBase1 = weightBase0 + channels * 9;
                int weightBase2 = weightBase1 + channels * 9;
                int weightBase3 = weightBase2 + channels * 9;
                int weightBase4 = weightBase3 + channels * 9;
                int weightBase5 = weightBase4 + channels * 9;
                int weightBase6 = weightBase5 + channels * 9;
                int weightBase7 = weightBase6 + channels * 9;
                float bias0 = bias == null ? 0.0f : bias[biasOffset + outputChannel];
                float bias1 = bias == null ? 0.0f : bias[biasOffset + outputChannel + 1];
                float bias2 = bias == null ? 0.0f : bias[biasOffset + outputChannel + 2];
                float bias3 = bias == null ? 0.0f : bias[biasOffset + outputChannel + 3];
                float bias4 = bias == null ? 0.0f : bias[biasOffset + outputChannel + 4];
                float bias5 = bias == null ? 0.0f : bias[biasOffset + outputChannel + 5];
                float bias6 = bias == null ? 0.0f : bias[biasOffset + outputChannel + 6];
                float bias7 = bias == null ? 0.0f : bias[biasOffset + outputChannel + 7];
                for (int oh = 0; oh < outputHeight; oh++) {
                    int outputRow0 = outputBase0 + oh * outputWidth;
                    int outputRow1 = outputBase1 + oh * outputWidth;
                    int outputRow2 = outputBase2 + oh * outputWidth;
                    int outputRow3 = outputBase3 + oh * outputWidth;
                    int outputRow4 = outputBase4 + oh * outputWidth;
                    int outputRow5 = outputBase5 + oh * outputWidth;
                    int outputRow6 = outputBase6 + oh * outputWidth;
                    int outputRow7 = outputBase7 + oh * outputWidth;
                    int ow = vectorStart;
                    // Keep padded edge columns on the scalar path.
                    while (ow < fullColumnEnd) {
                        FloatVector sum0 = FloatVector.broadcast(species, bias0);
                        FloatVector sum1 = FloatVector.broadcast(species, bias1);
                        FloatVector sum2 = FloatVector.broadcast(species, bias2);
                        FloatVector sum3 = FloatVector.broadcast(species, bias3);
                        FloatVector sum4 = FloatVector.broadcast(species, bias4);
                        FloatVector sum5 = FloatVector.broadcast(species, bias5);
                        FloatVector sum6 = FloatVector.broadcast(species, bias6);
                        FloatVector sum7 = FloatVector.broadcast(species, bias7);
                        for (int ic = 0; ic < channels; ic++) {
                            int inputBase = inputOffset + (n * channels + ic) * inputPlane;
                            int kernelBase = ic * 9;
                            for (int kh = 0; kh < 3; kh++) {
                                int ih = oh * 2 - 1 + kh;
                                if (ih < 0 || ih >= height) continue;
                                int inputRow = inputBase + ih * width + ow * 2 - 1;
                                for (int kw = 0; kw < 3; kw++) {
                                    FloatVector sample = FloatVector.fromArray(
                                            species, input, inputRow + kw, strideIndexes, 0);
                                    int kernelIndex = kernelBase + kh * 3 + kw;
                                    sum0 = sum0.add(sample.mul(weights[weightBase0 + kernelIndex]));
                                    sum1 = sum1.add(sample.mul(weights[weightBase1 + kernelIndex]));
                                    sum2 = sum2.add(sample.mul(weights[weightBase2 + kernelIndex]));
                                    sum3 = sum3.add(sample.mul(weights[weightBase3 + kernelIndex]));
                                    sum4 = sum4.add(sample.mul(weights[weightBase4 + kernelIndex]));
                                    sum5 = sum5.add(sample.mul(weights[weightBase5 + kernelIndex]));
                                    sum6 = sum6.add(sample.mul(weights[weightBase6 + kernelIndex]));
                                    sum7 = sum7.add(sample.mul(weights[weightBase7 + kernelIndex]));
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
                        if (ow == lastVectorStart) break;
                        ow = Math.min(ow + species.length(), lastVectorStart);
                    }
                    scalarEdge(input, inputOffset, weights, weightOffset, bias, biasOffset,
                            output, outputOffset, n, channels, height, width, outputChannels,
                            outputHeight, outputWidth, outputChannel, oh, 0, vectorStart);
                    scalarEdge(input, inputOffset, weights, weightOffset, bias, biasOffset,
                            output, outputOffset, n, channels, height, width, outputChannels,
                            outputHeight, outputWidth, outputChannel, oh,
                            fullColumnEnd, outputWidth);
                }
            }
        }
    }

    private static void scalarEdge(float[] input, int inputOffset, float[] weights,
                                   int weightOffset, float[] bias, int biasOffset,
                                   float[] output, int outputOffset, int n, int channels,
                                   int height, int width, int outputChannels, int outputHeight,
                                   int outputWidth, int firstOutputChannel, int oh,
                                   int firstColumn, int lastColumn) {
        int inputPlane = height * width;
        int outputPlane = outputHeight * outputWidth;
        for (int oc = 0; oc < 8; oc++) {
            int outputChannel = firstOutputChannel + oc;
            int outputRow = outputOffset
                    + (n * outputChannels + outputChannel) * outputPlane
                    + oh * outputWidth;
            int weightBase = weightOffset + outputChannel * channels * 9;
            for (int ow = firstColumn; ow < lastColumn; ow++) {
                float value = bias == null ? 0.0f : bias[biasOffset + outputChannel];
                for (int ic = 0; ic < channels; ic++) {
                    int inputBase = inputOffset + (n * channels + ic) * inputPlane;
                    int kernelBase = weightBase + ic * 9;
                    for (int kh = 0; kh < 3; kh++) {
                        int ih = oh * 2 - 1 + kh;
                        if (ih < 0 || ih >= height) continue;
                        for (int kw = 0; kw < 3; kw++) {
                            int iw = ow * 2 - 1 + kw;
                            if (iw < 0 || iw >= width) continue;
                            value += input[inputBase + ih * width + iw]
                                    * weights[kernelBase + kh * 3 + kw];
                        }
                    }
                }
                output[outputRow + ow] = value;
            }
        }
    }
}
