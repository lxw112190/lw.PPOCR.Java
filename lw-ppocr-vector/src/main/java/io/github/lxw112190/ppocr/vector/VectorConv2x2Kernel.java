package io.github.lxw112190.ppocr.vector;

import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorSpecies;

/** Vector microkernel for the common 2x2, bottom/right-padded convolution. */
final class VectorConv2x2Kernel {
    private VectorConv2x2Kernel() { }

    static void strideOne(float[] input, int inputOffset, float[] weights, int weightOffset,
                          float[] bias, int biasOffset, float[] output, int outputOffset,
                          int batch, int channels, int height, int width, int outputChannels,
                          VectorSpecies<Float> species) {
        int plane = height * width;
        int fullColumnEnd = width - 1;
        int vectorStart = fullColumnEnd >= species.length() ? 0 : fullColumnEnd;
        int lastVectorStart = fullColumnEnd - species.length();
        for (int n = 0; n < batch; n++) {
            for (int outputChannel = 0; outputChannel < outputChannels; outputChannel += 8) {
                int outputBase0 = outputOffset + (n * outputChannels + outputChannel) * plane;
                int outputBase1 = outputBase0 + plane;
                int outputBase2 = outputBase1 + plane;
                int outputBase3 = outputBase2 + plane;
                int outputBase4 = outputBase3 + plane;
                int outputBase5 = outputBase4 + plane;
                int outputBase6 = outputBase5 + plane;
                int outputBase7 = outputBase6 + plane;
                int weightBase0 = weightOffset + outputChannel * channels * 4;
                int weightBase1 = weightBase0 + channels * 4;
                int weightBase2 = weightBase1 + channels * 4;
                int weightBase3 = weightBase2 + channels * 4;
                int weightBase4 = weightBase3 + channels * 4;
                int weightBase5 = weightBase4 + channels * 4;
                int weightBase6 = weightBase5 + channels * 4;
                int weightBase7 = weightBase6 + channels * 4;
                float bias0 = bias == null ? 0.0f : bias[biasOffset + outputChannel];
                float bias1 = bias == null ? 0.0f : bias[biasOffset + outputChannel + 1];
                float bias2 = bias == null ? 0.0f : bias[biasOffset + outputChannel + 2];
                float bias3 = bias == null ? 0.0f : bias[biasOffset + outputChannel + 3];
                float bias4 = bias == null ? 0.0f : bias[biasOffset + outputChannel + 4];
                float bias5 = bias == null ? 0.0f : bias[biasOffset + outputChannel + 5];
                float bias6 = bias == null ? 0.0f : bias[biasOffset + outputChannel + 6];
                float bias7 = bias == null ? 0.0f : bias[biasOffset + outputChannel + 7];
                for (int oh = 0; oh < height; oh++) {
                    int outputRow0 = outputBase0 + oh * width;
                    int outputRow1 = outputBase1 + oh * width;
                    int outputRow2 = outputBase2 + oh * width;
                    int outputRow3 = outputBase3 + oh * width;
                    int outputRow4 = outputBase4 + oh * width;
                    int outputRow5 = outputBase5 + oh * width;
                    int outputRow6 = outputBase6 + oh * width;
                    int outputRow7 = outputBase7 + oh * width;
                    int ow = vectorStart;
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
                            int inputBase = inputOffset + (n * channels + ic) * plane
                                    + oh * width + ow;
                            int kernelIndex = ic * 4;
                            FloatVector sample = FloatVector.fromArray(species, input, inputBase);
                            sum0 = sum0.add(sample.mul(weights[weightBase0 + kernelIndex]));
                            sum1 = sum1.add(sample.mul(weights[weightBase1 + kernelIndex]));
                            sum2 = sum2.add(sample.mul(weights[weightBase2 + kernelIndex]));
                            sum3 = sum3.add(sample.mul(weights[weightBase3 + kernelIndex]));
                            sum4 = sum4.add(sample.mul(weights[weightBase4 + kernelIndex]));
                            sum5 = sum5.add(sample.mul(weights[weightBase5 + kernelIndex]));
                            sum6 = sum6.add(sample.mul(weights[weightBase6 + kernelIndex]));
                            sum7 = sum7.add(sample.mul(weights[weightBase7 + kernelIndex]));
                            sample = FloatVector.fromArray(species, input, inputBase + 1);
                            sum0 = sum0.add(sample.mul(weights[weightBase0 + kernelIndex + 1]));
                            sum1 = sum1.add(sample.mul(weights[weightBase1 + kernelIndex + 1]));
                            sum2 = sum2.add(sample.mul(weights[weightBase2 + kernelIndex + 1]));
                            sum3 = sum3.add(sample.mul(weights[weightBase3 + kernelIndex + 1]));
                            sum4 = sum4.add(sample.mul(weights[weightBase4 + kernelIndex + 1]));
                            sum5 = sum5.add(sample.mul(weights[weightBase5 + kernelIndex + 1]));
                            sum6 = sum6.add(sample.mul(weights[weightBase6 + kernelIndex + 1]));
                            sum7 = sum7.add(sample.mul(weights[weightBase7 + kernelIndex + 1]));
                            if (oh + 1 < height) {
                                sample = FloatVector.fromArray(species, input, inputBase + width);
                                sum0 = sum0.add(sample.mul(weights[weightBase0 + kernelIndex + 2]));
                                sum1 = sum1.add(sample.mul(weights[weightBase1 + kernelIndex + 2]));
                                sum2 = sum2.add(sample.mul(weights[weightBase2 + kernelIndex + 2]));
                                sum3 = sum3.add(sample.mul(weights[weightBase3 + kernelIndex + 2]));
                                sum4 = sum4.add(sample.mul(weights[weightBase4 + kernelIndex + 2]));
                                sum5 = sum5.add(sample.mul(weights[weightBase5 + kernelIndex + 2]));
                                sum6 = sum6.add(sample.mul(weights[weightBase6 + kernelIndex + 2]));
                                sum7 = sum7.add(sample.mul(weights[weightBase7 + kernelIndex + 2]));
                                sample = FloatVector.fromArray(species, input,
                                        inputBase + width + 1);
                                sum0 = sum0.add(sample.mul(weights[weightBase0 + kernelIndex + 3]));
                                sum1 = sum1.add(sample.mul(weights[weightBase1 + kernelIndex + 3]));
                                sum2 = sum2.add(sample.mul(weights[weightBase2 + kernelIndex + 3]));
                                sum3 = sum3.add(sample.mul(weights[weightBase3 + kernelIndex + 3]));
                                sum4 = sum4.add(sample.mul(weights[weightBase4 + kernelIndex + 3]));
                                sum5 = sum5.add(sample.mul(weights[weightBase5 + kernelIndex + 3]));
                                sum6 = sum6.add(sample.mul(weights[weightBase6 + kernelIndex + 3]));
                                sum7 = sum7.add(sample.mul(weights[weightBase7 + kernelIndex + 3]));
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
                            outputChannel, oh, 0, vectorStart);
                    scalarEdge(input, inputOffset, weights, weightOffset, bias, biasOffset,
                            output, outputOffset, n, channels, height, width, outputChannels,
                            outputChannel, oh, fullColumnEnd, width);
                }
            }
        }
    }

    private static void scalarEdge(float[] input, int inputOffset, float[] weights,
                                   int weightOffset, float[] bias, int biasOffset,
                                   float[] output, int outputOffset, int n, int channels,
                                   int height, int width, int outputChannels,
                                   int firstOutputChannel, int oh, int firstColumn,
                                   int lastColumn) {
        int plane = height * width;
        for (int oc = 0; oc < 8; oc++) {
            int outputChannel = firstOutputChannel + oc;
            int outputRow = outputOffset + (n * outputChannels + outputChannel) * plane
                    + oh * width;
            int weightBase = weightOffset + outputChannel * channels * 4;
            for (int ow = firstColumn; ow < lastColumn; ow++) {
                float value = bias == null ? 0.0f : bias[biasOffset + outputChannel];
                for (int ic = 0; ic < channels; ic++) {
                    int inputBase = inputOffset + (n * channels + ic) * plane;
                    int kernelBase = weightBase + ic * 4;
                    for (int kh = 0; kh < 2; kh++) {
                        int ih = oh + kh;
                        if (ih >= height) continue;
                        for (int kw = 0; kw < 2; kw++) {
                            int iw = ow + kw;
                            if (iw >= width) continue;
                            value += input[inputBase + ih * width + iw]
                                    * weights[kernelBase + kh * 2 + kw];
                        }
                    }
                }
                output[outputRow + ow] = value;
            }
        }
    }
}
