package io.github.lxw112190.ppocr.vector;

import java.util.Map;
import java.util.WeakHashMap;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorSpecies;

/**
 * Channel-vectorized 3x3 kernel for the large DET 16-channel feature maps.
 *
 * <p>The public tensor contract remains NCHW. Only the weight tiles are kept
 * in an NHWC-friendly [input-channel, kernel, output-channel] order, so the
 * kernel does not allocate or expose a second activation layout.</p>
 */
final class NhwcConv3x3Kernel {
    private static final VectorSpecies<Float> SPECIES = VectorSupport.F32;
    private static final Map<float[], PackedWeights> PACKED_WEIGHTS =
            new WeakHashMap<float[], PackedWeights>();

    private NhwcConv3x3Kernel() { }

    static boolean supports(int channels, int outputChannels, int height, int width) {
        return outputChannels == 16 && channels >= 32 && height >= 64 && width >= 64
                && (SPECIES.length() == 8 || SPECIES.length() == 16);
    }

    static void strideOne(float[] input, int inputOffset, float[] weights, int weightOffset,
                          float[] bias, int biasOffset, float[] output, int outputOffset,
                          int batch, int channels, int height, int width,
                          int outputChannels) {
        PackedWeights packed = packedWeights(weights, weightOffset, channels, outputChannels);
        int plane = height * width;
        boolean wideSpecies = SPECIES.length() >= 16;
        for (int n = 0; n < batch; n++) {
            int inputBatch = inputOffset + n * channels * plane;
            int outputBatch = outputOffset + n * outputChannels * plane;
            for (int oh = 0; oh < height; oh++) {
                int khStart = oh == 0 ? 1 : 0;
                int khEnd = oh == height - 1 ? 2 : 3;
                for (int ow = 0; ow < width; ow++) {
                    int kwStart = ow == 0 ? 1 : 0;
                    int kwEnd = ow == width - 1 ? 2 : 3;
                    FloatVector sum0 = bias == null
                            ? FloatVector.zero(SPECIES)
                            : FloatVector.fromArray(SPECIES, bias, biasOffset);
                    FloatVector sum1 = wideSpecies || bias == null ? null
                            : FloatVector.fromArray(SPECIES, bias, biasOffset + 8);
                    if (!wideSpecies && bias == null) sum1 = FloatVector.zero(SPECIES);
                    for (int ic = 0; ic < channels; ic++) {
                        int inputChannel = inputBatch + ic * plane;
                        int kernelBase = ic * 9;
                        for (int kh = khStart; kh < khEnd; kh++) {
                            int inputRow = inputChannel + (oh + kh - 1) * width;
                            int packedRow = (kernelBase + kh * 3 + kwStart) * outputChannels;
                            for (int kw = kwStart; kw < kwEnd; kw++) {
                                float sample = input[inputRow + ow + kw - 1];
                                FloatVector weight0 = FloatVector.fromArray(
                                        SPECIES, packed.values, packedRow);
                                sum0 = sum0.add(weight0.mul(sample));
                                if (!wideSpecies) {
                                    FloatVector weight1 = FloatVector.fromArray(
                                            SPECIES, packed.values, packedRow + 8);
                                    sum1 = sum1.add(weight1.mul(sample));
                                }
                                packedRow += outputChannels;
                            }
                        }
                    }
                    int pixel = oh * width + ow;
                    writeChannels(sum0, output, outputBatch, pixel, plane, 0);
                    if (!wideSpecies) {
                        writeChannels(sum1, output, outputBatch, pixel, plane, 8);
                    }
                }
            }
        }
    }

    private static void writeChannels(FloatVector values, float[] output, int outputBatch,
                                      int pixel, int plane, int channelOffset) {
        int lanes = Math.min(SPECIES.length(), 16 - channelOffset);
        for (int lane = 0; lane < lanes; lane++) {
            output[outputBatch + (channelOffset + lane) * plane + pixel] = values.lane(lane);
        }
    }

    private static PackedWeights packedWeights(float[] weights, int weightOffset,
                                               int channels, int outputChannels) {
        synchronized (PACKED_WEIGHTS) {
            PackedWeights value = PACKED_WEIGHTS.get(weights);
            if (value == null || value.weightOffset != weightOffset
                    || value.channels != channels || value.outputChannels != outputChannels) {
                value = new PackedWeights(weights, weightOffset, channels, outputChannels);
                PACKED_WEIGHTS.put(weights, value);
            }
            return value;
        }
    }

    private static final class PackedWeights {
        final int weightOffset;
        final int channels;
        final int outputChannels;
        final float[] values;

        PackedWeights(float[] source, int weightOffset, int channels, int outputChannels) {
            this.weightOffset = weightOffset;
            this.channels = channels;
            this.outputChannels = outputChannels;
            this.values = new float[channels * 9 * outputChannels];
            int sourceChannelStride = channels * 9;
            for (int ic = 0; ic < channels; ic++) {
                for (int kernel = 0; kernel < 9; kernel++) {
                    int destination = (ic * 9 + kernel) * outputChannels;
                    for (int oc = 0; oc < outputChannels; oc++) {
                        values[destination + oc] = source[weightOffset
                                + oc * sourceChannelStride + ic * 9 + kernel];
                    }
                }
            }
        }
    }
}
