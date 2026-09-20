package io.github.lxw112190.ppocr.vector;

import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorSpecies;

/**
 * Vectorized 1x1 convolution microkernel.
 *
 * <p>The implementation keeps weights in the existing OI layout and writes
 * NCHW output directly. It is intentionally isolated from the dispatch logic
 * so the pointwise hot path can be benchmarked independently.</p>
 */
final class VectorPointwiseKernel {
    private static final VectorSpecies<Float> SPECIES = VectorSupport.F32;
    private static final String BLOCK_PROPERTY = "lwppocr.vectorPointwiseBlock";
    private static final int OUTPUT_BLOCK = configuredOutputBlock();

    private static int configuredOutputBlock() {
        String value = System.getProperty(BLOCK_PROPERTY, "12");
        final int block;
        try {
            block = Integer.parseInt(value);
        } catch (NumberFormatException ex) {
            throw new IllegalArgumentException(BLOCK_PROPERTY + " must be 4, 8 or 12", ex);
        }
        return validateOutputBlock(block);
    }

    private static int validateOutputBlock(int block) {
        if (block != 4 && block != 8 && block != 12) {
            throw new IllegalArgumentException(BLOCK_PROPERTY + " must be 4, 8 or 12");
        }
        return block;
    }

    static void apply(float[] input, int inputOffset, float[] weights, int weightOffset,
                      float[] bias, int biasOffset, float[] output, int outputOffset,
                      int batch, int channels, int plane, int outputChannels, int groups) {
        applyWithBlock(OUTPUT_BLOCK, input, inputOffset, weights, weightOffset, bias, biasOffset,
                output, outputOffset, batch, channels, plane, outputChannels, groups);
    }

    static void applyForTesting(int outputBlock,
                                float[] input, int inputOffset, float[] weights, int weightOffset,
                                float[] bias, int biasOffset, float[] output, int outputOffset,
                                int batch, int channels, int plane, int outputChannels, int groups) {
        applyWithBlock(validateOutputBlock(outputBlock), input, inputOffset, weights, weightOffset,
                bias, biasOffset, output, outputOffset, batch, channels, plane,
                outputChannels, groups);
    }

    private static void applyWithBlock(int outputBlock, float[] input, int inputOffset,
                                       float[] weights, int weightOffset, float[] bias,
                                       int biasOffset, float[] output, int outputOffset,
                                       int batch, int channels, int plane, int outputChannels,
                                       int groups) {
        int inputsPerGroup = channels / groups;
        int outputsPerGroup = outputChannels / groups;
        int bound = SPECIES.loopBound(plane);
        for (int n = 0; n < batch; n++) {
            for (int group = 0; group < groups; group++) {
                int inputGroupBase = inputOffset +
                        (n * channels + group * inputsPerGroup) * plane;
                int oc = 0;
                if (outputBlock == 12) {
                    for (; oc + 11 < outputsPerGroup; oc += 12) {
                        pointwiseTwelve(input, inputGroupBase, weights, weightOffset,
                                bias, biasOffset, output, outputOffset, n, outputChannels,
                                plane, inputsPerGroup, group * outputsPerGroup + oc, bound);
                    }
                }
                if (outputBlock >= 8) {
                    for (; oc + 7 < outputsPerGroup; oc += 8) {
                        pointwiseEight(input, inputGroupBase, weights, weightOffset,
                                bias, biasOffset, output, outputOffset, n, outputChannels,
                                plane, inputsPerGroup, group * outputsPerGroup + oc, bound);
                    }
                }
                for (; oc + 3 < outputsPerGroup; oc += 4) {
                    pointwiseFour(input, inputGroupBase, weights, weightOffset,
                            bias, biasOffset, output, outputOffset, n, outputChannels,
                            plane, inputsPerGroup, group * outputsPerGroup + oc, bound);
                }
                for (; oc < outputsPerGroup; oc++) {
                    pointwiseOne(input, inputGroupBase, weights, weightOffset,
                            bias, biasOffset, output, outputOffset, n, outputChannels,
                            plane, inputsPerGroup, group * outputsPerGroup + oc, bound);
                }
            }
        }
    }

    private static void pointwiseEight(float[] input, int inputGroupBase,
                                        float[] weights, int weightOffset,
                                        float[] bias, int biasOffset,
                                        float[] output, int outputOffset,
                                        int batchIndex, int outputChannels, int plane,
                                        int inputsPerGroup, int channel0, int bound) {

        int output0 = outputOffset + (batchIndex * outputChannels + channel0) * plane;
        int output1 = output0 + plane;
        int output2 = output1 + plane;
        int output3 = output2 + plane;
        int output4 = output3 + plane;
        int output5 = output4 + plane;
        int output6 = output5 + plane;
        int output7 = output6 + plane;
        int weight0 = weightOffset + channel0 * inputsPerGroup;
        int weight1 = weight0 + inputsPerGroup;
        int weight2 = weight1 + inputsPerGroup;
        int weight3 = weight2 + inputsPerGroup;
        int weight4 = weight3 + inputsPerGroup;
        int weight5 = weight4 + inputsPerGroup;
        int weight6 = weight5 + inputsPerGroup;
        int weight7 = weight6 + inputsPerGroup;
        float bias0 = bias == null ? 0.0f : bias[biasOffset + channel0];
        float bias1 = bias == null ? 0.0f : bias[biasOffset + channel0 + 1];
        float bias2 = bias == null ? 0.0f : bias[biasOffset + channel0 + 2];
        float bias3 = bias == null ? 0.0f : bias[biasOffset + channel0 + 3];
        float bias4 = bias == null ? 0.0f : bias[biasOffset + channel0 + 4];
        float bias5 = bias == null ? 0.0f : bias[biasOffset + channel0 + 5];
        float bias6 = bias == null ? 0.0f : bias[biasOffset + channel0 + 6];
        float bias7 = bias == null ? 0.0f : bias[biasOffset + channel0 + 7];
        FloatVector biasVector0 = FloatVector.broadcast(SPECIES, bias0);
        FloatVector biasVector1 = FloatVector.broadcast(SPECIES, bias1);
        FloatVector biasVector2 = FloatVector.broadcast(SPECIES, bias2);
        FloatVector biasVector3 = FloatVector.broadcast(SPECIES, bias3);
        FloatVector biasVector4 = FloatVector.broadcast(SPECIES, bias4);
        FloatVector biasVector5 = FloatVector.broadcast(SPECIES, bias5);
        FloatVector biasVector6 = FloatVector.broadcast(SPECIES, bias6);
        FloatVector biasVector7 = FloatVector.broadcast(SPECIES, bias7);
        int i = 0;
        for (; i < bound; i += SPECIES.length()) {
            FloatVector result0 = biasVector0;
            FloatVector result1 = biasVector1;
            FloatVector result2 = biasVector2;
            FloatVector result3 = biasVector3;
            FloatVector result4 = biasVector4;
            FloatVector result5 = biasVector5;
            FloatVector result6 = biasVector6;
            FloatVector result7 = biasVector7;
            int inputIndex = inputGroupBase + i;
            for (int ic = 0; ic < inputsPerGroup; ic++) {
                FloatVector sample = FloatVector.fromArray(SPECIES, input, inputIndex);
                result0 = result0.add(sample.mul(weights[weight0 + ic]));
                result1 = result1.add(sample.mul(weights[weight1 + ic]));
                result2 = result2.add(sample.mul(weights[weight2 + ic]));
                result3 = result3.add(sample.mul(weights[weight3 + ic]));
                result4 = result4.add(sample.mul(weights[weight4 + ic]));
                result5 = result5.add(sample.mul(weights[weight5 + ic]));
                result6 = result6.add(sample.mul(weights[weight6 + ic]));
                result7 = result7.add(sample.mul(weights[weight7 + ic]));
                inputIndex += plane;
            }
            result0.intoArray(output, output0 + i);
            result1.intoArray(output, output1 + i);
            result2.intoArray(output, output2 + i);
            result3.intoArray(output, output3 + i);
            result4.intoArray(output, output4 + i);
            result5.intoArray(output, output5 + i);
            result6.intoArray(output, output6 + i);
            result7.intoArray(output, output7 + i);
        }
        for (; i < plane; i++) {
            float result0 = bias0;
            float result1 = bias1;
            float result2 = bias2;
            float result3 = bias3;
            float result4 = bias4;
            float result5 = bias5;
            float result6 = bias6;
            float result7 = bias7;
            int inputIndex = inputGroupBase + i;
            for (int ic = 0; ic < inputsPerGroup; ic++) {
                float sample = input[inputIndex];
                result0 += sample * weights[weight0 + ic];
                result1 += sample * weights[weight1 + ic];
                result2 += sample * weights[weight2 + ic];
                result3 += sample * weights[weight3 + ic];
                result4 += sample * weights[weight4 + ic];
                result5 += sample * weights[weight5 + ic];
                result6 += sample * weights[weight6 + ic];
                result7 += sample * weights[weight7 + ic];
                inputIndex += plane;
            }
            output[output0 + i] = result0;
            output[output1 + i] = result1;
            output[output2 + i] = result2;
            output[output3 + i] = result3;
            output[output4 + i] = result4;
            output[output5 + i] = result5;
            output[output6 + i] = result6;
            output[output7 + i] = result7;
        }

    }

    private static void pointwiseFour(float[] input, int inputGroupBase,
                                        float[] weights, int weightOffset,
                                        float[] bias, int biasOffset,
                                        float[] output, int outputOffset,
                                        int batchIndex, int outputChannels, int plane,
                                        int inputsPerGroup, int channel0, int bound) {

    int output0 = outputOffset + (batchIndex * outputChannels + channel0) * plane;
    int output1 = output0 + plane;
    int output2 = output1 + plane;
    int output3 = output2 + plane;
    int weight0 = weightOffset + channel0 * inputsPerGroup;
    int weight1 = weight0 + inputsPerGroup;
    int weight2 = weight1 + inputsPerGroup;
    int weight3 = weight2 + inputsPerGroup;
    float bias0 = bias == null ? 0.0f : bias[biasOffset + channel0];
    float bias1 = bias == null ? 0.0f : bias[biasOffset + channel0 + 1];
    float bias2 = bias == null ? 0.0f : bias[biasOffset + channel0 + 2];
    float bias3 = bias == null ? 0.0f : bias[biasOffset + channel0 + 3];
    FloatVector biasVector0 = FloatVector.broadcast(SPECIES, bias0);
    FloatVector biasVector1 = FloatVector.broadcast(SPECIES, bias1);
    FloatVector biasVector2 = FloatVector.broadcast(SPECIES, bias2);
    FloatVector biasVector3 = FloatVector.broadcast(SPECIES, bias3);
    int i = 0;
    for (; i < bound; i += SPECIES.length()) {
        FloatVector result0 = biasVector0;
        FloatVector result1 = biasVector1;
        FloatVector result2 = biasVector2;
        FloatVector result3 = biasVector3;
        int inputIndex = inputGroupBase + i;
        for (int ic = 0; ic < inputsPerGroup; ic++) {
            FloatVector sample = FloatVector.fromArray(SPECIES, input, inputIndex);
            result0 = result0.add(sample.mul(weights[weight0 + ic]));
            result1 = result1.add(sample.mul(weights[weight1 + ic]));
            result2 = result2.add(sample.mul(weights[weight2 + ic]));
            result3 = result3.add(sample.mul(weights[weight3 + ic]));
            inputIndex += plane;
        }
        result0.intoArray(output, output0 + i);
        result1.intoArray(output, output1 + i);
        result2.intoArray(output, output2 + i);
        result3.intoArray(output, output3 + i);
    }
    for (; i < plane; i++) {
        float result0 = bias0;
        float result1 = bias1;
        float result2 = bias2;
        float result3 = bias3;
        int inputIndex = inputGroupBase + i;
        for (int ic = 0; ic < inputsPerGroup; ic++) {
            float sample = input[inputIndex];
            result0 += sample * weights[weight0 + ic];
            result1 += sample * weights[weight1 + ic];
            result2 += sample * weights[weight2 + ic];
            result3 += sample * weights[weight3 + ic];
            inputIndex += plane;
        }
        output[output0 + i] = result0;
        output[output1 + i] = result1;
        output[output2 + i] = result2;
        output[output3 + i] = result3;
    }

    }

    private static void pointwiseOne(float[] input, int inputGroupBase,
                                        float[] weights, int weightOffset,
                                        float[] bias, int biasOffset,
                                        float[] output, int outputOffset,
                                        int batchIndex, int outputChannels, int plane,
                                        int inputsPerGroup, int channel, int bound) {

    int outputBase = outputOffset + (batchIndex * outputChannels + channel) * plane;
    int weightBase = weightOffset + channel * inputsPerGroup;
    float initial = bias == null ? 0.0f : bias[biasOffset + channel];
    FloatVector initialVector = FloatVector.broadcast(SPECIES, initial);
    int i = 0;
    for (; i < bound; i += SPECIES.length()) {
        FloatVector result = initialVector;
        int inputIndex = inputGroupBase + i;
        for (int ic = 0; ic < inputsPerGroup; ic++) {
            result = result.add(FloatVector.fromArray(SPECIES, input, inputIndex)
                    .mul(weights[weightBase + ic]));
            inputIndex += plane;
        }
        result.intoArray(output, outputBase + i);
    }
    for (; i < plane; i++) {
        float result = initial;
        int inputIndex = inputGroupBase + i;
        for (int ic = 0; ic < inputsPerGroup; ic++) {
            result += input[inputIndex] * weights[weightBase + ic];
            inputIndex += plane;
        }
        output[outputBase + i] = result;
    }

    }




    private static void pointwiseTwelve(float[] input, int inputGroupBase,
                                        float[] weights, int weightOffset,
                                        float[] bias, int biasOffset,
                                        float[] output, int outputOffset,
                                        int batchIndex, int outputChannels, int plane,
                                        int inputsPerGroup, int channel0, int bound) {
        int output0 = outputOffset + (batchIndex * outputChannels + channel0) * plane;
        int output1 = output0 + plane;
        int output2 = output1 + plane;
        int output3 = output2 + plane;
        int output4 = output3 + plane;
        int output5 = output4 + plane;
        int output6 = output5 + plane;
        int output7 = output6 + plane;
        int output8 = output7 + plane;
        int output9 = output8 + plane;
        int output10 = output9 + plane;
        int output11 = output10 + plane;
        int weight0 = weightOffset + channel0 * inputsPerGroup;
        int weight1 = weight0 + inputsPerGroup;
        int weight2 = weight1 + inputsPerGroup;
        int weight3 = weight2 + inputsPerGroup;
        int weight4 = weight3 + inputsPerGroup;
        int weight5 = weight4 + inputsPerGroup;
        int weight6 = weight5 + inputsPerGroup;
        int weight7 = weight6 + inputsPerGroup;
        int weight8 = weight7 + inputsPerGroup;
        int weight9 = weight8 + inputsPerGroup;
        int weight10 = weight9 + inputsPerGroup;
        int weight11 = weight10 + inputsPerGroup;
        float bias0 = bias == null ? 0.0f : bias[biasOffset + channel0];
        float bias1 = bias == null ? 0.0f : bias[biasOffset + channel0 + 1];
        float bias2 = bias == null ? 0.0f : bias[biasOffset + channel0 + 2];
        float bias3 = bias == null ? 0.0f : bias[biasOffset + channel0 + 3];
        float bias4 = bias == null ? 0.0f : bias[biasOffset + channel0 + 4];
        float bias5 = bias == null ? 0.0f : bias[biasOffset + channel0 + 5];
        float bias6 = bias == null ? 0.0f : bias[biasOffset + channel0 + 6];
        float bias7 = bias == null ? 0.0f : bias[biasOffset + channel0 + 7];
        float bias8 = bias == null ? 0.0f : bias[biasOffset + channel0 + 8];
        float bias9 = bias == null ? 0.0f : bias[biasOffset + channel0 + 9];
        float bias10 = bias == null ? 0.0f : bias[biasOffset + channel0 + 10];
        float bias11 = bias == null ? 0.0f : bias[biasOffset + channel0 + 11];
        FloatVector biasVector0 = FloatVector.broadcast(SPECIES, bias0);
        FloatVector biasVector1 = FloatVector.broadcast(SPECIES, bias1);
        FloatVector biasVector2 = FloatVector.broadcast(SPECIES, bias2);
        FloatVector biasVector3 = FloatVector.broadcast(SPECIES, bias3);
        FloatVector biasVector4 = FloatVector.broadcast(SPECIES, bias4);
        FloatVector biasVector5 = FloatVector.broadcast(SPECIES, bias5);
        FloatVector biasVector6 = FloatVector.broadcast(SPECIES, bias6);
        FloatVector biasVector7 = FloatVector.broadcast(SPECIES, bias7);
        FloatVector biasVector8 = FloatVector.broadcast(SPECIES, bias8);
        FloatVector biasVector9 = FloatVector.broadcast(SPECIES, bias9);
        FloatVector biasVector10 = FloatVector.broadcast(SPECIES, bias10);
        FloatVector biasVector11 = FloatVector.broadcast(SPECIES, bias11);
        int i = 0;
        for (; i < bound; i += SPECIES.length()) {
            FloatVector result0 = biasVector0;
            FloatVector result1 = biasVector1;
            FloatVector result2 = biasVector2;
            FloatVector result3 = biasVector3;
            FloatVector result4 = biasVector4;
            FloatVector result5 = biasVector5;
            FloatVector result6 = biasVector6;
            FloatVector result7 = biasVector7;
            FloatVector result8 = biasVector8;
            FloatVector result9 = biasVector9;
            FloatVector result10 = biasVector10;
            FloatVector result11 = biasVector11;
            int inputIndex = inputGroupBase + i;
            for (int ic = 0; ic < inputsPerGroup; ic++) {
                FloatVector sample = FloatVector.fromArray(SPECIES, input, inputIndex);
                result0 = result0.add(sample.mul(weights[weight0 + ic]));
                result1 = result1.add(sample.mul(weights[weight1 + ic]));
                result2 = result2.add(sample.mul(weights[weight2 + ic]));
                result3 = result3.add(sample.mul(weights[weight3 + ic]));
                result4 = result4.add(sample.mul(weights[weight4 + ic]));
                result5 = result5.add(sample.mul(weights[weight5 + ic]));
                result6 = result6.add(sample.mul(weights[weight6 + ic]));
                result7 = result7.add(sample.mul(weights[weight7 + ic]));
                result8 = result8.add(sample.mul(weights[weight8 + ic]));
                result9 = result9.add(sample.mul(weights[weight9 + ic]));
                result10 = result10.add(sample.mul(weights[weight10 + ic]));
                result11 = result11.add(sample.mul(weights[weight11 + ic]));
                inputIndex += plane;
            }
            result0.intoArray(output, output0 + i);
            result1.intoArray(output, output1 + i);
            result2.intoArray(output, output2 + i);
            result3.intoArray(output, output3 + i);
            result4.intoArray(output, output4 + i);
            result5.intoArray(output, output5 + i);
            result6.intoArray(output, output6 + i);
            result7.intoArray(output, output7 + i);
            result8.intoArray(output, output8 + i);
            result9.intoArray(output, output9 + i);
            result10.intoArray(output, output10 + i);
            result11.intoArray(output, output11 + i);
        }
        for (; i < plane; i++) {
            float result0 = bias0;
            float result1 = bias1;
            float result2 = bias2;
            float result3 = bias3;
            float result4 = bias4;
            float result5 = bias5;
            float result6 = bias6;
            float result7 = bias7;
            float result8 = bias8;
            float result9 = bias9;
            float result10 = bias10;
            float result11 = bias11;
            int inputIndex = inputGroupBase + i;
            for (int ic = 0; ic < inputsPerGroup; ic++) {
                float sample = input[inputIndex];
                result0 += sample * weights[weight0 + ic];
                result1 += sample * weights[weight1 + ic];
                result2 += sample * weights[weight2 + ic];
                result3 += sample * weights[weight3 + ic];
                result4 += sample * weights[weight4 + ic];
                result5 += sample * weights[weight5 + ic];
                result6 += sample * weights[weight6 + ic];
                result7 += sample * weights[weight7 + ic];
                result8 += sample * weights[weight8 + ic];
                result9 += sample * weights[weight9 + ic];
                result10 += sample * weights[weight10 + ic];
                result11 += sample * weights[weight11 + ic];
                inputIndex += plane;
            }
            output[output0 + i] = result0;
            output[output1 + i] = result1;
            output[output2 + i] = result2;
            output[output3 + i] = result3;
            output[output4 + i] = result4;
            output[output5 + i] = result5;
            output[output6 + i] = result6;
            output[output7 + i] = result7;
            output[output8 + i] = result8;
            output[output9 + i] = result9;
            output[output10 + i] = result10;
            output[output11 + i] = result11;
        }
    }

}
