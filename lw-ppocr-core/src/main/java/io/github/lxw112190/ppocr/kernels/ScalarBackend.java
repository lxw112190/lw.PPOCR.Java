package io.github.lxw112190.ppocr.kernels;

import io.github.lxw112190.ppocr.runtime.BinaryPlan;
import io.github.lxw112190.ppocr.runtime.BinaryVariant;
import java.util.Arrays;

/** Readable reference kernels; optimized backends must preserve their semantics. */
public final class ScalarBackend implements KernelBackend, ProjectionArgMaxBackend,
        InPlaceElementwiseBackend {
    @Override
    public void add(float[] left, int leftOffset, float[] right, int rightOffset,
                    float[] output, int outputOffset, int length) {
        binary(left, leftOffset, right, rightOffset, output, outputOffset, length, 0);
    }

    @Override
    public void mul(float[] left, int leftOffset, float[] right, int rightOffset,
                    float[] output, int outputOffset, int length) {
        binary(left, leftOffset, right, rightOffset, output, outputOffset, length, 1);
    }

    @Override
    public void div(float[] left, int leftOffset, float[] right, int rightOffset,
                    float[] output, int outputOffset, int length) {
        binary(left, leftOffset, right, rightOffset, output, outputOffset, length, 2);
    }

    @Override
    public void sub(float[] left, int leftOffset, float[] right, int rightOffset,
                    float[] output, int outputOffset, int length) {
        binary(left, leftOffset, right, rightOffset, output, outputOffset, length, 3);
    }

    @Override
    public void binary(BinaryOp operation, float[] left, int leftOffset, float[] right, int rightOffset,
                       float[] output, int outputOffset, BinaryPlan plan) {
        if (plan.getVariant() == BinaryVariant.SAME_SHAPE) {
            switch (operation) {
                case ADD: add(left, leftOffset, right, rightOffset, output, outputOffset, plan.getOutputLength()); break;
                case MUL: mul(left, leftOffset, right, rightOffset, output, outputOffset, plan.getOutputLength()); break;
                case DIV: div(left, leftOffset, right, rightOffset, output, outputOffset, plan.getOutputLength()); break;
                case SUB: sub(left, leftOffset, right, rightOffset, output, outputOffset, plan.getOutputLength()); break;
                case POW: pow(left, leftOffset, right, rightOffset, output, outputOffset, plan.getOutputLength()); break;
                default: throw new AssertionError("unknown binary operation");
            }
            return;
        }
        if (plan.getVariant() == BinaryVariant.RIGHT_SCALAR) {
            float scalar = right[rightOffset];
            for (int i = 0; i < plan.getOutputLength(); i++) {
                output[outputOffset + i] = applyBinary(operation, left[leftOffset + i], scalar);
            }
            return;
        }
        if (plan.getVariant() == BinaryVariant.LEFT_SCALAR) {
            float scalar = left[leftOffset];
            for (int i = 0; i < plan.getOutputLength(); i++) {
                output[outputOffset + i] = applyBinary(operation, scalar, right[rightOffset + i]);
            }
            return;
        }
        for (int linear = 0; linear < plan.getOutputLength(); linear++) {
            int remainder = linear;
            int leftIndex = 0;
            int rightIndex = 0;
            for (int axis = plan.getRank() - 1; axis >= 0; axis--) {
                int coordinate = remainder % plan.getOutputDimension(axis);
                remainder /= plan.getOutputDimension(axis);
                leftIndex += coordinate * plan.getLeftStride(axis);
                rightIndex += coordinate * plan.getRightStride(axis);
            }
            float a = left[leftOffset + leftIndex];
            float b = right[rightOffset + rightIndex];
            output[outputOffset + linear] = applyBinary(operation, a, b);
        }
    }

    @Override
    public void relu(float[] input, int inputOffset, float[] output, int outputOffset, int length) {
        for (int i = 0; i < length; i++) {
            output[outputOffset + i] = Math.max(0.0f, input[inputOffset + i]);
        }
    }

    @Override
    public void sigmoid(float[] input, int inputOffset, float[] output, int outputOffset, int length) {
        for (int i = 0; i < length; i++) {
            output[outputOffset + i] = (float) (1.0 / (1.0 + Math.exp(-input[inputOffset + i])));
        }
    }

    @Override
    public void erf(float[] input, int inputOffset, float[] output, int outputOffset, int length) {
        for (int i = 0; i < length; i++) {
            double value = input[inputOffset + i];
            double sign = value < 0 ? -1.0 : 1.0;
            value = Math.abs(value);
            double t = 1.0 / (1.0 + 0.3275911 * value);
            double polynomial = (((((1.061405429 * t - 1.453152027) * t)
                    + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t;
            output[outputOffset + i] = (float) (sign * (1.0 - polynomial * Math.exp(-value * value)));
        }
    }

    @Override
    public void hardSigmoid(float[] input, int inputOffset, float[] output, int outputOffset,
                            int length, float alpha, float beta) {
        for (int i = 0; i < length; i++) {
            output[outputOffset + i] = Math.max(0.0f, Math.min(1.0f, alpha * input[inputOffset + i] + beta));
        }
    }

    @Override
    public void sqrt(float[] input, int inputOffset, float[] output, int outputOffset, int length) {
        for (int i = 0; i < length; i++) {
            output[outputOffset + i] = (float) Math.sqrt(input[inputOffset + i]);
        }
    }

    @Override
    public void pow(float[] left, int leftOffset, float[] right, int rightOffset,
                    float[] output, int outputOffset, int length) {
        for (int i = 0; i < length; i++) {
            output[outputOffset + i] = (float) Math.pow(left[leftOffset + i], right[rightOffset + i]);
        }
    }

    @Override
    public void reduceMean(float[] input, int inputOffset, float[] output, int outputOffset,
                           int[] inputDimensions, int[] axes, boolean keepDimensions) {
        int rank = inputDimensions.length;
        boolean[] reduced = new boolean[rank];
        int reducedElements = 1;
        for (int axis : axes) {
            if (axis < 0) axis += rank;
            if (axis < 0 || axis >= rank || reduced[axis]) throw new IllegalArgumentException("invalid reduction axis");
            reduced[axis] = true;
            reducedElements *= inputDimensions[axis];
        }
        int[] inputStrides = strides(inputDimensions);
        int[] outputDimensions = new int[keepDimensions ? rank : rank - axes.length];
        int outputAxis = 0;
        for (int axis = 0; axis < rank; axis++) {
            if (keepDimensions) outputDimensions[outputAxis++] = reduced[axis] ? 1 : inputDimensions[axis];
            else if (!reduced[axis]) outputDimensions[outputAxis++] = inputDimensions[axis];
        }
        int outputLength = product(outputDimensions);
        for (int i = 0; i < outputLength; i++) output[outputOffset + i] = 0.0f;
        for (int linear = 0; linear < product(inputDimensions); linear++) {
            int remainder = linear;
            int target = 0;
            for (int axis = 0; axis < rank; axis++) {
                int coordinate = remainder / inputStrides[axis];
                remainder %= inputStrides[axis];
                if (!reduced[axis]) {
                    target = target * outputDimensions[keepDimensions ? axis : targetAxis(reduced, axis)] + coordinate;
                }
            }
            output[outputOffset + target] += input[inputOffset + linear];
        }
        for (int i = 0; i < outputLength; i++) output[outputOffset + i] /= reducedElements;
    }

    @Override
    public void concat(float[][] inputs, int[] inputOffsets, float[] output, int outputOffset,
                       int[] inputDimensions, int axis, int[] axisSizes) {
        int rank = inputDimensions.length - 1;
        int outer = 1;
        for (int i = 0; i < axis; i++) outer *= inputDimensions[i];
        int inner = 1;
        for (int i = axis + 1; i < inputDimensions.length; i++) inner *= inputDimensions[i];
        int outputAxis = 0;
        for (int size : axisSizes) outputAxis += size;
        int outputBlock = outputAxis * inner;
        for (int outerIndex = 0; outerIndex < outer; outerIndex++) {
            int destination = outputOffset + outerIndex * outputBlock;
            for (int inputIndex = 0; inputIndex < inputs.length; inputIndex++) {
                int count = axisSizes[inputIndex] * inner;
                System.arraycopy(inputs[inputIndex], inputOffsets[inputIndex] + outerIndex * axisSizes[inputIndex] * inner,
                        output, destination, count);
                destination += count;
            }
        }
    }

    @Override
    public void slice(float[] input, int inputOffset, float[] output, int outputOffset,
                      int[] inputDimensions, int[] starts, int[] axes, int[] steps) {
        int[] inputStrides = strides(inputDimensions);
        int[] outputDimensions = inputDimensions.clone();
        for (int i = 0; i < axes.length; i++) {
            int axis = axes[i] < 0 ? axes[i] + inputDimensions.length : axes[i];
            outputDimensions[axis] = (inputDimensions[axis] - starts[i] + steps[i] - 1) / steps[i];
        }
        int outputLength = product(outputDimensions);
        int[] outputStrides = strides(outputDimensions);
        for (int linear = 0; linear < outputLength; linear++) {
            int remainder = linear;
            int source = 0;
            for (int axis = 0; axis < outputDimensions.length; axis++) {
                int coordinate = remainder / outputStrides[axis];
                remainder %= outputStrides[axis];
                int sourceCoordinate = coordinate;
                for (int i = 0; i < axes.length; i++) {
                    int selectedAxis = axes[i] < 0 ? axes[i] + inputDimensions.length : axes[i];
                    if (selectedAxis == axis) sourceCoordinate = starts[i] + coordinate * steps[i];
                }
                source += sourceCoordinate * inputStrides[axis];
            }
            output[outputOffset + linear] = input[inputOffset + source];
        }
    }

    private static int[] strides(int[] dimensions) {
        int[] result = new int[dimensions.length];
        int stride = 1;
        for (int axis = dimensions.length - 1; axis >= 0; axis--) {
            result[axis] = stride;
            stride *= dimensions[axis];
        }
        return result;
    }

    private static int product(int[] dimensions) {
        int result = 1;
        for (int dimension : dimensions) result *= dimension;
        return result;
    }

    private static int targetAxis(boolean[] reduced, int axis) {
        int result = 0;
        for (int i = 0; i < axis; i++) if (!reduced[i]) result++;
        return result;
    }

    @Override
    public void matMul(float[] left, int leftOffset, float[] right, int rightOffset,
                       float[] output, int outputOffset, int rows, int inner, int columns) {
        for (int row = 0; row < rows; row++) {
            int outputRow = outputOffset + row * columns;
            Arrays.fill(output, outputRow, outputRow + columns, 0.0f);
            int leftRow = leftOffset + row * inner;
            for (int k = 0; k < inner; k++) {
                float value = left[leftRow + k];
                int rightRow = rightOffset + k * columns;
                for (int column = 0; column < columns; column++) {
                    output[outputRow + column] += value * right[rightRow + column];
                }
            }
        }
    }

    @Override
    public boolean supportsProjectionArgMax(int rows, int inner, int columns) {
        return rows > 0 && inner > 0 && columns > 0;
    }

    @Override
    public void projectionArgMax(float[] activations, int activationOffset,
                                 float[] weights, int weightOffset,
                                 float[] bias, int biasOffset,
                                 int rows, int inner, int columns,
                                 int[] bestIndices, float[] bestLogits,
                                 float[] bestProbabilities, float[] rowScratch) {
        requireProjectionBuffers(activations, activationOffset, weights, weightOffset,
                bias, biasOffset, rows, inner, columns, bestIndices, bestLogits,
                bestProbabilities, rowScratch);
        for (int rowBase = 0; rowBase < rows; rowBase += 4) {
            int blockRows = Math.min(4, rows - rowBase);
            matMul(activations, activationOffset + rowBase * inner, weights, weightOffset,
                    rowScratch, 0, blockRows, inner, columns);
            for (int localRow = 0; localRow < blockRows; localRow++) {
                int row = rowBase + localRow;
                int scratchBase = localRow * columns;
                int best = 0;
                for (int column = 0; column < columns; column++) {
                    float value = rowScratch[scratchBase + column] + bias[biasOffset + column];
                    if (!Float.isFinite(value)) {
                        throw new IllegalArgumentException("projection contains non-finite values");
                    }
                    rowScratch[scratchBase + column] = value;
                    if (column != 0 && value > rowScratch[scratchBase + best]) best = column;
                }
                float maximum = rowScratch[scratchBase + best];
                float sum = 0.0f;
                for (int column = 0; column < columns; column++) {
                    float value = (float) Math.exp(rowScratch[scratchBase + column] - maximum);
                    rowScratch[scratchBase + column] = value;
                    sum += value;
                }
                bestIndices[row] = best;
                bestLogits[row] = maximum;
                bestProbabilities[row] = rowScratch[scratchBase + best] / sum;
            }
        }
    }

    private static void requireProjectionBuffers(float[] activations, int activationOffset,
                                                 float[] weights, int weightOffset,
                                                 float[] bias, int biasOffset,
                                                 int rows, int inner, int columns,
                                                 int[] bestIndices, float[] bestLogits,
                                                 float[] bestProbabilities, float[] rowScratch) {
        if (activations == null || weights == null || bias == null || bestIndices == null ||
                bestLogits == null || bestProbabilities == null || rowScratch == null ||
                rows <= 0 || inner <= 0 || columns <= 0 || activationOffset < 0 ||
                weightOffset < 0 || biasOffset < 0 ||
                (long) rows * inner > activations.length - activationOffset ||
                (long) inner * columns > weights.length - weightOffset ||
                columns > bias.length - biasOffset || bestIndices.length < rows ||
                bestLogits.length < rows || bestProbabilities.length < rows ||
                rowScratch.length < (long) Math.min(rows, 4) * columns) {
            throw new IllegalArgumentException("projection buffers or dimensions are invalid");
        }
    }

    @Override
    public void conv(float[] input, int inputOffset, float[] weights, int weightOffset,
                     float[] bias, int biasOffset, float[] output, int outputOffset,
                     int batch, int channels, int height, int width, int outputChannels,
                     int kernelHeight, int kernelWidth, int strideHeight, int strideWidth,
                     int dilationHeight, int dilationWidth, int padTop, int padLeft,
                     int groups, int outputHeight, int outputWidth) {
        conv(input, inputOffset, weights, weightOffset, bias, biasOffset, output, outputOffset,
                batch, channels, height, width, outputChannels, kernelHeight, kernelWidth,
                strideHeight, strideWidth, dilationHeight, dilationWidth, padTop, padLeft,
                padTop, padLeft, groups, outputHeight, outputWidth);
    }

    @Override
    public void conv(float[] input, int inputOffset, float[] weights, int weightOffset,
                     float[] bias, int biasOffset, float[] output, int outputOffset,
                     int batch, int channels, int height, int width, int outputChannels,
                     int kernelHeight, int kernelWidth, int strideHeight, int strideWidth,
                     int dilationHeight, int dilationWidth, int padTop, int padLeft,
                     int padBottom, int padRight, int groups, int outputHeight, int outputWidth) {
        int inputChannelsPerGroup = channels / groups;
        int outputChannelsPerGroup = outputChannels / groups;
        if (kernelHeight == 1 && kernelWidth == 1 && dilationHeight == 1 && dilationWidth == 1) {
            convPointwise(input, inputOffset, weights, weightOffset, bias, biasOffset,
                    output, outputOffset, batch, channels, height, width, outputChannels,
                    strideHeight, strideWidth, padTop, padLeft, groups, outputHeight, outputWidth,
                    inputChannelsPerGroup, outputChannelsPerGroup);
            return;
        }
        if (groups == channels && outputChannels == channels && inputChannelsPerGroup == 1 &&
                outputChannelsPerGroup == 1) {
            convDepthwise(input, inputOffset, weights, weightOffset, bias, biasOffset,
                    output, outputOffset, batch, channels, height, width, kernelHeight,
                    kernelWidth, strideHeight, strideWidth, dilationHeight, dilationWidth,
                    padTop, padLeft, outputHeight, outputWidth);
            return;
        }
        for (int n = 0; n < batch; n++) {
            for (int group = 0; group < groups; group++) {
                for (int oc = 0; oc < outputChannelsPerGroup; oc++) {
                    int outputChannel = group * outputChannelsPerGroup + oc;
                    for (int oh = 0; oh < outputHeight; oh++) {
                        for (int ow = 0; ow < outputWidth; ow++) {
                            float sum = bias == null ? 0.0f : bias[biasOffset + outputChannel];
                            for (int ic = 0; ic < inputChannelsPerGroup; ic++) {
                                int inputChannel = group * inputChannelsPerGroup + ic;
                                for (int kh = 0; kh < kernelHeight; kh++) {
                                    int ih = oh * strideHeight - padTop + kh * dilationHeight;
                                    if (ih < 0 || ih >= height) continue;
                                    for (int kw = 0; kw < kernelWidth; kw++) {
                                        int iw = ow * strideWidth - padLeft + kw * dilationWidth;
                                        if (iw < 0 || iw >= width) continue;
                                        int inputIndex = inputOffset + ((n * channels + inputChannel) * height + ih) * width + iw;
                                        int weightIndex = weightOffset + (((outputChannel * inputChannelsPerGroup + ic) * kernelHeight + kh) * kernelWidth + kw);
                                        sum += input[inputIndex] * weights[weightIndex];
                                    }
                                }
                            }
                            output[outputOffset + ((n * outputChannels + outputChannel) * outputHeight + oh) * outputWidth + ow] = sum;
                        }
                    }
                }
            }
        }
    }

    private static void convPointwise(float[] input, int inputOffset, float[] weights,
                                      int weightOffset, float[] bias, int biasOffset,
                                      float[] output, int outputOffset, int batch, int channels,
                                      int height, int width, int outputChannels,
                                      int strideHeight, int strideWidth, int padTop, int padLeft,
                                      int groups, int outputHeight, int outputWidth,
                                      int inputChannelsPerGroup, int outputChannelsPerGroup) {
        int inputPlane = height * width;
        int outputPlane = outputHeight * outputWidth;
        for (int n = 0; n < batch; n++) {
            for (int group = 0; group < groups; group++) {
                int oc = 0;
                for (; oc + 3 < outputChannelsPerGroup; oc += 4) {
                    int channel0 = group * outputChannelsPerGroup + oc;
                    int channel1 = channel0 + 1;
                    int channel2 = channel0 + 2;
                    int channel3 = channel0 + 3;
                    int output0 = outputOffset + (n * outputChannels + channel0) * outputPlane;
                    int output1 = output0 + outputPlane;
                    int output2 = output1 + outputPlane;
                    int output3 = output2 + outputPlane;
                    Arrays.fill(output, output0, output0 + outputPlane,
                            bias == null ? 0.0f : bias[biasOffset + channel0]);
                    Arrays.fill(output, output1, output1 + outputPlane,
                            bias == null ? 0.0f : bias[biasOffset + channel1]);
                    Arrays.fill(output, output2, output2 + outputPlane,
                            bias == null ? 0.0f : bias[biasOffset + channel2]);
                    Arrays.fill(output, output3, output3 + outputPlane,
                            bias == null ? 0.0f : bias[biasOffset + channel3]);
                    int weight0 = weightOffset + channel0 * inputChannelsPerGroup;
                    int weight1 = weightOffset + channel1 * inputChannelsPerGroup;
                    int weight2 = weightOffset + channel2 * inputChannelsPerGroup;
                    int weight3 = weightOffset + channel3 * inputChannelsPerGroup;
                    for (int ic = 0; ic < inputChannelsPerGroup; ic++) {
                        int inputChannel = group * inputChannelsPerGroup + ic;
                        int inputBase = inputOffset + (n * channels + inputChannel) * inputPlane;
                        float value0 = weights[weight0 + ic];
                        float value1 = weights[weight1 + ic];
                        float value2 = weights[weight2 + ic];
                        float value3 = weights[weight3 + ic];
                        for (int oh = 0; oh < outputHeight; oh++) {
                            int ih = oh * strideHeight - padTop;
                            if (ih < 0 || ih >= height) continue;
                            int inputRow = inputBase + ih * width;
                            int row0 = output0 + oh * outputWidth;
                            int row1 = output1 + oh * outputWidth;
                            int row2 = output2 + oh * outputWidth;
                            int row3 = output3 + oh * outputWidth;
                            if (strideWidth == 1) {
                                int start = Math.max(0, padLeft);
                                int end = Math.min(outputWidth, width + padLeft);
                                int source = inputRow + start - padLeft;
                                for (int ow = start; ow < end; ow++) {
                                    float sample = input[source++];
                                    output[row0 + ow] += sample * value0;
                                    output[row1 + ow] += sample * value1;
                                    output[row2 + ow] += sample * value2;
                                    output[row3 + ow] += sample * value3;
                                }
                            } else {
                                for (int ow = 0; ow < outputWidth; ow++) {
                                    int iw = ow * strideWidth - padLeft;
                                    if (iw >= 0 && iw < width) {
                                        float sample = input[inputRow + iw];
                                        output[row0 + ow] += sample * value0;
                                        output[row1 + ow] += sample * value1;
                                        output[row2 + ow] += sample * value2;
                                        output[row3 + ow] += sample * value3;
                                    }
                                }
                            }
                        }
                    }
                }
                for (; oc < outputChannelsPerGroup; oc++) {
                    int outputChannel = group * outputChannelsPerGroup + oc;
                    int outputBase = outputOffset + (n * outputChannels + outputChannel) * outputPlane;
                    float initial = bias == null ? 0.0f : bias[biasOffset + outputChannel];
                    Arrays.fill(output, outputBase, outputBase + outputPlane, initial);
                    int weightBase = weightOffset + outputChannel * inputChannelsPerGroup;
                    for (int ic = 0; ic < inputChannelsPerGroup; ic++) {
                        int inputChannel = group * inputChannelsPerGroup + ic;
                        int inputBase = inputOffset + (n * channels + inputChannel) * inputPlane;
                        float weight = weights[weightBase + ic];
                        for (int oh = 0; oh < outputHeight; oh++) {
                            int ih = oh * strideHeight - padTop;
                            if (ih < 0 || ih >= height) continue;
                            int inputRow = inputBase + ih * width;
                            int outputRow = outputBase + oh * outputWidth;
                            if (strideWidth == 1) {
                                int start = Math.max(0, padLeft);
                                int end = Math.min(outputWidth, width + padLeft);
                                int source = inputRow + start - padLeft;
                                for (int ow = start; ow < end; ow++) {
                                    output[outputRow + ow] += input[source++] * weight;
                                }
                            } else {
                                for (int ow = 0; ow < outputWidth; ow++) {
                                    int iw = ow * strideWidth - padLeft;
                                    if (iw >= 0 && iw < width) {
                                        output[outputRow + ow] += input[inputRow + iw] * weight;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    private static void convDepthwise(float[] input, int inputOffset, float[] weights,
                                      int weightOffset, float[] bias, int biasOffset,
                                      float[] output, int outputOffset, int batch, int channels,
                                      int height, int width, int kernelHeight, int kernelWidth,
                                      int strideHeight, int strideWidth, int dilationHeight,
                                      int dilationWidth, int padTop, int padLeft,
                                      int outputHeight, int outputWidth) {
        int inputPlane = height * width;
        int outputPlane = outputHeight * outputWidth;
        int kernelPlane = kernelHeight * kernelWidth;
        for (int n = 0; n < batch; n++) {
            for (int channel = 0; channel < channels; channel++) {
                int inputBase = inputOffset + (n * channels + channel) * inputPlane;
                int outputBase = outputOffset + (n * channels + channel) * outputPlane;
                float initial = bias == null ? 0.0f : bias[biasOffset + channel];
                Arrays.fill(output, outputBase, outputBase + outputPlane, initial);
                int kernelBase = weightOffset + channel * kernelPlane;
                for (int kh = 0; kh < kernelHeight; kh++) {
                    for (int kw = 0; kw < kernelWidth; kw++) {
                        float weight = weights[kernelBase + kh * kernelWidth + kw];
                        for (int oh = 0; oh < outputHeight; oh++) {
                            int ih = oh * strideHeight - padTop + kh * dilationHeight;
                            if (ih < 0 || ih >= height) continue;
                            int inputRow = inputBase + ih * width;
                            int outputRow = outputBase + oh * outputWidth;
                            if (strideWidth == 1) {
                                int shift = padLeft - kw * dilationWidth;
                                int start = Math.max(0, shift);
                                int end = Math.min(outputWidth, width + shift);
                                int source = inputRow + start - shift;
                                for (int ow = start; ow < end; ow++) {
                                    output[outputRow + ow] += input[source++] * weight;
                                }
                            } else {
                                for (int ow = 0; ow < outputWidth; ow++) {
                                    int iw = ow * strideWidth - padLeft + kw * dilationWidth;
                                    if (iw >= 0 && iw < width) {
                                        output[outputRow + ow] += input[inputRow + iw] * weight;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    @Override
    public void batchNormalization(float[] input, int inputOffset, float[] scale, int scaleOffset,
                                   float[] bias, int biasOffset, float[] mean, int meanOffset,
                                   float[] variance, int varianceOffset, float epsilon,
                                   float[] output, int outputOffset, int[] dimensions) {
        int channels = dimensions[1];
        int spatial = 1;
        for (int axis = 2; axis < dimensions.length; axis++) spatial *= dimensions[axis];
        for (int channel = 0; channel < channels; channel++) {
            float factor = scale[scaleOffset + channel]
                    / (float) Math.sqrt(variance[varianceOffset + channel] + epsilon);
            for (int batch = 0; batch < dimensions[0]; batch++) {
                int base = inputOffset + (batch * channels + channel) * spatial;
                int outputBase = outputOffset + (batch * channels + channel) * spatial;
                for (int index = 0; index < spatial; index++) {
                    output[outputBase + index] = (input[base + index] - mean[meanOffset + channel]) * factor
                            + bias[biasOffset + channel];
                }
            }
        }
    }

    @Override
    public void softmax(float[] input, int inputOffset, float[] output, int outputOffset,
                        int outer, int axisLength, int inner) {
        for (int outerIndex = 0; outerIndex < outer; outerIndex++) {
            for (int innerIndex = 0; innerIndex < inner; innerIndex++) {
                float maximum = -Float.MAX_VALUE;
                for (int axis = 0; axis < axisLength; axis++) {
                    maximum = Math.max(maximum, input[inputOffset + (outerIndex * axisLength + axis) * inner + innerIndex]);
                }
                float sum = 0.0f;
                for (int axis = 0; axis < axisLength; axis++) {
                    float value = (float) Math.exp(input[inputOffset + (outerIndex * axisLength + axis) * inner + innerIndex] - maximum);
                    output[outputOffset + (outerIndex * axisLength + axis) * inner + innerIndex] = value;
                    sum += value;
                }
                for (int axis = 0; axis < axisLength; axis++) {
                    int index = outputOffset + (outerIndex * axisLength + axis) * inner + innerIndex;
                    output[index] /= sum;
                }
            }
        }
    }

    @Override
    public void pool(float[] input, int inputOffset, float[] output, int outputOffset,
                     int batch, int channels, int height, int width, int kernelHeight,
                     int kernelWidth, int strideHeight, int strideWidth, int padTop,
                     int padLeft, int outputHeight, int outputWidth, boolean maximum,
                     boolean countIncludePad) {
        pool(input, inputOffset, output, outputOffset, batch, channels, height, width,
                kernelHeight, kernelWidth, strideHeight, strideWidth, padTop, padLeft,
                padTop, padLeft, outputHeight, outputWidth, maximum, countIncludePad);
    }

    @Override
    public void pool(float[] input, int inputOffset, float[] output, int outputOffset,
                     int batch, int channels, int height, int width, int kernelHeight,
                     int kernelWidth, int strideHeight, int strideWidth, int padTop,
                     int padLeft, int padBottom, int padRight, int outputHeight,
                     int outputWidth, boolean maximum, boolean countIncludePad) {
        for (int n = 0; n < batch; n++) {
            for (int channel = 0; channel < channels; channel++) {
                for (int oh = 0; oh < outputHeight; oh++) {
                    for (int ow = 0; ow < outputWidth; ow++) {
                        float value = maximum ? -Float.MAX_VALUE : 0.0f;
                        int count = 0;
                        for (int kh = 0; kh < kernelHeight; kh++) {
                            int ih = oh * strideHeight - padTop + kh;
                            for (int kw = 0; kw < kernelWidth; kw++) {
                                int iw = ow * strideWidth - padLeft + kw;
                                if (ih < 0 || ih >= height || iw < 0 || iw >= width) {
                                    if (!maximum && countIncludePad) count++;
                                    continue;
                                }
                                float sample = input[inputOffset + ((n * channels + channel) * height + ih) * width + iw];
                                if (maximum) value = Math.max(value, sample);
                                else value += sample;
                                count++;
                            }
                        }
                        if (!maximum) value /= count == 0 ? 1 : count;
                        output[outputOffset + ((n * channels + channel) * outputHeight + oh) * outputWidth + ow] = value;
                    }
                }
            }
        }
    }

    @Override
    public void resizeNearest(float[] input, int inputOffset, float[] output, int outputOffset,
                              int batch, int channels, int inputHeight, int inputWidth,
                              int outputHeight, int outputWidth, float scaleHeight, float scaleWidth) {
        for (int n = 0; n < batch; n++) {
            for (int channel = 0; channel < channels; channel++) {
                for (int oh = 0; oh < outputHeight; oh++) {
                    int ih = Math.min(inputHeight - 1, Math.max(0, (int) Math.floor(oh / scaleHeight)));
                    for (int ow = 0; ow < outputWidth; ow++) {
                        int iw = Math.min(inputWidth - 1, Math.max(0, (int) Math.floor(ow / scaleWidth)));
                        output[outputOffset + ((n * channels + channel) * outputHeight + oh) * outputWidth + ow] =
                                input[inputOffset + ((n * channels + channel) * inputHeight + ih) * inputWidth + iw];
                    }
                }
            }
        }
    }

    @Override
    public void convTranspose(float[] input, int inputOffset, float[] weights, int weightOffset,
                              float[] bias, int biasOffset, float[] output, int outputOffset,
                              int batch, int inputChannels, int inputHeight, int inputWidth,
                              int outputChannels, int kernelHeight, int kernelWidth,
                              int strideHeight, int strideWidth, int dilationHeight,
                              int dilationWidth, int padTop, int padLeft, int groups,
                              int outputHeight, int outputWidth) {
        int inputChannelsPerGroup = inputChannels / groups;
        int outputChannelsPerGroup = outputChannels / groups;
        int outputElements = batch * outputChannels * outputHeight * outputWidth;
        for (int i = 0; i < outputElements; i++) output[outputOffset + i] = 0.0f;
        if (bias != null) {
            for (int n = 0; n < batch; n++) {
                for (int channel = 0; channel < outputChannels; channel++) {
                    for (int i = 0; i < outputHeight * outputWidth; i++) {
                        output[outputOffset + (n * outputChannels + channel) * outputHeight * outputWidth + i] = bias[biasOffset + channel];
                    }
                }
            }
        }
        for (int n = 0; n < batch; n++) {
            for (int group = 0; group < groups; group++) {
                for (int ic = 0; ic < inputChannelsPerGroup; ic++) {
                    int inputChannel = group * inputChannelsPerGroup + ic;
                    for (int ih = 0; ih < inputHeight; ih++) {
                        for (int iw = 0; iw < inputWidth; iw++) {
                            float sample = input[inputOffset + ((n * inputChannels + inputChannel) * inputHeight + ih) * inputWidth + iw];
                            for (int oc = 0; oc < outputChannelsPerGroup; oc++) {
                                int outputChannel = group * outputChannelsPerGroup + oc;
                                for (int kh = 0; kh < kernelHeight; kh++) {
                                    int oh = ih * strideHeight - padTop + kh * dilationHeight;
                                    if (oh < 0 || oh >= outputHeight) continue;
                                    for (int kw = 0; kw < kernelWidth; kw++) {
                                        int ow = iw * strideWidth - padLeft + kw * dilationWidth;
                                        if (ow < 0 || ow >= outputWidth) continue;
                                        int weightIndex = weightOffset + (((inputChannel * outputChannelsPerGroup + oc) * kernelHeight + kh) * kernelWidth + kw);
                                        int outputIndex = outputOffset + ((n * outputChannels + outputChannel) * outputHeight + oh) * outputWidth + ow;
                                        output[outputIndex] += sample * weights[weightIndex];
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    private static void binary(float[] left, int leftOffset, float[] right, int rightOffset,
                               float[] output, int outputOffset, int length, int operation) {
        for (int i = 0; i < length; i++) {
            float a = left[leftOffset + i];
            float b = right[rightOffset + i];
            switch (operation) {
                case 0: output[outputOffset + i] = a + b; break;
                case 1: output[outputOffset + i] = a * b; break;
                case 2: output[outputOffset + i] = a / b; break;
                case 3: output[outputOffset + i] = a - b; break;
                default: throw new AssertionError("unknown scalar operation");
            }
        }
    }

    private static float applyBinary(BinaryOp operation, float left, float right) {
        switch (operation) {
            case ADD: return left + right;
            case MUL: return left * right;
            case DIV: return left / right;
            case SUB: return left - right;
            case POW: return (float) Math.pow(left, right);
            default: throw new AssertionError("unknown binary operation");
        }
    }
}
