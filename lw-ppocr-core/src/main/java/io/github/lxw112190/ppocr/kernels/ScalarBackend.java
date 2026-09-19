package io.github.lxw112190.ppocr.kernels;

import io.github.lxw112190.ppocr.runtime.BinaryPlan;
import io.github.lxw112190.ppocr.runtime.BinaryVariant;
import java.util.Arrays;

/** Readable reference kernels; optimized backends must preserve their semantics. */
public final class ScalarBackend implements KernelBackend, FusedGeluBackend, ProjectionArgMaxBackend,
        InPlaceElementwiseBackend {
    @Override
    public void add(float[] left, int leftOffset, float[] right, int rightOffset,
                    float[] output, int outputOffset, int length) {
        for (int i = 0; i < length; i++) {
            output[outputOffset + i] = left[leftOffset + i] + right[rightOffset + i];
        }
    }

    @Override
    public void mul(float[] left, int leftOffset, float[] right, int rightOffset,
                    float[] output, int outputOffset, int length) {
        for (int i = 0; i < length; i++) {
            output[outputOffset + i] = left[leftOffset + i] * right[rightOffset + i];
        }
    }

    @Override
    public void div(float[] left, int leftOffset, float[] right, int rightOffset,
                    float[] output, int outputOffset, int length) {
        for (int i = 0; i < length; i++) {
            output[outputOffset + i] = left[leftOffset + i] / right[rightOffset + i];
        }
    }

    @Override
    public void sub(float[] left, int leftOffset, float[] right, int rightOffset,
                    float[] output, int outputOffset, int length) {
        for (int i = 0; i < length; i++) {
            output[outputOffset + i] = left[leftOffset + i] - right[rightOffset + i];
        }
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
            binaryRightScalar(operation, left, leftOffset, scalar, output, outputOffset,
                    plan.getOutputLength());
            return;
        }
        if (plan.getVariant() == BinaryVariant.LEFT_SCALAR) {
            float scalar = left[leftOffset];
            binaryLeftScalar(operation, scalar, right, rightOffset, output, outputOffset,
                    plan.getOutputLength());
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
            output[outputOffset + i] = erfScalar(input[inputOffset + i]);
        }
    }

    @Override
    public void gelu(float[] input, int inputOffset, float[] output, int outputOffset,
                     int length, float divisor, float addend, float multiplier) {
        for (int i = 0; i < length; i++) {
            float value = input[inputOffset + i];
            output[outputOffset + i] = ((erfScalar(value / divisor) + addend) * value)
                    * multiplier;
        }
    }

    private static float erfScalar(float input) {
        double value = input;
        double sign = value < 0 ? -1.0 : 1.0;
        value = Math.abs(value);
        double t = 1.0 / (1.0 + 0.3275911 * value);
        double polynomial = (((((1.061405429 * t - 1.453152027) * t)
                + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t;
        return (float) (sign * (1.0 - polynomial * Math.exp(-value * value)));
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
        int firstReduced = contiguousReducedSuffix(inputDimensions.length, axes);
        if (firstReduced >= 0) {
            int reducedElements = 1;
            for (int axis = firstReduced; axis < inputDimensions.length; axis++) {
                reducedElements *= inputDimensions[axis];
            }
            int outer = 1;
            for (int axis = 0; axis < firstReduced; axis++) {
                outer *= inputDimensions[axis];
            }
            for (int outerIndex = 0; outerIndex < outer; outerIndex++) {
                int source = inputOffset + outerIndex * reducedElements;
                float sum = 0.0f;
                for (int element = 0; element < reducedElements; element++) {
                    sum += input[source + element];
                }
                output[outputOffset + outerIndex] = sum / reducedElements;
            }
            return;
        }
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

    /** Returns the first axis of a contiguous reduced suffix, or -1 otherwise. */
    private static int contiguousReducedSuffix(int rank, int[] axes) {
        if (axes.length == 0) return -1;
        int first = rank;
        for (int i = 0; i < axes.length; i++) {
            int axis = axes[i] < 0 ? axes[i] + rank : axes[i];
            if (axis < 0 || axis >= rank) return -1;
            for (int previous = 0; previous < i; previous++) {
                int previousAxis = axes[previous] < 0 ? axes[previous] + rank : axes[previous];
                if (previousAxis == axis) return -1;
            }
            if (axis < first) first = axis;
        }
        if (rank - first != axes.length) return -1;
        for (int axis = first; axis < rank; axis++) {
            boolean present = false;
            for (int value : axes) {
                int normalized = value < 0 ? value + rank : value;
                if (normalized == axis) {
                    present = true;
                    break;
                }
            }
            if (!present) return -1;
        }
        return first;
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
                    sum += value;
                }
                bestIndices[row] = best;
                bestLogits[row] = maximum;
                bestProbabilities[row] = 1.0f / sum;
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
            if (strideHeight == 1 && strideWidth == 1 && padTop == 0 && padLeft == 0
                    && outputHeight == height && outputWidth == width) {
                convPointwiseUnitStride(input, inputOffset, weights, weightOffset, bias, biasOffset,
                        output, outputOffset, batch, channels, height, width, outputChannels,
                        groups, inputChannelsPerGroup, outputChannelsPerGroup);
                return;
            }
            convPointwise(input, inputOffset, weights, weightOffset, bias, biasOffset,
                    output, outputOffset, batch, channels, height, width, outputChannels,
                    strideHeight, strideWidth, padTop, padLeft, groups, outputHeight, outputWidth,
                    inputChannelsPerGroup, outputChannelsPerGroup);
            return;
        }
        if (groups == 1 && outputChannels >= 4 && outputChannels % 4 == 0
                && kernelHeight == 2 && kernelWidth == 2
                && strideHeight == 1 && strideWidth == 1
                && dilationHeight == 1 && dilationWidth == 1
                && padTop == 0 && padLeft == 0 && padBottom == 1 && padRight == 1
                && outputHeight == height && outputWidth == width) {
            convTwoByTwoStrideOne(input, inputOffset, weights, weightOffset,
                    bias, biasOffset, output, outputOffset, batch, channels, height, width,
                    outputChannels, outputHeight, outputWidth);
            return;
        }
        if (groups == 1 && outputChannels >= 4 && outputChannels % 4 == 0
                && kernelHeight == 3 && kernelWidth == 3
                && strideHeight == 2 && strideWidth == 2
                && dilationHeight == 1 && dilationWidth == 1
                && padTop == 1 && padLeft == 1 && padBottom == 1 && padRight == 1
                && outputHeight == (height + 1) / 2
                && outputWidth == (width + 1) / 2 && width >= 3) {
            convThreeByThreeStrideTwo(input, inputOffset, weights, weightOffset,
                    bias, biasOffset, output, outputOffset, batch, channels, height, width,
                    outputChannels, outputHeight, outputWidth);
            return;
        }
        if (groups == 1 && outputChannels >= 8 && outputChannels % 8 == 0
                && kernelHeight == 3 && kernelWidth == 3
                && strideHeight == 1 && strideWidth == 1
                && dilationHeight == 1 && dilationWidth == 1
                && padTop == 1 && padLeft == 1 && padBottom == 1 && padRight == 1
                && outputHeight == height && outputWidth == width && width >= 3) {
            convThreeByThreeStrideOneEight(input, inputOffset, weights, weightOffset,
                    bias, biasOffset, output, outputOffset, batch, channels, height, width,
                    outputChannels, outputHeight, outputWidth);
            return;
        }
        if (groups == 1 && outputChannels >= 4 && outputChannels % 4 == 0
                && kernelHeight == 3 && kernelWidth == 3
                && strideHeight == 1 && strideWidth == 1
                && dilationHeight == 1 && dilationWidth == 1
                && padTop == 1 && padLeft == 1 && padBottom == 1 && padRight == 1
                && outputHeight == height && outputWidth == width) {
            convThreeByThreeStrideOne(input, inputOffset, weights, weightOffset,
                    bias, biasOffset, output, outputOffset, batch, channels, height, width,
                    outputChannels, outputHeight, outputWidth);
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

    /** Scalar 1x1 unit-stride path with output-channel blocking and no boundary checks. */
    private static void convPointwiseUnitStride(
            float[] input, int inputOffset, float[] weights, int weightOffset,
            float[] bias, int biasOffset, float[] output, int outputOffset,
            int batch, int channels, int height, int width, int outputChannels,
            int groups, int inputChannelsPerGroup, int outputChannelsPerGroup) {
        int plane = height * width;
        for (int n = 0; n < batch; n++) {
            for (int group = 0; group < groups; group++) {
                int inputGroupBase = inputOffset
                        + (n * channels + group * inputChannelsPerGroup) * plane;
                int oc = 0;
                for (; oc + 7 < outputChannelsPerGroup; oc += 8) {
                    int channel0 = group * outputChannelsPerGroup + oc;
                    int output0 = outputOffset + (n * outputChannels + channel0) * plane;
                    int output1 = output0 + plane;
                    int output2 = output1 + plane;
                    int output3 = output2 + plane;
                    int output4 = output3 + plane;
                    int output5 = output4 + plane;
                    int output6 = output5 + plane;
                    int output7 = output6 + plane;
                    Arrays.fill(output, output0, output0 + plane,
                            bias == null ? 0.0f : bias[biasOffset + channel0]);
                    Arrays.fill(output, output1, output1 + plane,
                            bias == null ? 0.0f : bias[biasOffset + channel0 + 1]);
                    Arrays.fill(output, output2, output2 + plane,
                            bias == null ? 0.0f : bias[biasOffset + channel0 + 2]);
                    Arrays.fill(output, output3, output3 + plane,
                            bias == null ? 0.0f : bias[biasOffset + channel0 + 3]);
                    Arrays.fill(output, output4, output4 + plane,
                            bias == null ? 0.0f : bias[biasOffset + channel0 + 4]);
                    Arrays.fill(output, output5, output5 + plane,
                            bias == null ? 0.0f : bias[biasOffset + channel0 + 5]);
                    Arrays.fill(output, output6, output6 + plane,
                            bias == null ? 0.0f : bias[biasOffset + channel0 + 6]);
                    Arrays.fill(output, output7, output7 + plane,
                            bias == null ? 0.0f : bias[biasOffset + channel0 + 7]);
                    int weight0 = weightOffset + channel0 * inputChannelsPerGroup;
                    int weight1 = weight0 + inputChannelsPerGroup;
                    int weight2 = weight1 + inputChannelsPerGroup;
                    int weight3 = weight2 + inputChannelsPerGroup;
                    int weight4 = weight3 + inputChannelsPerGroup;
                    int weight5 = weight4 + inputChannelsPerGroup;
                    int weight6 = weight5 + inputChannelsPerGroup;
                    int weight7 = weight6 + inputChannelsPerGroup;
                    for (int ic = 0; ic < inputChannelsPerGroup; ic++) {
                        int inputBase = inputGroupBase + ic * plane;
                        float value0 = weights[weight0 + ic];
                        float value1 = weights[weight1 + ic];
                        float value2 = weights[weight2 + ic];
                        float value3 = weights[weight3 + ic];
                        float value4 = weights[weight4 + ic];
                        float value5 = weights[weight5 + ic];
                        float value6 = weights[weight6 + ic];
                        float value7 = weights[weight7 + ic];
                        for (int i = 0; i < plane; i++) {
                            float sample = input[inputBase + i];
                            output[output0 + i] += sample * value0;
                            output[output1 + i] += sample * value1;
                            output[output2 + i] += sample * value2;
                            output[output3 + i] += sample * value3;
                            output[output4 + i] += sample * value4;
                            output[output5 + i] += sample * value5;
                            output[output6 + i] += sample * value6;
                            output[output7 + i] += sample * value7;
                        }
                    }
                }
                for (; oc < outputChannelsPerGroup; oc++) {
                    int channel = group * outputChannelsPerGroup + oc;
                    int outputBase = outputOffset + (n * outputChannels + channel) * plane;
                    Arrays.fill(output, outputBase, outputBase + plane,
                            bias == null ? 0.0f : bias[biasOffset + channel]);
                    int weightBase = weightOffset + channel * inputChannelsPerGroup;
                    for (int ic = 0; ic < inputChannelsPerGroup; ic++) {
                        int inputBase = inputGroupBase + ic * plane;
                        float weight = weights[weightBase + ic];
                        for (int i = 0; i < plane; i++) {
                            output[outputBase + i] += input[inputBase + i] * weight;
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
                for (; oc + 7 < outputChannelsPerGroup; oc += 8) {
                    int channel0 = group * outputChannelsPerGroup + oc;
                    int channel1 = channel0 + 1;
                    int channel2 = channel0 + 2;
                    int channel3 = channel0 + 3;
                    int channel4 = channel0 + 4;
                    int channel5 = channel0 + 5;
                    int channel6 = channel0 + 6;
                    int channel7 = channel0 + 7;
                    int output0 = outputOffset + (n * outputChannels + channel0) * outputPlane;
                    int output1 = output0 + outputPlane;
                    int output2 = output1 + outputPlane;
                    int output3 = output2 + outputPlane;
                    int output4 = output3 + outputPlane;
                    int output5 = output4 + outputPlane;
                    int output6 = output5 + outputPlane;
                    int output7 = output6 + outputPlane;
                    float initial0 = bias == null ? 0.0f : bias[biasOffset + channel0];
                    float initial1 = bias == null ? 0.0f : bias[biasOffset + channel1];
                    float initial2 = bias == null ? 0.0f : bias[biasOffset + channel2];
                    float initial3 = bias == null ? 0.0f : bias[biasOffset + channel3];
                    float initial4 = bias == null ? 0.0f : bias[biasOffset + channel4];
                    float initial5 = bias == null ? 0.0f : bias[biasOffset + channel5];
                    float initial6 = bias == null ? 0.0f : bias[biasOffset + channel6];
                    float initial7 = bias == null ? 0.0f : bias[biasOffset + channel7];
                    Arrays.fill(output, output0, output0 + outputPlane, initial0);
                    Arrays.fill(output, output1, output1 + outputPlane, initial1);
                    Arrays.fill(output, output2, output2 + outputPlane, initial2);
                    Arrays.fill(output, output3, output3 + outputPlane, initial3);
                    Arrays.fill(output, output4, output4 + outputPlane, initial4);
                    Arrays.fill(output, output5, output5 + outputPlane, initial5);
                    Arrays.fill(output, output6, output6 + outputPlane, initial6);
                    Arrays.fill(output, output7, output7 + outputPlane, initial7);
                    int weight0 = weightOffset + channel0 * inputChannelsPerGroup;
                    int weight1 = weight0 + inputChannelsPerGroup;
                    int weight2 = weight1 + inputChannelsPerGroup;
                    int weight3 = weight2 + inputChannelsPerGroup;
                    int weight4 = weight3 + inputChannelsPerGroup;
                    int weight5 = weight4 + inputChannelsPerGroup;
                    int weight6 = weight5 + inputChannelsPerGroup;
                    int weight7 = weight6 + inputChannelsPerGroup;
                    for (int ic = 0; ic < inputChannelsPerGroup; ic++) {
                        int inputChannel = group * inputChannelsPerGroup + ic;
                        int inputBase = inputOffset + (n * channels + inputChannel) * inputPlane;
                        float value0 = weights[weight0 + ic];
                        float value1 = weights[weight1 + ic];
                        float value2 = weights[weight2 + ic];
                        float value3 = weights[weight3 + ic];
                        float value4 = weights[weight4 + ic];
                        float value5 = weights[weight5 + ic];
                        float value6 = weights[weight6 + ic];
                        float value7 = weights[weight7 + ic];
                        for (int oh = 0; oh < outputHeight; oh++) {
                            int ih = oh * strideHeight - padTop;
                            if (ih < 0 || ih >= height) continue;
                            int inputRow = inputBase + ih * width;
                            int row0 = output0 + oh * outputWidth;
                            int row1 = output1 + oh * outputWidth;
                            int row2 = output2 + oh * outputWidth;
                            int row3 = output3 + oh * outputWidth;
                            int row4 = output4 + oh * outputWidth;
                            int row5 = output5 + oh * outputWidth;
                            int row6 = output6 + oh * outputWidth;
                            int row7 = output7 + oh * outputWidth;
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
                                    output[row4 + ow] += sample * value4;
                                    output[row5 + ow] += sample * value5;
                                    output[row6 + ow] += sample * value6;
                                    output[row7 + ow] += sample * value7;
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
                                        output[row4 + ow] += sample * value4;
                                        output[row5 + ow] += sample * value5;
                                        output[row6 + ow] += sample * value6;
                                        output[row7 + ow] += sample * value7;
                                    }
                                }
                            }
                        }
                    }
                }
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

    /** Scalar 3x3 stride-two path; each output keeps the generic accumulation order. */
    private static void convThreeByThreeStrideTwo(float[] input, int inputOffset,
                                                  float[] weights, int weightOffset,
                                                  float[] bias, int biasOffset,
                                                  float[] output, int outputOffset, int batch,
                                                  int channels, int height, int width,
                                                  int outputChannels, int outputHeight,
                                                  int outputWidth) {
        int inputPlane = height * width;
        int outputPlane = outputHeight * outputWidth;
        int kernelPlane = channels * 9;
        for (int n = 0; n < batch; n++) {
            for (int outputChannel = 0; outputChannel < outputChannels; outputChannel += 4) {
                int output0 = outputOffset + (n * outputChannels + outputChannel) * outputPlane;
                int output1 = output0 + outputPlane;
                int output2 = output1 + outputPlane;
                int output3 = output2 + outputPlane;
                float initial0 = bias == null ? 0.0f : bias[biasOffset + outputChannel];
                float initial1 = bias == null ? 0.0f : bias[biasOffset + outputChannel + 1];
                float initial2 = bias == null ? 0.0f : bias[biasOffset + outputChannel + 2];
                float initial3 = bias == null ? 0.0f : bias[biasOffset + outputChannel + 3];
                Arrays.fill(output, output0, output0 + outputPlane, initial0);
                Arrays.fill(output, output1, output1 + outputPlane, initial1);
                Arrays.fill(output, output2, output2 + outputPlane, initial2);
                Arrays.fill(output, output3, output3 + outputPlane, initial3);
                for (int inputChannel = 0; inputChannel < channels; inputChannel++) {
                    int inputBase = inputOffset + (n * channels + inputChannel) * inputPlane;
                    int weight0 = weightOffset + outputChannel * kernelPlane + inputChannel * 9;
                    int weight1 = weight0 + kernelPlane;
                    int weight2 = weight1 + kernelPlane;
                    int weight3 = weight2 + kernelPlane;
                    for (int outputY = 0; outputY < outputHeight; outputY++) {
                        int inputY = outputY * 2 - 1;
                        int outputRow0 = output0 + outputY * outputWidth;
                        int outputRow1 = output1 + outputY * outputWidth;
                        int outputRow2 = output2 + outputY * outputWidth;
                        int outputRow3 = output3 + outputY * outputWidth;
                        for (int kernelY = 0; kernelY < 3; kernelY++) {
                            int sourceY = inputY + kernelY;
                            if (sourceY < 0 || sourceY >= height) continue;
                            int inputRow = inputBase + sourceY * width;
                            int kernelRow0 = weight0 + kernelY * 3;
                            int kernelRow1 = weight1 + kernelY * 3;
                            int kernelRow2 = weight2 + kernelY * 3;
                            int kernelRow3 = weight3 + kernelY * 3;
                            int interiorEnd = width / 2;
                            for (int outputX = 1; outputX < interiorEnd; outputX++) {
                                int sampleIndex = inputRow + outputX * 2 - 1;
                                float sample = input[sampleIndex];
                                output[outputRow0 + outputX] += sample * weights[kernelRow0];
                                output[outputRow1 + outputX] += sample * weights[kernelRow1];
                                output[outputRow2 + outputX] += sample * weights[kernelRow2];
                                output[outputRow3 + outputX] += sample * weights[kernelRow3];
                                sample = input[sampleIndex + 1];
                                output[outputRow0 + outputX] += sample * weights[kernelRow0 + 1];
                                output[outputRow1 + outputX] += sample * weights[kernelRow1 + 1];
                                output[outputRow2 + outputX] += sample * weights[kernelRow2 + 1];
                                output[outputRow3 + outputX] += sample * weights[kernelRow3 + 1];
                                sample = input[sampleIndex + 2];
                                output[outputRow0 + outputX] += sample * weights[kernelRow0 + 2];
                                output[outputRow1 + outputX] += sample * weights[kernelRow1 + 2];
                                output[outputRow2 + outputX] += sample * weights[kernelRow2 + 2];
                                output[outputRow3 + outputX] += sample * weights[kernelRow3 + 2];
                            }
                            for (int outputX = 0; outputX < outputWidth; outputX++) {
                                if (outputX != 0 && outputX < interiorEnd) continue;
                                int inputX = outputX * 2 - 1;
                                for (int kernelX = 0; kernelX < 3; kernelX++) {
                                    int sourceX = inputX + kernelX;
                                    if (sourceX < 0 || sourceX >= width) continue;
                                    float sample = input[inputRow + sourceX];
                                    output[outputRow0 + outputX] += sample * weights[kernelRow0 + kernelX];
                                    output[outputRow1 + outputX] += sample * weights[kernelRow1 + kernelX];
                                    output[outputRow2 + outputX] += sample * weights[kernelRow2 + kernelX];
                                    output[outputRow3 + outputX] += sample * weights[kernelRow3 + kernelX];
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    /** Scalar 2x2 SAME_UPPER path; the right and bottom edges are zero-padded. */
    private static void convTwoByTwoStrideOne(float[] input, int inputOffset,
                                              float[] weights, int weightOffset,
                                              float[] bias, int biasOffset,
                                              float[] output, int outputOffset, int batch,
                                              int channels, int height, int width,
                                              int outputChannels, int outputHeight,
                                              int outputWidth) {
        int inputPlane = height * width;
        int outputPlane = outputHeight * outputWidth;
        int kernelPlane = channels * 4;
        for (int n = 0; n < batch; n++) {
            for (int outputChannel = 0; outputChannel < outputChannels; outputChannel += 4) {
                int output0 = outputOffset + (n * outputChannels + outputChannel) * outputPlane;
                int output1 = output0 + outputPlane;
                int output2 = output1 + outputPlane;
                int output3 = output2 + outputPlane;
                float initial0 = bias == null ? 0.0f : bias[biasOffset + outputChannel];
                float initial1 = bias == null ? 0.0f : bias[biasOffset + outputChannel + 1];
                float initial2 = bias == null ? 0.0f : bias[biasOffset + outputChannel + 2];
                float initial3 = bias == null ? 0.0f : bias[biasOffset + outputChannel + 3];
                Arrays.fill(output, output0, output0 + outputPlane, initial0);
                Arrays.fill(output, output1, output1 + outputPlane, initial1);
                Arrays.fill(output, output2, output2 + outputPlane, initial2);
                Arrays.fill(output, output3, output3 + outputPlane, initial3);
                for (int outputY = 0; outputY < outputHeight; outputY++) {
                    int outputRow0 = output0 + outputY * outputWidth;
                    int outputRow1 = output1 + outputY * outputWidth;
                    int outputRow2 = output2 + outputY * outputWidth;
                    int outputRow3 = output3 + outputY * outputWidth;
                    int inputRow = inputOffset + (n * channels) * inputPlane + outputY * width;
                    boolean hasBottom = outputY + 1 < height;
                    for (int outputX = 0; outputX < outputWidth; outputX++) {
                        boolean hasRight = outputX + 1 < width;
                        int inputIndex = inputRow + outputX;
                        int outputIndex0 = outputRow0 + outputX;
                        int outputIndex1 = outputRow1 + outputX;
                        int outputIndex2 = outputRow2 + outputX;
                        int outputIndex3 = outputRow3 + outputX;
                        if (hasBottom && hasRight) {
                            for (int inputChannel = 0; inputChannel < channels; inputChannel++) {
                                int inputBase = inputIndex + inputChannel * inputPlane;
                                int weight0 = weightOffset + outputChannel * kernelPlane + inputChannel * 4;
                                int weight1 = weight0 + kernelPlane;
                                int weight2 = weight1 + kernelPlane;
                                int weight3 = weight2 + kernelPlane;
                                float sample = input[inputBase];
                                output[outputIndex0] += sample * weights[weight0];
                                output[outputIndex1] += sample * weights[weight1];
                                output[outputIndex2] += sample * weights[weight2];
                                output[outputIndex3] += sample * weights[weight3];
                                sample = input[inputBase + 1];
                                output[outputIndex0] += sample * weights[weight0 + 1];
                                output[outputIndex1] += sample * weights[weight1 + 1];
                                output[outputIndex2] += sample * weights[weight2 + 1];
                                output[outputIndex3] += sample * weights[weight3 + 1];
                                int bottom = inputBase + width;
                                sample = input[bottom];
                                output[outputIndex0] += sample * weights[weight0 + 2];
                                output[outputIndex1] += sample * weights[weight1 + 2];
                                output[outputIndex2] += sample * weights[weight2 + 2];
                                output[outputIndex3] += sample * weights[weight3 + 2];
                                sample = input[bottom + 1];
                                output[outputIndex0] += sample * weights[weight0 + 3];
                                output[outputIndex1] += sample * weights[weight1 + 3];
                                output[outputIndex2] += sample * weights[weight2 + 3];
                                output[outputIndex3] += sample * weights[weight3 + 3];
                            }
                        } else {
                            for (int inputChannel = 0; inputChannel < channels; inputChannel++) {
                                int inputBase = inputIndex + inputChannel * inputPlane;
                                int weight0 = weightOffset + outputChannel * kernelPlane + inputChannel * 4;
                                int weight1 = weight0 + kernelPlane;
                                int weight2 = weight1 + kernelPlane;
                                int weight3 = weight2 + kernelPlane;
                                float sample = input[inputBase];
                                output[outputIndex0] += sample * weights[weight0];
                                output[outputIndex1] += sample * weights[weight1];
                                output[outputIndex2] += sample * weights[weight2];
                                output[outputIndex3] += sample * weights[weight3];
                                if (hasRight) {
                                    sample = input[inputBase + 1];
                                    output[outputIndex0] += sample * weights[weight0 + 1];
                                    output[outputIndex1] += sample * weights[weight1 + 1];
                                    output[outputIndex2] += sample * weights[weight2 + 1];
                                    output[outputIndex3] += sample * weights[weight3 + 1];
                                }
                                if (hasBottom) {
                                    int bottom = inputBase + width;
                                    sample = input[bottom];
                                    output[outputIndex0] += sample * weights[weight0 + 2];
                                    output[outputIndex1] += sample * weights[weight1 + 2];
                                    output[outputIndex2] += sample * weights[weight2 + 2];
                                    output[outputIndex3] += sample * weights[weight3 + 2];
                                    if (hasRight) {
                                        sample = input[bottom + 1];
                                        output[outputIndex0] += sample * weights[weight0 + 3];
                                        output[outputIndex1] += sample * weights[weight1 + 3];
                                        output[outputIndex2] += sample * weights[weight2 + 3];
                                        output[outputIndex3] += sample * weights[weight3 + 3];
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    /** Scalar 3x3 same-padding path with eight output channels kept in flight. */
    private static void convThreeByThreeStrideOneEight(float[] input, int inputOffset,
                                                       float[] weights, int weightOffset,
                                                       float[] bias, int biasOffset,
                                                       float[] output, int outputOffset, int batch,
                                                       int channels, int height, int width,
                                                       int outputChannels, int outputHeight,
                                                       int outputWidth) {
        int inputPlane = height * width;
        int outputPlane = outputHeight * outputWidth;
        int kernelPlane = channels * 9;
        for (int n = 0; n < batch; n++) {
            for (int outputChannel = 0; outputChannel < outputChannels; outputChannel += 8) {
                int output0 = outputOffset + (n * outputChannels + outputChannel) * outputPlane;
                int output1 = output0 + outputPlane;
                int output2 = output1 + outputPlane;
                int output3 = output2 + outputPlane;
                int output4 = output3 + outputPlane;
                int output5 = output4 + outputPlane;
                int output6 = output5 + outputPlane;
                int output7 = output6 + outputPlane;
                float initial0 = bias == null ? 0.0f : bias[biasOffset + outputChannel];
                float initial1 = bias == null ? 0.0f : bias[biasOffset + outputChannel + 1];
                float initial2 = bias == null ? 0.0f : bias[biasOffset + outputChannel + 2];
                float initial3 = bias == null ? 0.0f : bias[biasOffset + outputChannel + 3];
                float initial4 = bias == null ? 0.0f : bias[biasOffset + outputChannel + 4];
                float initial5 = bias == null ? 0.0f : bias[biasOffset + outputChannel + 5];
                float initial6 = bias == null ? 0.0f : bias[biasOffset + outputChannel + 6];
                float initial7 = bias == null ? 0.0f : bias[biasOffset + outputChannel + 7];
                Arrays.fill(output, output0, output0 + outputPlane, initial0);
                Arrays.fill(output, output1, output1 + outputPlane, initial1);
                Arrays.fill(output, output2, output2 + outputPlane, initial2);
                Arrays.fill(output, output3, output3 + outputPlane, initial3);
                Arrays.fill(output, output4, output4 + outputPlane, initial4);
                Arrays.fill(output, output5, output5 + outputPlane, initial5);
                Arrays.fill(output, output6, output6 + outputPlane, initial6);
                Arrays.fill(output, output7, output7 + outputPlane, initial7);
                for (int inputChannel = 0; inputChannel < channels; inputChannel++) {
                    int inputBase = inputOffset + (n * channels + inputChannel) * inputPlane;
                    int weight0 = weightOffset + outputChannel * kernelPlane + inputChannel * 9;
                    int weight1 = weight0 + kernelPlane;
                    int weight2 = weight1 + kernelPlane;
                    int weight3 = weight2 + kernelPlane;
                    int weight4 = weight3 + kernelPlane;
                    int weight5 = weight4 + kernelPlane;
                    int weight6 = weight5 + kernelPlane;
                    int weight7 = weight6 + kernelPlane;
                    for (int outputY = 0; outputY < outputHeight; outputY++) {
                        int inputY = outputY - 1;
                        int outputRow0 = output0 + outputY * outputWidth;
                        int outputRow1 = output1 + outputY * outputWidth;
                        int outputRow2 = output2 + outputY * outputWidth;
                        int outputRow3 = output3 + outputY * outputWidth;
                        int outputRow4 = output4 + outputY * outputWidth;
                        int outputRow5 = output5 + outputY * outputWidth;
                        int outputRow6 = output6 + outputY * outputWidth;
                        int outputRow7 = output7 + outputY * outputWidth;
                        for (int kernelY = 0; kernelY < 3; kernelY++) {
                            int sourceY = inputY + kernelY;
                            if (sourceY < 0 || sourceY >= height) continue;
                            int inputRow = inputBase + sourceY * width;
                            int kernelRow0 = weight0 + kernelY * 3;
                            int kernelRow1 = weight1 + kernelY * 3;
                            int kernelRow2 = weight2 + kernelY * 3;
                            int kernelRow3 = weight3 + kernelY * 3;
                            int kernelRow4 = weight4 + kernelY * 3;
                            int kernelRow5 = weight5 + kernelY * 3;
                            int kernelRow6 = weight6 + kernelY * 3;
                            int kernelRow7 = weight7 + kernelY * 3;
                            for (int outputX = 1; outputX + 1 < outputWidth; outputX++) {
                                int outputIndex0 = outputRow0 + outputX;
                                int outputIndex1 = outputRow1 + outputX;
                                int outputIndex2 = outputRow2 + outputX;
                                int outputIndex3 = outputRow3 + outputX;
                                int outputIndex4 = outputRow4 + outputX;
                                int outputIndex5 = outputRow5 + outputX;
                                int outputIndex6 = outputRow6 + outputX;
                                int outputIndex7 = outputRow7 + outputX;
                                int sampleIndex = inputRow + outputX - 1;
                                float sample = input[sampleIndex];
                                output[outputIndex0] += sample * weights[kernelRow0];
                                output[outputIndex1] += sample * weights[kernelRow1];
                                output[outputIndex2] += sample * weights[kernelRow2];
                                output[outputIndex3] += sample * weights[kernelRow3];
                                output[outputIndex4] += sample * weights[kernelRow4];
                                output[outputIndex5] += sample * weights[kernelRow5];
                                output[outputIndex6] += sample * weights[kernelRow6];
                                output[outputIndex7] += sample * weights[kernelRow7];
                                sample = input[sampleIndex + 1];
                                output[outputIndex0] += sample * weights[kernelRow0 + 1];
                                output[outputIndex1] += sample * weights[kernelRow1 + 1];
                                output[outputIndex2] += sample * weights[kernelRow2 + 1];
                                output[outputIndex3] += sample * weights[kernelRow3 + 1];
                                output[outputIndex4] += sample * weights[kernelRow4 + 1];
                                output[outputIndex5] += sample * weights[kernelRow5 + 1];
                                output[outputIndex6] += sample * weights[kernelRow6 + 1];
                                output[outputIndex7] += sample * weights[kernelRow7 + 1];
                                sample = input[sampleIndex + 2];
                                output[outputIndex0] += sample * weights[kernelRow0 + 2];
                                output[outputIndex1] += sample * weights[kernelRow1 + 2];
                                output[outputIndex2] += sample * weights[kernelRow2 + 2];
                                output[outputIndex3] += sample * weights[kernelRow3 + 2];
                                output[outputIndex4] += sample * weights[kernelRow4 + 2];
                                output[outputIndex5] += sample * weights[kernelRow5 + 2];
                                output[outputIndex6] += sample * weights[kernelRow6 + 2];
                                output[outputIndex7] += sample * weights[kernelRow7 + 2];
                            }
                            for (int outputX = 0; outputX < outputWidth; outputX += outputWidth - 1) {
                                int outputIndex0 = outputRow0 + outputX;
                                int outputIndex1 = outputRow1 + outputX;
                                int outputIndex2 = outputRow2 + outputX;
                                int outputIndex3 = outputRow3 + outputX;
                                int outputIndex4 = outputRow4 + outputX;
                                int outputIndex5 = outputRow5 + outputX;
                                int outputIndex6 = outputRow6 + outputX;
                                int outputIndex7 = outputRow7 + outputX;
                                int inputX = outputX - 1;
                                for (int kernelX = 0; kernelX < 3; kernelX++) {
                                    int sourceX = inputX + kernelX;
                                    if (sourceX < 0 || sourceX >= width) continue;
                                    float sample = input[inputRow + sourceX];
                                    output[outputIndex0] += sample * weights[kernelRow0 + kernelX];
                                    output[outputIndex1] += sample * weights[kernelRow1 + kernelX];
                                    output[outputIndex2] += sample * weights[kernelRow2 + kernelX];
                                    output[outputIndex3] += sample * weights[kernelRow3 + kernelX];
                                    output[outputIndex4] += sample * weights[kernelRow4 + kernelX];
                                    output[outputIndex5] += sample * weights[kernelRow5 + kernelX];
                                    output[outputIndex6] += sample * weights[kernelRow6 + kernelX];
                                    output[outputIndex7] += sample * weights[kernelRow7 + kernelX];
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    /** Scalar 3x3 same-padding path; each output keeps the generic accumulation order. */
    private static void convThreeByThreeStrideOne(float[] input, int inputOffset,
                                                  float[] weights, int weightOffset,
                                                  float[] bias, int biasOffset,
                                                  float[] output, int outputOffset, int batch,
                                                  int channels, int height, int width,
                                                  int outputChannels, int outputHeight,
                                                  int outputWidth) {
        int inputPlane = height * width;
        int outputPlane = outputHeight * outputWidth;
        int kernelPlane = channels * 9;
        for (int n = 0; n < batch; n++) {
            for (int outputChannel = 0; outputChannel < outputChannels; outputChannel += 4) {
                int output0 = outputOffset + (n * outputChannels + outputChannel) * outputPlane;
                int output1 = output0 + outputPlane;
                int output2 = output1 + outputPlane;
                int output3 = output2 + outputPlane;
                float initial0 = bias == null ? 0.0f : bias[biasOffset + outputChannel];
                float initial1 = bias == null ? 0.0f : bias[biasOffset + outputChannel + 1];
                float initial2 = bias == null ? 0.0f : bias[biasOffset + outputChannel + 2];
                float initial3 = bias == null ? 0.0f : bias[biasOffset + outputChannel + 3];
                Arrays.fill(output, output0, output0 + outputPlane, initial0);
                Arrays.fill(output, output1, output1 + outputPlane, initial1);
                Arrays.fill(output, output2, output2 + outputPlane, initial2);
                Arrays.fill(output, output3, output3 + outputPlane, initial3);
                for (int inputChannel = 0; inputChannel < channels; inputChannel++) {
                    int inputBase = inputOffset + (n * channels + inputChannel) * inputPlane;
                    int weight0 = weightOffset + outputChannel * kernelPlane + inputChannel * 9;
                    int weight1 = weight0 + kernelPlane;
                    int weight2 = weight1 + kernelPlane;
                    int weight3 = weight2 + kernelPlane;
                    for (int outputY = 0; outputY < outputHeight; outputY++) {
                        int inputY = outputY - 1;
                        int outputRow0 = output0 + outputY * outputWidth;
                        int outputRow1 = output1 + outputY * outputWidth;
                        int outputRow2 = output2 + outputY * outputWidth;
                        int outputRow3 = output3 + outputY * outputWidth;
                        for (int kernelY = 0; kernelY < 3; kernelY++) {
                            int sourceY = inputY + kernelY;
                            if (sourceY < 0 || sourceY >= height) continue;
                            int inputRow = inputBase + sourceY * width;
                            int kernelRow0 = weight0 + kernelY * 3;
                            int kernelRow1 = weight1 + kernelY * 3;
                            int kernelRow2 = weight2 + kernelY * 3;
                            int kernelRow3 = weight3 + kernelY * 3;
                            for (int outputX = 0; outputX < outputWidth; outputX++) {
                                if (outputX > 0 && outputX + 1 < width) {
                                    int sampleIndex = inputRow + outputX - 1;
                                    float sample = input[sampleIndex];
                                    output[outputRow0 + outputX] += sample * weights[kernelRow0];
                                    output[outputRow1 + outputX] += sample * weights[kernelRow1];
                                    output[outputRow2 + outputX] += sample * weights[kernelRow2];
                                    output[outputRow3 + outputX] += sample * weights[kernelRow3];
                                    sample = input[sampleIndex + 1];
                                    output[outputRow0 + outputX] += sample * weights[kernelRow0 + 1];
                                    output[outputRow1 + outputX] += sample * weights[kernelRow1 + 1];
                                    output[outputRow2 + outputX] += sample * weights[kernelRow2 + 1];
                                    output[outputRow3 + outputX] += sample * weights[kernelRow3 + 1];
                                    sample = input[sampleIndex + 2];
                                    output[outputRow0 + outputX] += sample * weights[kernelRow0 + 2];
                                    output[outputRow1 + outputX] += sample * weights[kernelRow1 + 2];
                                    output[outputRow2 + outputX] += sample * weights[kernelRow2 + 2];
                                    output[outputRow3 + outputX] += sample * weights[kernelRow3 + 2];
                                } else {
                                    int inputX = outputX - 1;
                                    for (int kernelX = 0; kernelX < 3; kernelX++) {
                                        int sourceX = inputX + kernelX;
                                        if (sourceX < 0 || sourceX >= width) continue;
                                        float sample = input[inputRow + sourceX];
                                        output[outputRow0 + outputX] += sample * weights[kernelRow0 + kernelX];
                                        output[outputRow1 + outputX] += sample * weights[kernelRow1 + kernelX];
                                        output[outputRow2 + outputX] += sample * weights[kernelRow2 + kernelX];
                                        output[outputRow3 + outputX] += sample * weights[kernelRow3 + kernelX];
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
        if (groups > 0 && inputChannels % groups == 0 && outputChannels % groups == 0
                && kernelHeight == 2 && kernelWidth == 2
                && strideHeight == 2 && strideWidth == 2
                && dilationHeight == 1 && dilationWidth == 1
                && padTop == 0 && padLeft == 0
                && (long) outputHeight == (long) inputHeight * 2L
                && (long) outputWidth == (long) inputWidth * 2L) {
            convTransposeTwoByTwoStrideTwo(input, inputOffset, weights, weightOffset,
                    bias, biasOffset, output, outputOffset, batch, inputChannels,
                    inputHeight, inputWidth, outputChannels, groups, outputHeight,
                    outputWidth, inputChannelsPerGroup, outputChannelsPerGroup);
            return;
        }
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

    /** Exact 2x2 stride-two transposed convolution used by the Tiny DET decoder. */
    private static void convTransposeTwoByTwoStrideTwo(
            float[] input, int inputOffset, float[] weights, int weightOffset,
            float[] bias, int biasOffset, float[] output, int outputOffset,
            int batch, int inputChannels, int inputHeight, int inputWidth,
            int outputChannels, int groups, int outputHeight, int outputWidth,
            int inputChannelsPerGroup, int outputChannelsPerGroup) {
        int outputPlane = outputHeight * outputWidth;
        int outputElements = batch * outputChannels * outputPlane;
        if (bias == null) {
            Arrays.fill(output, outputOffset, outputOffset + outputElements, 0.0f);
        } else {
            for (int n = 0; n < batch; n++) {
                for (int channel = 0; channel < outputChannels; channel++) {
                    int outputBase = outputOffset + (n * outputChannels + channel) * outputPlane;
                    Arrays.fill(output, outputBase, outputBase + outputPlane,
                            bias[biasOffset + channel]);
                }
            }
        }

        for (int n = 0; n < batch; n++) {
            for (int group = 0; group < groups; group++) {
                for (int ic = 0; ic < inputChannelsPerGroup; ic++) {
                    int inputChannel = group * inputChannelsPerGroup + ic;
                    int inputBase = inputOffset
                            + (n * inputChannels + inputChannel) * inputHeight * inputWidth;
                    for (int ih = 0; ih < inputHeight; ih++) {
                        int outputY = ih * 2;
                        for (int iw = 0; iw < inputWidth; iw++) {
                            float sample = input[inputBase + ih * inputWidth + iw];
                            int outputX = iw * 2;
                            int outputIndex = outputOffset
                                    + (n * outputChannels + group * outputChannelsPerGroup)
                                    * outputPlane
                                    + outputY * outputWidth + outputX;
                            for (int oc = 0; oc < outputChannelsPerGroup; oc++) {
                                int weightIndex = weightOffset
                                        + (inputChannel * outputChannelsPerGroup + oc) * 4;
                                int destination = outputIndex + oc * outputPlane;
                                output[destination] += sample * weights[weightIndex];
                                output[destination + 1] += sample * weights[weightIndex + 1];
                                output[destination + outputWidth] += sample * weights[weightIndex + 2];
                                output[destination + outputWidth + 1] += sample * weights[weightIndex + 3];
                            }
                        }
                    }
                }
            }
        }
    }

    private static void binaryRightScalar(BinaryOp operation, float[] left, int leftOffset,
                                          float scalar, float[] output, int outputOffset,
                                          int length) {
        switch (operation) {
            case ADD:
                for (int i = 0; i < length; i++) output[outputOffset + i] = left[leftOffset + i] + scalar;
                return;
            case MUL:
                for (int i = 0; i < length; i++) output[outputOffset + i] = left[leftOffset + i] * scalar;
                return;
            case DIV:
                for (int i = 0; i < length; i++) output[outputOffset + i] = left[leftOffset + i] / scalar;
                return;
            case SUB:
                for (int i = 0; i < length; i++) output[outputOffset + i] = left[leftOffset + i] - scalar;
                return;
            case POW:
                for (int i = 0; i < length; i++) {
                    output[outputOffset + i] = (float) Math.pow(left[leftOffset + i], scalar);
                }
                return;
            default:
                throw new AssertionError("unknown scalar operation");
        }
    }

    private static void binaryLeftScalar(BinaryOp operation, float scalar, float[] right,
                                         int rightOffset, float[] output, int outputOffset,
                                         int length) {
        switch (operation) {
            case ADD:
                for (int i = 0; i < length; i++) output[outputOffset + i] = scalar + right[rightOffset + i];
                return;
            case MUL:
                for (int i = 0; i < length; i++) output[outputOffset + i] = scalar * right[rightOffset + i];
                return;
            case DIV:
                for (int i = 0; i < length; i++) output[outputOffset + i] = scalar / right[rightOffset + i];
                return;
            case SUB:
                for (int i = 0; i < length; i++) output[outputOffset + i] = scalar - right[rightOffset + i];
                return;
            case POW:
                for (int i = 0; i < length; i++) {
                    output[outputOffset + i] = (float) Math.pow(scalar, right[rightOffset + i]);
                }
                return;
            default:
                throw new AssertionError("unknown scalar operation");
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
