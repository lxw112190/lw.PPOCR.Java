package io.github.lxw112190.ppocr.kernels;

/** Readable reference kernels; optimized backends must preserve their semantics. */
public final class ScalarBackend implements KernelBackend {
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
            for (int column = 0; column < columns; column++) {
                float sum = 0.0f;
                for (int k = 0; k < inner; k++) {
                    sum += left[leftOffset + row * inner + k] * right[rightOffset + k * columns + column];
                }
                output[outputOffset + row * columns + column] = sum;
            }
        }
    }

    @Override
    public void conv(float[] input, int inputOffset, float[] weights, int weightOffset,
                     float[] bias, int biasOffset, float[] output, int outputOffset,
                     int batch, int channels, int height, int width, int outputChannels,
                     int kernelHeight, int kernelWidth, int strideHeight, int strideWidth,
                     int dilationHeight, int dilationWidth, int padTop, int padLeft,
                     int groups, int outputHeight, int outputWidth) {
        int inputChannelsPerGroup = channels / groups;
        int outputChannelsPerGroup = outputChannels / groups;
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
}
