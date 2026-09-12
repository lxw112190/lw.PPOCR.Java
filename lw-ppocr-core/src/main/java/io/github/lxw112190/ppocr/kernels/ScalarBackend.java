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
