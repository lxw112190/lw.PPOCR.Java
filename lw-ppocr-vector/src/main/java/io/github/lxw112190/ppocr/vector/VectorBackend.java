package io.github.lxw112190.ppocr.vector;

import io.github.lxw112190.ppocr.kernels.BinaryOp;
import io.github.lxw112190.ppocr.kernels.KernelBackend;
import io.github.lxw112190.ppocr.kernels.ScalarBackend;
import io.github.lxw112190.ppocr.runtime.BinaryPlan;
import io.github.lxw112190.ppocr.runtime.BinaryVariant;
import java.util.Arrays;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorSpecies;

/** Optional JDK 25 Vector API backend with scalar fallback for unsupported kernels. */
public final class VectorBackend implements KernelBackend {
    private static final VectorSpecies<Float> SPECIES = FloatVector.SPECIES_PREFERRED;
    private final ScalarBackend scalar = new ScalarBackend();

    @Override
    public void add(float[] left, int leftOffset, float[] right, int rightOffset,
                    float[] output, int outputOffset, int length) {
        int bound = SPECIES.loopBound(length);
        int i = 0;
        for (; i < bound; i += SPECIES.length()) {
            FloatVector.fromArray(SPECIES, left, leftOffset + i)
                    .add(FloatVector.fromArray(SPECIES, right, rightOffset + i))
                    .intoArray(output, outputOffset + i);
        }
        for (; i < length; i++) output[outputOffset + i] = left[leftOffset + i] + right[rightOffset + i];
    }

    @Override
    public void mul(float[] left, int leftOffset, float[] right, int rightOffset,
                    float[] output, int outputOffset, int length) {
        int bound = SPECIES.loopBound(length);
        int i = 0;
        for (; i < bound; i += SPECIES.length()) {
            FloatVector.fromArray(SPECIES, left, leftOffset + i)
                    .mul(FloatVector.fromArray(SPECIES, right, rightOffset + i))
                    .intoArray(output, outputOffset + i);
        }
        for (; i < length; i++) output[outputOffset + i] = left[leftOffset + i] * right[rightOffset + i];
    }

    @Override
    public void div(float[] left, int leftOffset, float[] right, int rightOffset,
                    float[] output, int outputOffset, int length) {
        int bound = SPECIES.loopBound(length);
        int i = 0;
        for (; i < bound; i += SPECIES.length()) {
            FloatVector.fromArray(SPECIES, left, leftOffset + i)
                    .div(FloatVector.fromArray(SPECIES, right, rightOffset + i))
                    .intoArray(output, outputOffset + i);
        }
        for (; i < length; i++) output[outputOffset + i] = left[leftOffset + i] / right[rightOffset + i];
    }

    @Override
    public void sub(float[] left, int leftOffset, float[] right, int rightOffset,
                    float[] output, int outputOffset, int length) {
        int bound = SPECIES.loopBound(length);
        int i = 0;
        for (; i < bound; i += SPECIES.length()) {
            FloatVector.fromArray(SPECIES, left, leftOffset + i)
                    .sub(FloatVector.fromArray(SPECIES, right, rightOffset + i))
                    .intoArray(output, outputOffset + i);
        }
        for (; i < length; i++) output[outputOffset + i] = left[leftOffset + i] - right[rightOffset + i];
    }

    @Override
    public void binary(BinaryOp operation, float[] left, int leftOffset, float[] right,
                       int rightOffset, float[] output, int outputOffset, BinaryPlan plan) {
        if (operation == BinaryOp.POW) {
            scalar.binary(operation, left, leftOffset, right, rightOffset, output, outputOffset, plan);
            return;
        }
        if (plan.getVariant() == BinaryVariant.SAME_SHAPE) {
            binaryContiguous(operation, left, leftOffset, 1, right, rightOffset, 1,
                    output, outputOffset, plan.getOutputLength());
            return;
        }
        if (plan.getVariant() == BinaryVariant.RIGHT_SCALAR) {
            binaryContiguous(operation, left, leftOffset, 1, right, rightOffset, 0,
                    output, outputOffset, plan.getOutputLength());
            return;
        }
        if (plan.getVariant() == BinaryVariant.LEFT_SCALAR) {
            binaryContiguous(operation, left, leftOffset, 0, right, rightOffset, 1,
                    output, outputOffset, plan.getOutputLength());
            return;
        }
        binaryBroadcast(operation, left, leftOffset, right, rightOffset, output, outputOffset, plan);
    }

    private static void binaryBroadcast(BinaryOp operation, float[] left, int leftOffset,
                                        float[] right, int rightOffset, float[] output,
                                        int outputOffset, BinaryPlan plan) {
        int lastAxis = plan.getRank() - 1;
        int inner = plan.getOutputDimension(lastAxis);
        int rows = plan.getOutputLength() / inner;
        int leftInnerStride = plan.getLeftStride(lastAxis);
        int rightInnerStride = plan.getRightStride(lastAxis);
        for (int row = 0; row < rows; row++) {
            int remainder = row;
            int leftIndex = 0;
            int rightIndex = 0;
            for (int axis = lastAxis - 1; axis >= 0; axis--) {
                int coordinate = remainder % plan.getOutputDimension(axis);
                remainder /= plan.getOutputDimension(axis);
                leftIndex += coordinate * plan.getLeftStride(axis);
                rightIndex += coordinate * plan.getRightStride(axis);
            }
            binaryContiguous(operation, left, leftOffset + leftIndex, leftInnerStride,
                    right, rightOffset + rightIndex, rightInnerStride,
                    output, outputOffset + row * inner, inner);
        }
    }

    private static void binaryContiguous(BinaryOp operation, float[] left, int leftOffset,
                                         int leftStride, float[] right, int rightOffset,
                                         int rightStride, float[] output, int outputOffset,
                                         int length) {
        int bound = SPECIES.loopBound(length);
        int i = 0;
        if (leftStride == 0 && rightStride == 0) {
            Arrays.fill(output, outputOffset, outputOffset + length,
                    apply(operation, left[leftOffset], right[rightOffset]));
            return;
        } else if (leftStride == 0) {
            float scalarValue = left[leftOffset];
            for (; i < bound; i += SPECIES.length()) {
                applyLeftScalar(operation, scalarValue,
                        FloatVector.fromArray(SPECIES, right, rightOffset + i))
                        .intoArray(output, outputOffset + i);
            }
        } else if (rightStride == 0) {
            float scalarValue = right[rightOffset];
            for (; i < bound; i += SPECIES.length()) {
                applyRightScalar(operation,
                        FloatVector.fromArray(SPECIES, left, leftOffset + i), scalarValue)
                        .intoArray(output, outputOffset + i);
            }
        } else {
            for (; i < bound; i += SPECIES.length()) {
                apply(operation, FloatVector.fromArray(SPECIES, left, leftOffset + i),
                        FloatVector.fromArray(SPECIES, right, rightOffset + i))
                        .intoArray(output, outputOffset + i);
            }
        }
        for (; i < length; i++) {
            float a = left[leftOffset + i * leftStride];
            float b = right[rightOffset + i * rightStride];
            output[outputOffset + i] = apply(operation, a, b);
        }
    }

    private static FloatVector applyRightScalar(BinaryOp operation, FloatVector left, float right) {
        switch (operation) {
            case ADD: return left.add(right);
            case MUL: return left.mul(right);
            case DIV: return left.div(right);
            case SUB: return left.sub(right);
            default: throw new AssertionError("unsupported vector binary operation: " + operation);
        }
    }

    private static FloatVector applyLeftScalar(BinaryOp operation, float left, FloatVector right) {
        switch (operation) {
            case ADD: return right.add(left);
            case MUL: return right.mul(left);
            case DIV: return FloatVector.broadcast(SPECIES, left).div(right);
            case SUB: return right.neg().add(left);
            default: throw new AssertionError("unsupported vector binary operation: " + operation);
        }
    }

    private static FloatVector apply(BinaryOp operation, FloatVector left, FloatVector right) {
        switch (operation) {
            case ADD: return left.add(right);
            case MUL: return left.mul(right);
            case DIV: return left.div(right);
            case SUB: return left.sub(right);
            default: throw new AssertionError("unsupported vector binary operation: " + operation);
        }
    }

    private static float apply(BinaryOp operation, float left, float right) {
        switch (operation) {
            case ADD: return left + right;
            case MUL: return left * right;
            case DIV: return left / right;
            case SUB: return left - right;
            default: throw new AssertionError("unsupported vector binary operation: " + operation);
        }
    }

    @Override
    public void relu(float[] input, int inputOffset, float[] output, int outputOffset, int length) {
        int bound = SPECIES.loopBound(length);
        int i = 0;
        for (; i < bound; i += SPECIES.length()) {
            FloatVector.fromArray(SPECIES, input, inputOffset + i).max(0.0f)
                    .intoArray(output, outputOffset + i);
        }
        for (; i < length; i++) output[outputOffset + i] = Math.max(0.0f, input[inputOffset + i]);
    }

    @Override public void sigmoid(float[] input, int inputOffset, float[] output, int outputOffset, int length) {
        scalar.sigmoid(input, inputOffset, output, outputOffset, length);
    }
    @Override public void erf(float[] input, int inputOffset, float[] output, int outputOffset, int length) {
        scalar.erf(input, inputOffset, output, outputOffset, length);
    }
    @Override public void hardSigmoid(float[] input, int inputOffset, float[] output, int outputOffset,
                                      int length, float alpha, float beta) {
        int bound = SPECIES.loopBound(length);
        int i = 0;
        for (; i < bound; i += SPECIES.length()) {
            FloatVector.fromArray(SPECIES, input, inputOffset + i).mul(alpha).add(beta)
                    .max(0.0f).min(1.0f).intoArray(output, outputOffset + i);
        }
        for (; i < length; i++) {
            output[outputOffset + i] = Math.max(0.0f,
                    Math.min(1.0f, alpha * input[inputOffset + i] + beta));
        }
    }
    @Override public void sqrt(float[] input, int inputOffset, float[] output, int outputOffset, int length) {
        scalar.sqrt(input, inputOffset, output, outputOffset, length);
    }
    @Override public void pow(float[] left, int leftOffset, float[] right, int rightOffset,
                              float[] output, int outputOffset, int length) {
        scalar.pow(left, leftOffset, right, rightOffset, output, outputOffset, length);
    }
    @Override public void reduceMean(float[] input, int inputOffset, float[] output, int outputOffset,
                                     int[] dimensions, int[] axes, boolean keepDimensions) {
        scalar.reduceMean(input, inputOffset, output, outputOffset, dimensions, axes, keepDimensions);
    }
    @Override public void concat(float[][] inputs, int[] inputOffsets, float[] output, int outputOffset,
                                 int[] dimensions, int axis, int[] axisSizes) {
        scalar.concat(inputs, inputOffsets, output, outputOffset, dimensions, axis, axisSizes);
    }
    @Override public void slice(float[] input, int inputOffset, float[] output, int outputOffset,
                                int[] dimensions, int[] starts, int[] axes, int[] steps) {
        scalar.slice(input, inputOffset, output, outputOffset, dimensions, starts, axes, steps);
    }

    @Override
    public void matMul(float[] left, int leftOffset, float[] right, int rightOffset,
                       float[] output, int outputOffset, int rows, int inner, int columns) {
        int bound = SPECIES.loopBound(columns);
        for (int row = 0; row < rows; row++) {
            int outputRow = outputOffset + row * columns;
            Arrays.fill(output, outputRow, outputRow + columns, 0.0f);
            int leftRow = leftOffset + row * inner;
            for (int k = 0; k < inner; k++) {
                float value = left[leftRow + k];
                int rightRow = rightOffset + k * columns;
                int column = 0;
                for (; column < bound; column += SPECIES.length()) {
                    FloatVector result = FloatVector.fromArray(SPECIES, output, outputRow + column)
                            .add(FloatVector.fromArray(SPECIES, right, rightRow + column).mul(value));
                    result.intoArray(output, outputRow + column);
                }
                for (; column < columns; column++) {
                    output[outputRow + column] += value * right[rightRow + column];
                }
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
        if (kernelHeight == 1 && kernelWidth == 1 && strideHeight == 1 && strideWidth == 1 &&
                dilationHeight == 1 && dilationWidth == 1 && padTop == 0 && padLeft == 0 &&
                padBottom == 0 && padRight == 0 && outputHeight == height && outputWidth == width) {
            pointwise(input, inputOffset, weights, weightOffset, bias, biasOffset, output,
                    outputOffset, batch, channels, height * width, outputChannels, groups);
            return;
        }
        scalar.conv(input, inputOffset, weights, weightOffset, bias, biasOffset, output, outputOffset,
                batch, channels, height, width, outputChannels, kernelHeight, kernelWidth,
                strideHeight, strideWidth, dilationHeight, dilationWidth, padTop, padLeft,
                padBottom, padRight, groups, outputHeight, outputWidth);
    }

    private static void pointwise(float[] input, int inputOffset, float[] weights, int weightOffset,
                                  float[] bias, int biasOffset, float[] output, int outputOffset,
                                  int batch, int channels, int plane, int outputChannels, int groups) {
        int inputsPerGroup = channels / groups;
        int outputsPerGroup = outputChannels / groups;
        int bound = SPECIES.loopBound(plane);
        for (int n = 0; n < batch; n++) {
            for (int group = 0; group < groups; group++) {
                int oc = 0;
                for (; oc + 3 < outputsPerGroup; oc += 4) {
                    int channel0 = group * outputsPerGroup + oc;
                    int output0 = outputOffset + (n * outputChannels + channel0) * plane;
                    int output1 = output0 + plane;
                    int output2 = output1 + plane;
                    int output3 = output2 + plane;
                    Arrays.fill(output, output0, output0 + plane, bias == null ? 0.0f : bias[biasOffset + channel0]);
                    Arrays.fill(output, output1, output1 + plane, bias == null ? 0.0f : bias[biasOffset + channel0 + 1]);
                    Arrays.fill(output, output2, output2 + plane, bias == null ? 0.0f : bias[biasOffset + channel0 + 2]);
                    Arrays.fill(output, output3, output3 + plane, bias == null ? 0.0f : bias[biasOffset + channel0 + 3]);
                    int weight0 = weightOffset + channel0 * inputsPerGroup;
                    int weight1 = weight0 + inputsPerGroup;
                    int weight2 = weight1 + inputsPerGroup;
                    int weight3 = weight2 + inputsPerGroup;
                    for (int ic = 0; ic < inputsPerGroup; ic++) {
                        int inputBase = inputOffset + (n * channels + group * inputsPerGroup + ic) * plane;
                        float w0 = weights[weight0 + ic];
                        float w1 = weights[weight1 + ic];
                        float w2 = weights[weight2 + ic];
                        float w3 = weights[weight3 + ic];
                        int i = 0;
                        for (; i < bound; i += SPECIES.length()) {
                            FloatVector sample = FloatVector.fromArray(SPECIES, input, inputBase + i);
                            FloatVector.fromArray(SPECIES, output, output0 + i).add(sample.mul(w0)).intoArray(output, output0 + i);
                            FloatVector.fromArray(SPECIES, output, output1 + i).add(sample.mul(w1)).intoArray(output, output1 + i);
                            FloatVector.fromArray(SPECIES, output, output2 + i).add(sample.mul(w2)).intoArray(output, output2 + i);
                            FloatVector.fromArray(SPECIES, output, output3 + i).add(sample.mul(w3)).intoArray(output, output3 + i);
                        }
                        for (; i < plane; i++) {
                            float sample = input[inputBase + i];
                            output[output0 + i] += sample * w0;
                            output[output1 + i] += sample * w1;
                            output[output2 + i] += sample * w2;
                            output[output3 + i] += sample * w3;
                        }
                    }
                }
                for (; oc < outputsPerGroup; oc++) {
                    int channel = group * outputsPerGroup + oc;
                    int outputBase = outputOffset + (n * outputChannels + channel) * plane;
                    Arrays.fill(output, outputBase, outputBase + plane,
                            bias == null ? 0.0f : bias[biasOffset + channel]);
                    int weightBase = weightOffset + channel * inputsPerGroup;
                    for (int ic = 0; ic < inputsPerGroup; ic++) {
                        int inputBase = inputOffset + (n * channels + group * inputsPerGroup + ic) * plane;
                        float weight = weights[weightBase + ic];
                        int i = 0;
                        for (; i < bound; i += SPECIES.length()) {
                            FloatVector result = FloatVector.fromArray(SPECIES, output, outputBase + i)
                                    .add(FloatVector.fromArray(SPECIES, input, inputBase + i).mul(weight));
                            result.intoArray(output, outputBase + i);
                        }
                        for (; i < plane; i++) output[outputBase + i] += input[inputBase + i] * weight;
                    }
                }
            }
        }
    }

    @Override public void batchNormalization(float[] input, int inputOffset, float[] scale, int scaleOffset,
                                             float[] bias, int biasOffset, float[] mean, int meanOffset,
                                             float[] variance, int varianceOffset, float epsilon,
                                             float[] output, int outputOffset, int[] dimensions) {
        scalar.batchNormalization(input, inputOffset, scale, scaleOffset, bias, biasOffset,
                mean, meanOffset, variance, varianceOffset, epsilon, output, outputOffset, dimensions);
    }
    @Override public void softmax(float[] input, int inputOffset, float[] output, int outputOffset,
                                  int outer, int axisLength, int inner) {
        scalar.softmax(input, inputOffset, output, outputOffset, outer, axisLength, inner);
    }
    @Override public void pool(float[] input, int inputOffset, float[] output, int outputOffset,
                               int batch, int channels, int height, int width, int kernelHeight,
                               int kernelWidth, int strideHeight, int strideWidth, int padTop,
                               int padLeft, int outputHeight, int outputWidth, boolean maximum,
                               boolean countIncludePad) {
        scalar.pool(input, inputOffset, output, outputOffset, batch, channels, height, width,
                kernelHeight, kernelWidth, strideHeight, strideWidth, padTop, padLeft,
                outputHeight, outputWidth, maximum, countIncludePad);
    }
    @Override public void pool(float[] input, int inputOffset, float[] output, int outputOffset,
                               int batch, int channels, int height, int width, int kernelHeight,
                               int kernelWidth, int strideHeight, int strideWidth, int padTop,
                               int padLeft, int padBottom, int padRight, int outputHeight,
                               int outputWidth, boolean maximum, boolean countIncludePad) {
        scalar.pool(input, inputOffset, output, outputOffset, batch, channels, height, width,
                kernelHeight, kernelWidth, strideHeight, strideWidth, padTop, padLeft,
                padBottom, padRight, outputHeight, outputWidth, maximum, countIncludePad);
    }
    @Override public void resizeNearest(float[] input, int inputOffset, float[] output, int outputOffset,
                                        int batch, int channels, int inputHeight, int inputWidth,
                                        int outputHeight, int outputWidth, float scaleHeight, float scaleWidth) {
        scalar.resizeNearest(input, inputOffset, output, outputOffset, batch, channels,
                inputHeight, inputWidth, outputHeight, outputWidth, scaleHeight, scaleWidth);
    }
    @Override public void convTranspose(float[] input, int inputOffset, float[] weights, int weightOffset,
                                        float[] bias, int biasOffset, float[] output, int outputOffset,
                                        int batch, int inputChannels, int inputHeight, int inputWidth,
                                        int outputChannels, int kernelHeight, int kernelWidth,
                                        int strideHeight, int strideWidth, int dilationHeight,
                                        int dilationWidth, int padTop, int padLeft, int groups,
                                        int outputHeight, int outputWidth) {
        scalar.convTranspose(input, inputOffset, weights, weightOffset, bias, biasOffset,
                output, outputOffset, batch, inputChannels, inputHeight, inputWidth, outputChannels,
                kernelHeight, kernelWidth, strideHeight, strideWidth, dilationHeight, dilationWidth,
                padTop, padLeft, groups, outputHeight, outputWidth);
    }
}
