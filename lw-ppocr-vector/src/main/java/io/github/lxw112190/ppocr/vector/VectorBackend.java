package io.github.lxw112190.ppocr.vector;

import io.github.lxw112190.ppocr.kernels.BinaryOp;
import io.github.lxw112190.ppocr.kernels.FusedGeluBackend;
import io.github.lxw112190.ppocr.kernels.KernelBackend;
import io.github.lxw112190.ppocr.kernels.ScalarBackend;
import io.github.lxw112190.ppocr.runtime.BinaryPlan;
import io.github.lxw112190.ppocr.runtime.BinaryVariant;
import java.util.Arrays;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorMask;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorShuffle;
import jdk.incubator.vector.VectorSpecies;

/** Optional JDK 25 Vector API backend with scalar fallback for unsupported kernels. */
public final class VectorBackend implements KernelBackend, FusedGeluBackend {
    private static final VectorSpecies<Float> SPECIES = FloatVector.SPECIES_PREFERRED;
    private static final int[] STRIDE_TWO_INDEXES = strideIndexes(2);
    private static final VectorShuffle<Float> ZIP_LOW = VectorShuffle.makeZip(SPECIES, 0);
    private static final VectorShuffle<Float> ZIP_HIGH = VectorShuffle.makeZip(SPECIES, 1);
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
    @Override
    public void erf(float[] input, int inputOffset, float[] output, int outputOffset, int length) {
        int bound = SPECIES.loopBound(length);
        int i = 0;
        for (; i < bound; i += SPECIES.length()) {
            FloatVector value = FloatVector.fromArray(SPECIES, input, inputOffset + i);
            erf(value).intoArray(output, outputOffset + i);
        }
        if (i < length) scalar.erf(input, inputOffset + i, output, outputOffset + i, length - i);
    }

    @Override
    public void gelu(float[] input, int inputOffset, float[] output, int outputOffset,
                     int length, float divisor, float addend, float multiplier) {
        int bound = SPECIES.loopBound(length);
        int i = 0;
        for (; i < bound; i += SPECIES.length()) {
            FloatVector value = FloatVector.fromArray(SPECIES, input, inputOffset + i);
            erf(value.div(divisor)).add(addend).mul(value).mul(multiplier)
                    .intoArray(output, outputOffset + i);
        }
        for (; i < length; i++) {
            float value = input[inputOffset + i];
            output[outputOffset + i] = ((erfScalar(value / divisor) + addend) * value)
                    * multiplier;
        }
    }

    private static FloatVector erf(FloatVector value) {
        VectorMask<Float> negative = value.compare(VectorOperators.LT, 0.0f);
        FloatVector magnitude = value.abs();
        FloatVector t = FloatVector.broadcast(SPECIES, 1.0f)
                .div(magnitude.mul(0.3275911f).add(1.0f));
        FloatVector polynomial = t.mul(1.061405429f).sub(1.453152027f)
                .mul(t).add(1.421413741f)
                .mul(t).sub(0.284496736f)
                .mul(t).add(0.254829592f)
                .mul(t);
        FloatVector result = polynomial.mul(magnitude.mul(magnitude).neg()
                .lanewise(VectorOperators.EXP)).neg().add(1.0f);
        return result.blend(result.neg(), negative);
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
    @Override
    public void reduceMean(float[] input, int inputOffset, float[] output, int outputOffset,
                           int[] dimensions, int[] axes, boolean keepDimensions) {
        int firstReduced = contiguousReducedSuffix(dimensions.length, axes);
        if (firstReduced < 0) {
            scalar.reduceMean(input, inputOffset, output, outputOffset, dimensions, axes, keepDimensions);
            return;
        }
        int reducedElements = 1;
        for (int axis = firstReduced; axis < dimensions.length; axis++) {
            reducedElements = Math.multiplyExact(reducedElements, dimensions[axis]);
        }
        int outer = 1;
        for (int axis = 0; axis < firstReduced; axis++) {
            outer = Math.multiplyExact(outer, dimensions[axis]);
        }
        int bound = SPECIES.loopBound(reducedElements);
        for (int index = 0; index < outer; index++) {
            int base = inputOffset + index * reducedElements;
            FloatVector vectorSum = FloatVector.zero(SPECIES);
            int element = 0;
            for (; element < bound; element += SPECIES.length()) {
                vectorSum = vectorSum.add(FloatVector.fromArray(SPECIES, input, base + element));
            }
            float sum = vectorSum.reduceLanes(VectorOperators.ADD);
            for (; element < reducedElements; element++) sum += input[base + element];
            output[outputOffset + index] = sum / reducedElements;
        }
    }

    private static int contiguousReducedSuffix(int rank, int[] axes) {
        if (axes.length == 0) return -1;
        boolean[] reduced = new boolean[rank];
        int first = rank;
        for (int axisValue : axes) {
            int axis = axisValue < 0 ? axisValue + rank : axisValue;
            if (axis < 0 || axis >= rank || reduced[axis]) return -1;
            reduced[axis] = true;
            first = Math.min(first, axis);
        }
        if (rank - first != axes.length) return -1;
        for (int axis = first; axis < rank; axis++) if (!reduced[axis]) return -1;
        return first;
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
        int vectorWidth = SPECIES.length();
        int pairedBound = columns - columns % (vectorWidth * 2);
        int blockWidth = vectorWidth * 4;
        int blockBound = columns - columns % blockWidth;
        int row = 0;
        for (; row + 3 < rows; row += 4) {
            int leftRow0 = leftOffset + row * inner;
            int leftRow1 = leftRow0 + inner;
            int leftRow2 = leftRow1 + inner;
            int leftRow3 = leftRow2 + inner;
            int outputRow0 = outputOffset + row * columns;
            int outputRow1 = outputRow0 + columns;
            int outputRow2 = outputRow1 + columns;
            int outputRow3 = outputRow2 + columns;
            int column = 0;
            for (; column < pairedBound; column += vectorWidth * 2) {
                FloatVector sum00 = FloatVector.zero(SPECIES);
                FloatVector sum01 = FloatVector.zero(SPECIES);
                FloatVector sum10 = FloatVector.zero(SPECIES);
                FloatVector sum11 = FloatVector.zero(SPECIES);
                FloatVector sum20 = FloatVector.zero(SPECIES);
                FloatVector sum21 = FloatVector.zero(SPECIES);
                FloatVector sum30 = FloatVector.zero(SPECIES);
                FloatVector sum31 = FloatVector.zero(SPECIES);
                int rightRow = rightOffset + column;
                for (int k = 0; k < inner; k++) {
                    FloatVector values0 = FloatVector.fromArray(SPECIES, right, rightRow);
                    FloatVector values1 = FloatVector.fromArray(
                            SPECIES, right, rightRow + vectorWidth);
                    float left0 = left[leftRow0 + k];
                    float left1 = left[leftRow1 + k];
                    float left2 = left[leftRow2 + k];
                    float left3 = left[leftRow3 + k];
                    sum00 = sum00.add(values0.mul(left0));
                    sum01 = sum01.add(values1.mul(left0));
                    sum10 = sum10.add(values0.mul(left1));
                    sum11 = sum11.add(values1.mul(left1));
                    sum20 = sum20.add(values0.mul(left2));
                    sum21 = sum21.add(values1.mul(left2));
                    sum30 = sum30.add(values0.mul(left3));
                    sum31 = sum31.add(values1.mul(left3));
                    rightRow += columns;
                }
                sum00.intoArray(output, outputRow0 + column);
                sum01.intoArray(output, outputRow0 + column + vectorWidth);
                sum10.intoArray(output, outputRow1 + column);
                sum11.intoArray(output, outputRow1 + column + vectorWidth);
                sum20.intoArray(output, outputRow2 + column);
                sum21.intoArray(output, outputRow2 + column + vectorWidth);
                sum30.intoArray(output, outputRow3 + column);
                sum31.intoArray(output, outputRow3 + column + vectorWidth);
            }
            for (; column < bound; column += vectorWidth) {
                FloatVector sum0 = FloatVector.zero(SPECIES);
                FloatVector sum1 = FloatVector.zero(SPECIES);
                FloatVector sum2 = FloatVector.zero(SPECIES);
                FloatVector sum3 = FloatVector.zero(SPECIES);
                int rightRow = rightOffset + column;
                for (int k = 0; k < inner; k++) {
                    FloatVector values = FloatVector.fromArray(SPECIES, right, rightRow);
                    sum0 = sum0.add(values.mul(left[leftRow0 + k]));
                    sum1 = sum1.add(values.mul(left[leftRow1 + k]));
                    sum2 = sum2.add(values.mul(left[leftRow2 + k]));
                    sum3 = sum3.add(values.mul(left[leftRow3 + k]));
                    rightRow += columns;
                }
                sum0.intoArray(output, outputRow0 + column);
                sum1.intoArray(output, outputRow1 + column);
                sum2.intoArray(output, outputRow2 + column);
                sum3.intoArray(output, outputRow3 + column);
            }
            for (; column < columns; column++) {
                float sum0 = 0.0f;
                float sum1 = 0.0f;
                float sum2 = 0.0f;
                float sum3 = 0.0f;
                int rightIndex = rightOffset + column;
                for (int k = 0; k < inner; k++) {
                    float value = right[rightIndex];
                    sum0 += left[leftRow0 + k] * value;
                    sum1 += left[leftRow1 + k] * value;
                    sum2 += left[leftRow2 + k] * value;
                    sum3 += left[leftRow3 + k] * value;
                    rightIndex += columns;
                }
                output[outputRow0 + column] = sum0;
                output[outputRow1 + column] = sum1;
                output[outputRow2 + column] = sum2;
                output[outputRow3 + column] = sum3;
            }
        }
        for (; row < rows; row++) {
            int outputRow = outputOffset + row * columns;
            int leftRow = leftOffset + row * inner;
            int column = 0;
            for (; column < blockBound; column += blockWidth) {
                FloatVector sum0 = FloatVector.zero(SPECIES);
                FloatVector sum1 = FloatVector.zero(SPECIES);
                FloatVector sum2 = FloatVector.zero(SPECIES);
                FloatVector sum3 = FloatVector.zero(SPECIES);
                int rightRow = rightOffset + column;
                for (int k = 0; k < inner; k++) {
                    float value = left[leftRow + k];
                    sum0 = sum0.add(FloatVector.fromArray(SPECIES, right, rightRow).mul(value));
                    sum1 = sum1.add(FloatVector.fromArray(
                            SPECIES, right, rightRow + vectorWidth).mul(value));
                    sum2 = sum2.add(FloatVector.fromArray(
                            SPECIES, right, rightRow + vectorWidth * 2).mul(value));
                    sum3 = sum3.add(FloatVector.fromArray(
                            SPECIES, right, rightRow + vectorWidth * 3).mul(value));
                    rightRow += columns;
                }
                sum0.intoArray(output, outputRow + column);
                sum1.intoArray(output, outputRow + column + vectorWidth);
                sum2.intoArray(output, outputRow + column + vectorWidth * 2);
                sum3.intoArray(output, outputRow + column + vectorWidth * 3);
            }
            for (; column < bound; column += SPECIES.length()) {
                FloatVector sum = FloatVector.zero(SPECIES);
                int rightRow = rightOffset + column;
                for (int k = 0; k < inner; k++) {
                    sum = sum.add(FloatVector.fromArray(SPECIES, right, rightRow)
                            .mul(left[leftRow + k]));
                    rightRow += columns;
                }
                sum.intoArray(output, outputRow + column);
            }
            for (; column < columns; column++) {
                float sum = 0.0f;
                int rightIndex = rightOffset + column;
                for (int k = 0; k < inner; k++) {
                    sum += left[leftRow + k] * right[rightIndex];
                    rightIndex += columns;
                }
                output[outputRow + column] = sum;
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
        if (groups == channels && outputChannels == channels && strideWidth == 1) {
            depthwise(input, inputOffset, weights, weightOffset, bias, biasOffset, output,
                    outputOffset, batch, channels, height, width, kernelHeight, kernelWidth,
                    strideHeight, dilationHeight, dilationWidth, padTop, padLeft,
                    outputHeight, outputWidth);
            return;
        }
        if (strideWidth == 1) {
            generalStrideOne(input, inputOffset, weights, weightOffset, bias, biasOffset,
                    output, outputOffset, batch, channels, height, width, outputChannels,
                    kernelHeight, kernelWidth, strideHeight, dilationHeight, dilationWidth,
                    padTop, padLeft, groups, outputHeight, outputWidth);
            return;
        }
        if (strideWidth == 2) {
            generalStrideTwo(input, inputOffset, weights, weightOffset, bias, biasOffset,
                    output, outputOffset, batch, channels, height, width, outputChannels,
                    kernelHeight, kernelWidth, strideHeight, dilationHeight, dilationWidth,
                    padTop, padLeft, groups, outputHeight, outputWidth);
            return;
        }
        scalar.conv(input, inputOffset, weights, weightOffset, bias, biasOffset, output, outputOffset,
                batch, channels, height, width, outputChannels, kernelHeight, kernelWidth,
                strideHeight, strideWidth, dilationHeight, dilationWidth, padTop, padLeft,
                padBottom, padRight, groups, outputHeight, outputWidth);
    }

    private static void generalStrideTwo(float[] input, int inputOffset, float[] weights,
                                         int weightOffset, float[] bias, int biasOffset,
                                         float[] output, int outputOffset, int batch, int channels,
                                         int height, int width, int outputChannels, int kernelHeight,
                                         int kernelWidth, int strideHeight, int dilationHeight,
                                         int dilationWidth, int padTop, int padLeft, int groups,
                                         int outputHeight, int outputWidth) {
        int inputChannelsPerGroup = channels / groups;
        int outputChannelsPerGroup = outputChannels / groups;
        int inputPlane = height * width;
        int outputPlane = outputHeight * outputWidth;
        int kernelPlane = kernelHeight * kernelWidth;
        for (int n = 0; n < batch; n++) {
            for (int group = 0; group < groups; group++) {
                int oc = 0;
                for (; oc + 7 < outputChannelsPerGroup; oc += 8) {
                    int outputChannel = group * outputChannelsPerGroup + oc;
                    int outputBase0 = outputOffset +
                            (n * outputChannels + outputChannel) * outputPlane;
                    int outputBase1 = outputBase0 + outputPlane;
                    int outputBase2 = outputBase1 + outputPlane;
                    int outputBase3 = outputBase2 + outputPlane;
                    int outputBase4 = outputBase3 + outputPlane;
                    int outputBase5 = outputBase4 + outputPlane;
                    int outputBase6 = outputBase5 + outputPlane;
                    int outputBase7 = outputBase6 + outputPlane;
                    Arrays.fill(output, outputBase0, outputBase0 + outputPlane,
                            bias == null ? 0.0f : bias[biasOffset + outputChannel]);
                    Arrays.fill(output, outputBase1, outputBase1 + outputPlane,
                            bias == null ? 0.0f : bias[biasOffset + outputChannel + 1]);
                    Arrays.fill(output, outputBase2, outputBase2 + outputPlane,
                            bias == null ? 0.0f : bias[biasOffset + outputChannel + 2]);
                    Arrays.fill(output, outputBase3, outputBase3 + outputPlane,
                            bias == null ? 0.0f : bias[biasOffset + outputChannel + 3]);
                    Arrays.fill(output, outputBase4, outputBase4 + outputPlane,
                            bias == null ? 0.0f : bias[biasOffset + outputChannel + 4]);
                    Arrays.fill(output, outputBase5, outputBase5 + outputPlane,
                            bias == null ? 0.0f : bias[biasOffset + outputChannel + 5]);
                    Arrays.fill(output, outputBase6, outputBase6 + outputPlane,
                            bias == null ? 0.0f : bias[biasOffset + outputChannel + 6]);
                    Arrays.fill(output, outputBase7, outputBase7 + outputPlane,
                            bias == null ? 0.0f : bias[biasOffset + outputChannel + 7]);
                    int channelWeightStride = inputChannelsPerGroup * kernelPlane;
                    for (int ic = 0; ic < inputChannelsPerGroup; ic++) {
                        int inputChannel = group * inputChannelsPerGroup + ic;
                        int inputBase = inputOffset +
                                (n * channels + inputChannel) * inputPlane;
                        int weightBase0 = weightOffset +
                                (outputChannel * inputChannelsPerGroup + ic) * kernelPlane;
                        int weightBase1 = weightBase0 + channelWeightStride;
                        int weightBase2 = weightBase1 + channelWeightStride;
                        int weightBase3 = weightBase2 + channelWeightStride;
                        int weightBase4 = weightBase3 + channelWeightStride;
                        int weightBase5 = weightBase4 + channelWeightStride;
                        int weightBase6 = weightBase5 + channelWeightStride;
                        int weightBase7 = weightBase6 + channelWeightStride;
                        for (int kh = 0; kh < kernelHeight; kh++) {
                            for (int kw = 0; kw < kernelWidth; kw++) {
                                int kernelIndex = kh * kernelWidth + kw;
                                float weight0 = weights[weightBase0 + kernelIndex];
                                float weight1 = weights[weightBase1 + kernelIndex];
                                float weight2 = weights[weightBase2 + kernelIndex];
                                float weight3 = weights[weightBase3 + kernelIndex];
                                float weight4 = weights[weightBase4 + kernelIndex];
                                float weight5 = weights[weightBase5 + kernelIndex];
                                float weight6 = weights[weightBase6 + kernelIndex];
                                float weight7 = weights[weightBase7 + kernelIndex];
                                int shift = padLeft - kw * dilationWidth;
                                int start = Math.max(0, ceilDiv(shift, 2));
                                int end = Math.min(outputWidth, ceilDiv(width + shift, 2));
                                int vectorEnd = start +
                                        SPECIES.loopBound(Math.max(0, end - start));
                                for (int oh = 0; oh < outputHeight; oh++) {
                                    int ih = oh * strideHeight - padTop + kh * dilationHeight;
                                    if (ih < 0 || ih >= height) continue;
                                    int source = inputBase + ih * width + start * 2 - shift;
                                    int destination0 = outputBase0 + oh * outputWidth;
                                    int destination1 = outputBase1 + oh * outputWidth;
                                    int destination2 = outputBase2 + oh * outputWidth;
                                    int destination3 = outputBase3 + oh * outputWidth;
                                    int destination4 = outputBase4 + oh * outputWidth;
                                    int destination5 = outputBase5 + oh * outputWidth;
                                    int destination6 = outputBase6 + oh * outputWidth;
                                    int destination7 = outputBase7 + oh * outputWidth;
                                    int ow = start;
                                    for (; ow < vectorEnd; ow += SPECIES.length()) {
                                        FloatVector sample = FloatVector.fromArray(
                                                SPECIES, input, source, STRIDE_TWO_INDEXES, 0);
                                        FloatVector.fromArray(SPECIES, output, destination0 + ow)
                                                .add(sample.mul(weight0))
                                                .intoArray(output, destination0 + ow);
                                        FloatVector.fromArray(SPECIES, output, destination1 + ow)
                                                .add(sample.mul(weight1))
                                                .intoArray(output, destination1 + ow);
                                        FloatVector.fromArray(SPECIES, output, destination2 + ow)
                                                .add(sample.mul(weight2))
                                                .intoArray(output, destination2 + ow);
                                        FloatVector.fromArray(SPECIES, output, destination3 + ow)
                                                .add(sample.mul(weight3))
                                                .intoArray(output, destination3 + ow);
                                        FloatVector.fromArray(SPECIES, output, destination4 + ow)
                                                .add(sample.mul(weight4))
                                                .intoArray(output, destination4 + ow);
                                        FloatVector.fromArray(SPECIES, output, destination5 + ow)
                                                .add(sample.mul(weight5))
                                                .intoArray(output, destination5 + ow);
                                        FloatVector.fromArray(SPECIES, output, destination6 + ow)
                                                .add(sample.mul(weight6))
                                                .intoArray(output, destination6 + ow);
                                        FloatVector.fromArray(SPECIES, output, destination7 + ow)
                                                .add(sample.mul(weight7))
                                                .intoArray(output, destination7 + ow);
                                        source += SPECIES.length() * 2;
                                    }
                                    for (; ow < end; ow++) {
                                        float sample = input[source];
                                        output[destination0 + ow] += sample * weight0;
                                        output[destination1 + ow] += sample * weight1;
                                        output[destination2 + ow] += sample * weight2;
                                        output[destination3 + ow] += sample * weight3;
                                        output[destination4 + ow] += sample * weight4;
                                        output[destination5 + ow] += sample * weight5;
                                        output[destination6 + ow] += sample * weight6;
                                        output[destination7 + ow] += sample * weight7;
                                        source += 2;
                                    }
                                }
                            }
                        }
                    }
                }
                for (; oc + 3 < outputChannelsPerGroup; oc += 4) {
                    int outputChannel = group * outputChannelsPerGroup + oc;
                    int outputBase0 = outputOffset +
                            (n * outputChannels + outputChannel) * outputPlane;
                    int outputBase1 = outputBase0 + outputPlane;
                    int outputBase2 = outputBase1 + outputPlane;
                    int outputBase3 = outputBase2 + outputPlane;
                    Arrays.fill(output, outputBase0, outputBase0 + outputPlane,
                            bias == null ? 0.0f : bias[biasOffset + outputChannel]);
                    Arrays.fill(output, outputBase1, outputBase1 + outputPlane,
                            bias == null ? 0.0f : bias[biasOffset + outputChannel + 1]);
                    Arrays.fill(output, outputBase2, outputBase2 + outputPlane,
                            bias == null ? 0.0f : bias[biasOffset + outputChannel + 2]);
                    Arrays.fill(output, outputBase3, outputBase3 + outputPlane,
                            bias == null ? 0.0f : bias[biasOffset + outputChannel + 3]);
                    int channelWeightStride = inputChannelsPerGroup * kernelPlane;
                    for (int ic = 0; ic < inputChannelsPerGroup; ic++) {
                        int inputChannel = group * inputChannelsPerGroup + ic;
                        int inputBase = inputOffset +
                                (n * channels + inputChannel) * inputPlane;
                        int weightBase0 = weightOffset +
                                (outputChannel * inputChannelsPerGroup + ic) * kernelPlane;
                        int weightBase1 = weightBase0 + channelWeightStride;
                        int weightBase2 = weightBase1 + channelWeightStride;
                        int weightBase3 = weightBase2 + channelWeightStride;
                        for (int kh = 0; kh < kernelHeight; kh++) {
                            for (int kw = 0; kw < kernelWidth; kw++) {
                                int kernelIndex = kh * kernelWidth + kw;
                                float weight0 = weights[weightBase0 + kernelIndex];
                                float weight1 = weights[weightBase1 + kernelIndex];
                                float weight2 = weights[weightBase2 + kernelIndex];
                                float weight3 = weights[weightBase3 + kernelIndex];
                                int shift = padLeft - kw * dilationWidth;
                                int start = Math.max(0, ceilDiv(shift, 2));
                                int end = Math.min(outputWidth, ceilDiv(width + shift, 2));
                                int vectorEnd = start +
                                        SPECIES.loopBound(Math.max(0, end - start));
                                for (int oh = 0; oh < outputHeight; oh++) {
                                    int ih = oh * strideHeight - padTop + kh * dilationHeight;
                                    if (ih < 0 || ih >= height) continue;
                                    int source = inputBase + ih * width + start * 2 - shift;
                                    int destination0 = outputBase0 + oh * outputWidth;
                                    int destination1 = outputBase1 + oh * outputWidth;
                                    int destination2 = outputBase2 + oh * outputWidth;
                                    int destination3 = outputBase3 + oh * outputWidth;
                                    int ow = start;
                                    for (; ow < vectorEnd; ow += SPECIES.length()) {
                                        FloatVector sample = FloatVector.fromArray(
                                                SPECIES, input, source, STRIDE_TWO_INDEXES, 0);
                                        FloatVector.fromArray(SPECIES, output, destination0 + ow)
                                                .add(sample.mul(weight0))
                                                .intoArray(output, destination0 + ow);
                                        FloatVector.fromArray(SPECIES, output, destination1 + ow)
                                                .add(sample.mul(weight1))
                                                .intoArray(output, destination1 + ow);
                                        FloatVector.fromArray(SPECIES, output, destination2 + ow)
                                                .add(sample.mul(weight2))
                                                .intoArray(output, destination2 + ow);
                                        FloatVector.fromArray(SPECIES, output, destination3 + ow)
                                                .add(sample.mul(weight3))
                                                .intoArray(output, destination3 + ow);
                                        source += SPECIES.length() * 2;
                                    }
                                    for (; ow < end; ow++) {
                                        float sample = input[source];
                                        output[destination0 + ow] += sample * weight0;
                                        output[destination1 + ow] += sample * weight1;
                                        output[destination2 + ow] += sample * weight2;
                                        output[destination3 + ow] += sample * weight3;
                                        source += 2;
                                    }
                                }
                            }
                        }
                    }
                }
                for (; oc < outputChannelsPerGroup; oc++) {
                    int outputChannel = group * outputChannelsPerGroup + oc;
                    int outputBase = outputOffset + (n * outputChannels + outputChannel) * outputPlane;
                    Arrays.fill(output, outputBase, outputBase + outputPlane,
                            bias == null ? 0.0f : bias[biasOffset + outputChannel]);
                    for (int ic = 0; ic < inputChannelsPerGroup; ic++) {
                        int inputChannel = group * inputChannelsPerGroup + ic;
                        int inputBase = inputOffset + (n * channels + inputChannel) * inputPlane;
                        int weightBase = weightOffset +
                                (outputChannel * inputChannelsPerGroup + ic) * kernelPlane;
                        for (int kh = 0; kh < kernelHeight; kh++) {
                            for (int kw = 0; kw < kernelWidth; kw++) {
                                float weight = weights[weightBase + kh * kernelWidth + kw];
                                int shift = padLeft - kw * dilationWidth;
                                int start = Math.max(0, ceilDiv(shift, 2));
                                int end = Math.min(outputWidth, ceilDiv(width + shift, 2));
                                int vectorEnd = start + SPECIES.loopBound(Math.max(0, end - start));
                                for (int oh = 0; oh < outputHeight; oh++) {
                                    int ih = oh * strideHeight - padTop + kh * dilationHeight;
                                    if (ih < 0 || ih >= height) continue;
                                    int source = inputBase + ih * width + start * 2 - shift;
                                    int destination = outputBase + oh * outputWidth;
                                    int ow = start;
                                    for (; ow < vectorEnd; ow += SPECIES.length()) {
                                        FloatVector result = FloatVector.fromArray(
                                                SPECIES, output, destination + ow)
                                                .add(FloatVector.fromArray(SPECIES, input, source,
                                                        STRIDE_TWO_INDEXES, 0).mul(weight));
                                        result.intoArray(output, destination + ow);
                                        source += SPECIES.length() * 2;
                                    }
                                    for (; ow < end; ow++) {
                                        output[destination + ow] += input[source] * weight;
                                        source += 2;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    private static int ceilDiv(int value, int divisor) {
        return -Math.floorDiv(-value, divisor);
    }

    private static int[] strideIndexes(int stride) {
        int[] indexes = new int[SPECIES.length()];
        for (int lane = 0; lane < indexes.length; lane++) indexes[lane] = lane * stride;
        return indexes;
    }

    private static void generalStrideOne(float[] input, int inputOffset, float[] weights,
                                         int weightOffset, float[] bias, int biasOffset,
                                         float[] output, int outputOffset, int batch, int channels,
                                         int height, int width, int outputChannels, int kernelHeight,
                                         int kernelWidth, int strideHeight, int dilationHeight,
                                         int dilationWidth, int padTop, int padLeft, int groups,
                                         int outputHeight, int outputWidth) {
        int inputChannelsPerGroup = channels / groups;
        int outputChannelsPerGroup = outputChannels / groups;
        int inputPlane = height * width;
        int outputPlane = outputHeight * outputWidth;
        for (int n = 0; n < batch; n++) {
            for (int group = 0; group < groups; group++) {
                for (int oc = 0; oc < outputChannelsPerGroup; oc++) {
                    int outputChannel = group * outputChannelsPerGroup + oc;
                    int outputBase = outputOffset + (n * outputChannels + outputChannel) * outputPlane;
                    Arrays.fill(output, outputBase, outputBase + outputPlane,
                            bias == null ? 0.0f : bias[biasOffset + outputChannel]);
                    for (int ic = 0; ic < inputChannelsPerGroup; ic++) {
                        int inputChannel = group * inputChannelsPerGroup + ic;
                        int inputBase = inputOffset + (n * channels + inputChannel) * inputPlane;
                        int weightBase = weightOffset +
                                (outputChannel * inputChannelsPerGroup + ic) * kernelHeight * kernelWidth;
                        for (int kh = 0; kh < kernelHeight; kh++) {
                            for (int kw = 0; kw < kernelWidth; kw++) {
                                float weight = weights[weightBase + kh * kernelWidth + kw];
                                int shift = padLeft - kw * dilationWidth;
                                int start = Math.max(0, shift);
                                int end = Math.min(outputWidth, width + shift);
                                int vectorEnd = start + SPECIES.loopBound(end - start);
                                for (int oh = 0; oh < outputHeight; oh++) {
                                    int ih = oh * strideHeight - padTop + kh * dilationHeight;
                                    if (ih < 0 || ih >= height) continue;
                                    int source = inputBase + ih * width + start - shift;
                                    int destination = outputBase + oh * outputWidth;
                                    int ow = start;
                                    for (; ow < vectorEnd; ow += SPECIES.length()) {
                                        FloatVector result = FloatVector.fromArray(
                                                SPECIES, output, destination + ow)
                                                .add(FloatVector.fromArray(SPECIES, input, source)
                                                        .mul(weight));
                                        result.intoArray(output, destination + ow);
                                        source += SPECIES.length();
                                    }
                                    for (; ow < end; ow++) {
                                        output[destination + ow] += input[source++] * weight;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    private static void pointwise(float[] input, int inputOffset, float[] weights, int weightOffset,
                                  float[] bias, int biasOffset, float[] output, int outputOffset,
                                  int batch, int channels, int plane, int outputChannels, int groups) {
        int inputsPerGroup = channels / groups;
        int outputsPerGroup = outputChannels / groups;
        int bound = SPECIES.loopBound(plane);
        for (int n = 0; n < batch; n++) {
            for (int group = 0; group < groups; group++) {
                int inputGroupBase = inputOffset +
                        (n * channels + group * inputsPerGroup) * plane;
                int oc = 0;
                for (; oc + 11 < outputsPerGroup; oc += 12) {
                    pointwiseTwelve(input, inputGroupBase, weights, weightOffset,
                            bias, biasOffset, output, outputOffset, n, outputChannels,
                            plane, inputsPerGroup, group * outputsPerGroup + oc, bound);
                }
                for (; oc + 7 < outputsPerGroup; oc += 8) {
                    int channel0 = group * outputsPerGroup + oc;
                    int output0 = outputOffset + (n * outputChannels + channel0) * plane;
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
                for (; oc + 3 < outputsPerGroup; oc += 4) {
                    int channel0 = group * outputsPerGroup + oc;
                    int output0 = outputOffset + (n * outputChannels + channel0) * plane;
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
                for (; oc < outputsPerGroup; oc++) {
                    int channel = group * outputsPerGroup + oc;
                    int outputBase = outputOffset + (n * outputChannels + channel) * plane;
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
            }
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

    private static void depthwise(float[] input, int inputOffset, float[] weights,
                                  int weightOffset, float[] bias, int biasOffset,
                                  float[] output, int outputOffset, int batch, int channels,
                                  int height, int width, int kernelHeight, int kernelWidth,
                                  int strideHeight, int dilationHeight, int dilationWidth,
                                  int padTop, int padLeft, int outputHeight, int outputWidth) {
        int inputPlane = height * width;
        int outputPlane = outputHeight * outputWidth;
        int kernelPlane = kernelHeight * kernelWidth;
        for (int n = 0; n < batch; n++) {
            for (int channel = 0; channel < channels; channel++) {
                int inputBase = inputOffset + (n * channels + channel) * inputPlane;
                int outputBase = outputOffset + (n * channels + channel) * outputPlane;
                Arrays.fill(output, outputBase, outputBase + outputPlane,
                        bias == null ? 0.0f : bias[biasOffset + channel]);
                int kernelBase = weightOffset + channel * kernelPlane;
                for (int kh = 0; kh < kernelHeight; kh++) {
                    for (int kw = 0; kw < kernelWidth; kw++) {
                        float weight = weights[kernelBase + kh * kernelWidth + kw];
                        int shift = padLeft - kw * dilationWidth;
                        int start = Math.max(0, shift);
                        int end = Math.min(outputWidth, width + shift);
                        int vectorEnd = start + SPECIES.loopBound(end - start);
                        for (int oh = 0; oh < outputHeight; oh++) {
                            int ih = oh * strideHeight - padTop + kh * dilationHeight;
                            if (ih < 0 || ih >= height) continue;
                            int inputRow = inputBase + ih * width;
                            int outputRow = outputBase + oh * outputWidth;
                            int source = inputRow + start - shift;
                            int ow = start;
                            for (; ow < vectorEnd; ow += SPECIES.length()) {
                                FloatVector result = FloatVector.fromArray(SPECIES, output, outputRow + ow)
                                        .add(FloatVector.fromArray(SPECIES, input, source).mul(weight));
                                result.intoArray(output, outputRow + ow);
                                source += SPECIES.length();
                            }
                            for (; ow < end; ow++) output[outputRow + ow] += input[source++] * weight;
                        }
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
        if (inner != 1) {
            scalar.softmax(input, inputOffset, output, outputOffset, outer, axisLength, inner);
            return;
        }
        int bound = SPECIES.loopBound(axisLength);
        for (int outerIndex = 0; outerIndex < outer; outerIndex++) {
            int inputBase = inputOffset + outerIndex * axisLength;
            int outputBase = outputOffset + outerIndex * axisLength;
            FloatVector vectorMaximum = FloatVector.broadcast(SPECIES, -Float.MAX_VALUE);
            int axis = 0;
            for (; axis < bound; axis += SPECIES.length()) {
                vectorMaximum = vectorMaximum.max(
                        FloatVector.fromArray(SPECIES, input, inputBase + axis));
            }
            float maximum = vectorMaximum.reduceLanes(VectorOperators.MAX);
            for (; axis < axisLength; axis++) maximum = Math.max(maximum, input[inputBase + axis]);

            FloatVector vectorSum = FloatVector.zero(SPECIES);
            axis = 0;
            for (; axis < bound; axis += SPECIES.length()) {
                FloatVector values = FloatVector.fromArray(SPECIES, input, inputBase + axis)
                        .sub(maximum).lanewise(VectorOperators.EXP);
                values.intoArray(output, outputBase + axis);
                vectorSum = vectorSum.add(values);
            }
            float sum = vectorSum.reduceLanes(VectorOperators.ADD);
            for (; axis < axisLength; axis++) {
                float value = (float) Math.exp(input[inputBase + axis] - maximum);
                output[outputBase + axis] = value;
                sum += value;
            }

            axis = 0;
            for (; axis < bound; axis += SPECIES.length()) {
                FloatVector.fromArray(SPECIES, output, outputBase + axis).div(sum)
                        .intoArray(output, outputBase + axis);
            }
            for (; axis < axisLength; axis++) output[outputBase + axis] /= sum;
        }
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
        if (groups == 1 && kernelHeight == 2 && kernelWidth == 2 &&
                strideHeight == 2 && strideWidth == 2 &&
                dilationHeight == 1 && dilationWidth == 1 &&
                padTop == 0 && padLeft == 0 &&
                outputHeight == inputHeight * 2 && outputWidth == inputWidth * 2) {
            convTransposeTwoByTwo(input, inputOffset, weights, weightOffset, bias, biasOffset,
                    output, outputOffset, batch, inputChannels, inputHeight, inputWidth,
                    outputChannels, outputHeight, outputWidth);
            return;
        }
        scalar.convTranspose(input, inputOffset, weights, weightOffset, bias, biasOffset,
                output, outputOffset, batch, inputChannels, inputHeight, inputWidth, outputChannels,
                kernelHeight, kernelWidth, strideHeight, strideWidth, dilationHeight, dilationWidth,
                padTop, padLeft, groups, outputHeight, outputWidth);
    }

    private static void convTransposeTwoByTwo(float[] input, int inputOffset, float[] weights,
                                               int weightOffset, float[] bias, int biasOffset,
                                               float[] output, int outputOffset, int batch,
                                               int inputChannels, int inputHeight, int inputWidth,
                                               int outputChannels, int outputHeight, int outputWidth) {
        int inputPlane = inputHeight * inputWidth;
        int outputPlane = outputHeight * outputWidth;
        int bound = SPECIES.loopBound(inputWidth);
        for (int n = 0; n < batch; n++) {
            for (int oc = 0; oc < outputChannels; oc++) {
                float initial = bias == null ? 0.0f : bias[biasOffset + oc];
                FloatVector initialVector = FloatVector.broadcast(SPECIES, initial);
                int outputBase = outputOffset + (n * outputChannels + oc) * outputPlane;
                for (int ih = 0; ih < inputHeight; ih++) {
                    int outputRow0 = outputBase + ih * 2 * outputWidth;
                    int outputRow1 = outputRow0 + outputWidth;
                    int iw = 0;
                    for (; iw < bound; iw += SPECIES.length()) {
                        FloatVector sum00 = initialVector;
                        FloatVector sum01 = initialVector;
                        FloatVector sum10 = initialVector;
                        FloatVector sum11 = initialVector;
                        for (int ic = 0; ic < inputChannels; ic++) {
                            int inputBase = inputOffset +
                                    (n * inputChannels + ic) * inputPlane + ih * inputWidth + iw;
                            int kernel = weightOffset + (ic * outputChannels + oc) * 4;
                            FloatVector sample = FloatVector.fromArray(SPECIES, input, inputBase);
                            sum00 = sum00.add(sample.mul(weights[kernel]));
                            sum01 = sum01.add(sample.mul(weights[kernel + 1]));
                            sum10 = sum10.add(sample.mul(weights[kernel + 2]));
                            sum11 = sum11.add(sample.mul(weights[kernel + 3]));
                        }
                        int outputColumn = iw * 2;
                        sum00.rearrange(ZIP_LOW, sum01)
                                .intoArray(output, outputRow0 + outputColumn);
                        sum00.rearrange(ZIP_HIGH, sum01)
                                .intoArray(output, outputRow0 + outputColumn + SPECIES.length());
                        sum10.rearrange(ZIP_LOW, sum11)
                                .intoArray(output, outputRow1 + outputColumn);
                        sum10.rearrange(ZIP_HIGH, sum11)
                                .intoArray(output, outputRow1 + outputColumn + SPECIES.length());
                    }
                    for (; iw < inputWidth; iw++) {
                        float sum00 = initial;
                        float sum01 = initial;
                        float sum10 = initial;
                        float sum11 = initial;
                        for (int ic = 0; ic < inputChannels; ic++) {
                            float sample = input[inputOffset +
                                    (n * inputChannels + ic) * inputPlane + ih * inputWidth + iw];
                            int kernel = weightOffset + (ic * outputChannels + oc) * 4;
                            sum00 += sample * weights[kernel];
                            sum01 += sample * weights[kernel + 1];
                            sum10 += sample * weights[kernel + 2];
                            sum11 += sample * weights[kernel + 3];
                        }
                        int outputColumn = iw * 2;
                        output[outputRow0 + outputColumn] = sum00;
                        output[outputRow0 + outputColumn + 1] = sum01;
                        output[outputRow1 + outputColumn] = sum10;
                        output[outputRow1 + outputColumn + 1] = sum11;
                    }
                }
            }
        }
    }
}
