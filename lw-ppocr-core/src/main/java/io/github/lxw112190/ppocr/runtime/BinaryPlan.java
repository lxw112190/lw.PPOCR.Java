package io.github.lxw112190.ppocr.runtime;

import java.util.Arrays;

/** Immutable broadcast mapping prepared once for a binary graph node. */
public final class BinaryPlan {
    private final BinaryVariant variant;
    private final int outputLength;
    private final int[] outputShape;
    private final int[] leftStrides;
    private final int[] rightStrides;

    public BinaryPlan(TensorShape left, TensorShape right, TensorShape output) {
        if (left == null || right == null || output == null) {
            throw new IllegalArgumentException("binary shapes are required");
        }
        int rank = output.getRank();
        if (left.getRank() > rank || right.getRank() > rank) {
            throw new IllegalArgumentException("binary output rank is too small");
        }
        this.outputShape = output.getDimensions();
        validateShape(left, right, outputShape);
        this.outputLength = checkedLength(output);
        this.leftStrides = alignedStrides(left, outputShape);
        this.rightStrides = alignedStrides(right, outputShape);
        this.variant = classify(left, right, outputShape);
    }

    public BinaryVariant getVariant() { return variant; }
    public int getOutputLength() { return outputLength; }
    public int getRank() { return outputShape.length; }
    public int getOutputDimension(int axis) { return outputShape[axis]; }
    public int getLeftStride(int axis) { return leftStrides[axis]; }
    public int getRightStride(int axis) { return rightStrides[axis]; }
    public int[] getOutputShape() { return outputShape.clone(); }
    public int[] getLeftStrides() { return leftStrides.clone(); }
    public int[] getRightStrides() { return rightStrides.clone(); }

    private static int[] alignedStrides(TensorShape shape, int[] outputShape) {
        int rank = outputShape.length;
        int[] strides = new int[rank];
        int[] source = rowStrides(shape.getDimensions());
        int leading = rank - shape.getRank();
        for (int axis = 0; axis < rank; axis++) {
            int sourceAxis = axis - leading;
            if (sourceAxis < 0 || shape.get(sourceAxis) == 1) {
                strides[axis] = 0;
            } else {
                strides[axis] = source[sourceAxis];
            }
        }
        return strides;
    }

    private static BinaryVariant classify(TensorShape left, TensorShape right, int[] output) {
        if (Arrays.equals(left.getDimensions(), output) && Arrays.equals(right.getDimensions(), output)) {
            return BinaryVariant.SAME_SHAPE;
        }
        if (right.getRank() == 1 && right.get(0) == 1 && Arrays.equals(left.getDimensions(), output)) {
            return BinaryVariant.RIGHT_SCALAR;
        }
        if (left.getRank() == 1 && left.get(0) == 1 && Arrays.equals(right.getDimensions(), output)) {
            return BinaryVariant.LEFT_SCALAR;
        }
        return BinaryVariant.GENERAL;
    }

    private static void validateShape(TensorShape left, TensorShape right, int[] output) {
        int rank = output.length;
        for (int axis = 0; axis < rank; axis++) {
            int leftAxis = axis - (rank - left.getRank());
            int rightAxis = axis - (rank - right.getRank());
            int leftDimension = leftAxis < 0 ? 1 : left.get(leftAxis);
            int rightDimension = rightAxis < 0 ? 1 : right.get(rightAxis);
            int expected = Math.max(leftDimension, rightDimension);
            if (leftDimension != rightDimension && leftDimension != 1 && rightDimension != 1) {
                throw new IllegalArgumentException("binary shapes are not broadcast compatible");
            }
            if (output[axis] != expected) {
                throw new IllegalArgumentException("binary output shape does not match broadcast shape");
            }
        }
    }

    private static int[] rowStrides(int[] dimensions) {
        int[] strides = new int[dimensions.length];
        int stride = 1;
        for (int axis = dimensions.length - 1; axis >= 0; axis--) {
            strides[axis] = stride;
            stride = Math.multiplyExact(stride, dimensions[axis]);
        }
        return strides;
    }

    private static int checkedLength(TensorShape shape) {
        long length = shape.getElementCount();
        if (length > Integer.MAX_VALUE) throw new IllegalArgumentException("binary tensor is too large");
        return (int) length;
    }
}
