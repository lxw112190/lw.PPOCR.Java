package io.github.lxw112190.ppocr.runtime;

import java.util.Arrays;

/** Concrete, positive tensor dimensions used by a prepared execution. */
public final class TensorShape {
    private final int[] dimensions;
    private final long elementCount;

    public TensorShape(int... dimensions) {
        if (dimensions == null || dimensions.length == 0) {
            throw new IllegalArgumentException("tensor shape must have at least one dimension");
        }
        long count = 1;
        for (int dimension : dimensions) {
            if (dimension <= 0 || count > Long.MAX_VALUE / dimension) {
                throw new IllegalArgumentException("tensor shape is invalid or overflows");
            }
            count *= dimension;
        }
        this.dimensions = dimensions.clone();
        this.elementCount = count;
    }

    public static TensorShape of(int... dimensions) {
        return new TensorShape(dimensions);
    }

    public int getRank() { return dimensions.length; }
    public int get(int axis) { return dimensions[axis]; }
    public int[] getDimensions() { return dimensions.clone(); }
    int[] dimensionsUnsafe() { return dimensions; }
    public long getElementCount() { return elementCount; }

    public boolean equalsDimensions(int[] other) {
        return Arrays.equals(dimensions, other);
    }

    @Override
    public boolean equals(Object other) {
        return other instanceof TensorShape && Arrays.equals(dimensions, ((TensorShape) other).dimensions);
    }

    @Override
    public int hashCode() { return Arrays.hashCode(dimensions); }

    @Override
    public String toString() { return Arrays.toString(dimensions); }
}
