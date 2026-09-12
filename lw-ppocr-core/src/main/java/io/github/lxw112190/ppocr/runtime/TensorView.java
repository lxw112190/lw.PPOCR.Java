package io.github.lxw112190.ppocr.runtime;

/** Lightweight view into a primitive tensor buffer. */
public final class TensorView {
    private final float[] data;
    private final int offset;
    private final int length;
    private final TensorShape shape;

    TensorView(float[] data, int offset, int length, TensorShape shape) {
        this.data = data;
        this.offset = offset;
        this.length = length;
        this.shape = shape;
    }

    public float[] data() { return data; }
    public int offset() { return offset; }
    public int length() { return length; }
    public TensorShape shape() { return shape; }
}
