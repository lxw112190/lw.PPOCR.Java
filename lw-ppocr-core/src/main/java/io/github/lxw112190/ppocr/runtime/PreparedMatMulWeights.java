package io.github.lxw112190.ppocr.runtime;

import io.github.lxw112190.ppocr.model.OcrErrorCode;
import io.github.lxw112190.ppocr.model.OcrException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;

/** Immutable description of canonical row-major MatMul weights. */
public final class PreparedMatMulWeights {
    private final float[] canonical;
    private final int offset;
    private final int inner;
    private final int columns;

    PreparedMatMulWeights(float[] canonical, int offset, int inner, int columns) {
        if (canonical == null || offset < 0 || inner <= 0 || columns <= 0 ||
                (long) inner * columns > canonical.length - offset) {
            throw new OcrException(OcrErrorCode.INVALID_ARGUMENT,
                    "projection weights are invalid");
        }
        this.canonical = canonical;
        this.offset = offset;
        this.inner = inner;
        this.columns = columns;
    }

    /** Materializes projection weights directly from little-endian LWM bytes. */
    static PreparedMatMulWeights fromLittleEndian(ByteBuffer raw, int inner, int columns) {
        if (raw == null || inner <= 0 || columns <= 0
                || (long) inner * columns > Integer.MAX_VALUE / 4L
                || raw.remaining() != (long) inner * columns * 4L) {
            throw new OcrException(OcrErrorCode.INVALID_ARGUMENT,
                    "projection weights are invalid");
        }
        float[] prepared = new float[inner * columns];
        ByteBuffer source = raw.duplicate().order(ByteOrder.LITTLE_ENDIAN);
        for (int i = 0; i < prepared.length; i++) {
            prepared[i] = source.getFloat(i * 4);
        }
        return new PreparedMatMulWeights(prepared, 0, inner, columns);
    }

    float[] canonical() { return canonical; }
    int offset() { return offset; }
    public int getInner() { return inner; }
    public int getColumns() { return columns; }
    long packedBytes() { return (long) inner * columns * Float.BYTES; }
}
