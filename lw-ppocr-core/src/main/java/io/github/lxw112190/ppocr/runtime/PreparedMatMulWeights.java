package io.github.lxw112190.ppocr.runtime;

import io.github.lxw112190.ppocr.model.OcrErrorCode;
import io.github.lxw112190.ppocr.model.OcrException;

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

    float[] canonical() { return canonical; }
    int offset() { return offset; }
    public int getInner() { return inner; }
    public int getColumns() { return columns; }
}
