package io.github.lxw112190.ppocr.runtime;

import io.github.lxw112190.ppocr.model.OcrErrorCode;
import io.github.lxw112190.ppocr.model.OcrException;

/**
 * Non-owning contiguous FP32 tensor view.
 *
 * <p>The backing array belongs to an {@link InferenceSession} and remains
 * valid until the owning session is closed.</p>
 */
public final class FloatTensorView {
    private final float[] array;
    private final int offset;
    private final int length;

    FloatTensorView(float[] array, int offset, int length) {
        if (array == null || offset < 0 || length < 0
                || offset > array.length || length > array.length - offset) {
            throw new OcrException(OcrErrorCode.INVALID_ARGUMENT,
                    "invalid FP32 tensor view");
        }
        this.array = array;
        this.offset = offset;
        this.length = length;
    }

    public float[] array() {
        return array;
    }

    public int offset() {
        return offset;
    }

    public int length() {
        return length;
    }
}
