package io.github.lxw112190.ppocr.runtime;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;

/**
 * Read-only LWM constant storage with lazy canonical FP32 materialization.
 *
 * The raw view keeps the model bytes as the source of truth.  A canonical
 * array is created only when a kernel actually needs this tensor.
 */
final class ConstantTensor {
    private final ByteBuffer raw;
    private final int length;
    private volatile float[] canonical;

    ConstantTensor(ByteBuffer raw, int length) {
        if (raw == null || length < 0 || (long) length * 4L != raw.remaining()) {
            throw new IllegalArgumentException("constant byte length does not match tensor shape");
        }
        this.raw = raw.slice().asReadOnlyBuffer().order(ByteOrder.LITTLE_ENDIAN);
        this.length = length;
    }

    int length() {
        return length;
    }

    /** Returns a read-only little-endian view without materializing a float array. */
    ByteBuffer raw() {
        return raw.duplicate().order(ByteOrder.LITTLE_ENDIAN);
    }

    float[] canonical() {
        float[] value = canonical;
        if (value != null) return value;
        synchronized (this) {
            value = canonical;
            if (value == null) {
                value = new float[length];
                ByteBuffer source = raw.duplicate().order(ByteOrder.LITTLE_ENDIAN);
                for (int i = 0; i < length; i++) {
                    value[i] = source.getFloat(i * 4);
                }
                canonical = value;
            }
            return value;
        }
    }

    boolean isMaterialized() {
        return canonical != null;
    }

    long decodedBytes() {
        return canonical == null ? 0L : (long) canonical.length * Float.BYTES;
    }
}
