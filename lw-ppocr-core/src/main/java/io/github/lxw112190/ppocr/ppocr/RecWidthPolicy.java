package io.github.lxw112190.ppocr.ppocr;

import io.github.lxw112190.ppocr.image.BgrImage;
import io.github.lxw112190.ppocr.model.OcrErrorCode;
import io.github.lxw112190.ppocr.model.OcrException;

/** Adaptive REC target-width buckets shared with the C runtime. */
public final class RecWidthPolicy {
    private static final int[] BUCKETS = {192, 320, 480, 640, 960};

    private RecWidthPolicy() { }

    static int bucketCount() {
        return BUCKETS.length;
    }

    /** Returns the fixed-bucket slot, or -1 for a width outside the shared policy. */
    static int bucketIndex(int width) {
        for (int i = 0; i < BUCKETS.length; i++) {
            if (BUCKETS[i] == width) return i;
        }
        return -1;
    }

    static int chooseBucketIndex(BgrImage source, int maximumWidth) {
        validate(source, maximumWidth);
        long scaledWidth = ((long) RecPreprocess.INPUT_HEIGHT * source.width() + source.height() - 1L)
                / source.height();
        int lastValid = -1;
        for (int i = 0; i < BUCKETS.length; i++) {
            int bucket = BUCKETS[i];
            if (bucket > maximumWidth) break;
            lastValid = i;
            if (scaledWidth <= bucket) return i;
        }
        return lastValid;
    }

    static int widthForIndex(int index, int maximumWidth) {
        if (maximumWidth <= 0) {
            throw new OcrException(OcrErrorCode.INVALID_ARGUMENT, "REC width limit is required");
        }
        return index < 0 ? maximumWidth : Math.min(BUCKETS[index], maximumWidth);
    }

    public static int chooseTargetWidth(BgrImage source, int maximumWidth) {
        return widthForIndex(chooseBucketIndex(source, maximumWidth), maximumWidth);
    }

    private static void validate(BgrImage source, int maximumWidth) {
        if (source == null || maximumWidth <= 0) {
            throw new OcrException(OcrErrorCode.INVALID_ARGUMENT,
                    "source image and REC width limit are required");
        }
    }
}
