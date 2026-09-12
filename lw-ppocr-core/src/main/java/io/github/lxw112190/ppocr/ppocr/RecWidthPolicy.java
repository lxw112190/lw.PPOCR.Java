package io.github.lxw112190.ppocr.ppocr;

import io.github.lxw112190.ppocr.image.BgrImage;
import io.github.lxw112190.ppocr.model.OcrErrorCode;
import io.github.lxw112190.ppocr.model.OcrException;

/** Adaptive REC target-width buckets shared with the C runtime. */
public final class RecWidthPolicy {
    private static final int[] BUCKETS = {192, 320, 480, 640, 960};

    private RecWidthPolicy() { }

    public static int chooseTargetWidth(BgrImage source, int maximumWidth) {
        if (source == null || maximumWidth <= 0) {
            throw new OcrException(OcrErrorCode.INVALID_ARGUMENT, "source image and REC width limit are required");
        }
        long scaledWidth = ((long) RecPreprocess.INPUT_HEIGHT * source.width() + source.height() - 1L)
                / source.height();
        for (int bucket : BUCKETS) {
            if (bucket > maximumWidth) break;
            if (scaledWidth <= bucket) return bucket;
        }
        return maximumWidth;
    }
}
