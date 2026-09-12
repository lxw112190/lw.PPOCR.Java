package io.github.lxw112190.ppocr.ppocr;

import io.github.lxw112190.ppocr.image.BgrImage;
import io.github.lxw112190.ppocr.model.OcrErrorCode;
import io.github.lxw112190.ppocr.model.OcrException;

/** Chooses C-compatible dynamic DET dimensions from the source image. */
public final class DetInputShapePolicy {
    private DetInputShapePolicy() { }

    public static DetInputShape choose(BgrImage source, int maximumSideLength) {
        if (source == null || maximumSideLength < 32) {
            throw new OcrException(OcrErrorCode.INVALID_ARGUMENT,
                    "source image and DET maximum side length are required");
        }
        int maximumSide = Math.max(source.width(), source.height());
        double ratio = maximumSide > maximumSideLength
                ? (double) maximumSideLength / maximumSide : 1.0;
        int width = alignedDimension(source.width() * ratio);
        int height = alignedDimension(source.height() * ratio);
        return new DetInputShape(width, height,
                (float) width / source.width(), (float) height / source.height());
    }

    private static int alignedDimension(double value) {
        long aligned = Math.max(32L, (long) Math.floor(value / 32.0 + 0.5) * 32L);
        if (aligned > Integer.MAX_VALUE) {
            throw new OcrException(OcrErrorCode.RESOURCE_LIMIT, "DET dimensions are too large");
        }
        return (int) aligned;
    }
}
