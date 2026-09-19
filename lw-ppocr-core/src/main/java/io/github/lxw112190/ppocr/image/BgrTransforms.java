package io.github.lxw112190.ppocr.image;

import io.github.lxw112190.ppocr.model.OcrErrorCode;
import io.github.lxw112190.ppocr.model.OcrException;

/** Non-mutating transforms for caller-owned BGR images. */
public final class BgrTransforms {
    private BgrTransforms() { }

    public static BgrImage rotate180(BgrImage source) {
        if (source == null) {
            throw new OcrException(OcrErrorCode.INVALID_ARGUMENT, "source image is required");
        }
        long rowBytes = (long) source.width() * 3;
        long byteCount = rowBytes * source.height();
        if (byteCount > Integer.MAX_VALUE) {
            throw new OcrException(OcrErrorCode.RESOURCE_LIMIT, "rotated image is too large");
        }
        byte[] rotated = new byte[(int) byteCount];
        byte[] pixels = source.pixels();
        for (int y = 0; y < source.height(); y++) {
            int sourceRow = source.offset() + y * source.stride();
            int destinationRow = (source.height() - 1 - y) * (int) rowBytes;
            for (int x = 0; x < source.width(); x++) {
                int sourcePixel = sourceRow + x * 3;
                int destinationPixel = destinationRow + (source.width() - 1 - x) * 3;
                rotated[destinationPixel] = pixels[sourcePixel];
                rotated[destinationPixel + 1] = pixels[sourcePixel + 1];
                rotated[destinationPixel + 2] = pixels[sourcePixel + 2];
            }
        }
        return new BgrImage(rotated, source.width(), source.height(), (int) rowBytes);
    }
}
