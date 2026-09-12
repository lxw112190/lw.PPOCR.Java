package io.github.lxw112190.ppocr.ppocr;

import io.github.lxw112190.ppocr.image.BgrImage;
import io.github.lxw112190.ppocr.model.OcrErrorCode;
import io.github.lxw112190.ppocr.model.OcrException;

/** Pure-Java REC preprocessing matching lw.PPOCR.C's BGR bilinear contract. */
public final class RecPreprocess {
    public static final int INPUT_HEIGHT = 48;
    private static final double NORMALIZE_SCALE = 2.0 / 255.0;

    private RecPreprocess() { }

    public static RecPreprocessResult resizeNormalize(BgrImage source, int targetWidth) {
        if (source == null || targetWidth <= 0) {
            throw new OcrException(OcrErrorCode.INVALID_ARGUMENT, "source image and target width are required");
        }
        long outputElements = 3L * INPUT_HEIGHT * targetWidth;
        if (outputElements > Integer.MAX_VALUE) {
            throw new OcrException(OcrErrorCode.RESOURCE_LIMIT, "REC tensor is too large");
        }
        long scaledWidth = (long) INPUT_HEIGHT * source.width();
        int resizedWidth = (int) Math.min(targetWidth,
                (scaledWidth + source.height() - 1L) / source.height());
        float[] output = new float[(int) outputElements];
        float padding = (float) (128.0 * NORMALIZE_SCALE - 1.0);
        java.util.Arrays.fill(output, padding);
        long plane = (long) INPUT_HEIGHT * targetWidth;
        byte[] pixels = source.pixels();
        for (int outputY = 0; outputY < INPUT_HEIGHT; outputY++) {
            double sourceY = ((double) outputY + 0.5) * source.height() / INPUT_HEIGHT - 0.5;
            int sourceY0Raw = (int) Math.floor(sourceY);
            int sourceY1Raw = sourceY0Raw + 1;
            int sourceY0 = clamp(sourceY0Raw, source.height());
            int sourceY1 = clamp(sourceY1Raw, source.height());
            double weightY = sourceY - sourceY0Raw;
            for (int outputX = 0; outputX < resizedWidth; outputX++) {
                double sourceX = ((double) outputX + 0.5) * source.width() / resizedWidth - 0.5;
                int sourceX0Raw = (int) Math.floor(sourceX);
                int sourceX1Raw = sourceX0Raw + 1;
                int sourceX0 = clamp(sourceX0Raw, source.width());
                int sourceX1 = clamp(sourceX1Raw, source.width());
                double weightX = sourceX - sourceX0Raw;
                for (int channel = 0; channel < 3; channel++) {
                    double topLeft = pixels[sourceY0 * source.stride() + sourceX0 * 3 + channel] & 0xff;
                    double topRight = pixels[sourceY0 * source.stride() + sourceX1 * 3 + channel] & 0xff;
                    double bottomLeft = pixels[sourceY1 * source.stride() + sourceX0 * 3 + channel] & 0xff;
                    double bottomRight = pixels[sourceY1 * source.stride() + sourceX1 * 3 + channel] & 0xff;
                    double top = topLeft + (topRight - topLeft) * weightX;
                    double bottom = bottomLeft + (bottomRight - bottomLeft) * weightX;
                    double value = top + (bottom - top) * weightY;
                    int index = (int) (channel * plane + (long) outputY * targetWidth + outputX);
                    output[index] = (float) (value * NORMALIZE_SCALE - 1.0);
                }
            }
        }
        return new RecPreprocessResult(output, resizedWidth);
    }

    private static int clamp(int coordinate, int limit) {
        if (coordinate < 0) return 0;
        if (coordinate >= limit) return limit - 1;
        return coordinate;
    }
}
