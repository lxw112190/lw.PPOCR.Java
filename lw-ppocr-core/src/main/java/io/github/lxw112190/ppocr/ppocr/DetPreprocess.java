package io.github.lxw112190.ppocr.ppocr;

import io.github.lxw112190.ppocr.image.BgrImage;
import io.github.lxw112190.ppocr.model.OcrErrorCode;
import io.github.lxw112190.ppocr.model.OcrException;

/** Pure-Java DET preprocessing matching lw.PPOCR.C's BGR input contract. */
public final class DetPreprocess {
    private static final double[] MEAN = {0.485, 0.456, 0.406};
    private static final double[] INVERSE_STD = {1.0 / 0.229, 1.0 / 0.224, 1.0 / 0.225};

    private DetPreprocess() { }

    public static DetPreprocessResult resizeNormalize(BgrImage source, int limitSideLength) {
        if (source == null || limitSideLength < 32) {
            throw new OcrException(OcrErrorCode.INVALID_ARGUMENT, "source image and DET side limit are required");
        }
        int maximumSide = Math.max(source.width(), source.height());
        double ratio = maximumSide > limitSideLength ? (double) limitSideLength / maximumSide : 1.0;
        long resizedWidth = roundedMultipleOf32(source.width() * ratio);
        long resizedHeight = roundedMultipleOf32(source.height() * ratio);
        resizedWidth = Math.max(32L, resizedWidth);
        resizedHeight = Math.max(32L, resizedHeight);
        if (resizedWidth > Integer.MAX_VALUE || resizedHeight > Integer.MAX_VALUE) {
            throw new OcrException(OcrErrorCode.RESOURCE_LIMIT, "DET image dimensions are too large");
        }
        return resizeNormalize(source, (int) resizedWidth, (int) resizedHeight);
    }

    public static DetPreprocessResult resizeNormalize(BgrImage source, int resizedWidth,
                                                       int resizedHeight) {
        if (source == null || resizedWidth <= 0 || resizedHeight <= 0) {
            throw new OcrException(OcrErrorCode.INVALID_ARGUMENT, "source image and DET dimensions are required");
        }
        int width = resizedWidth;
        int height = resizedHeight;
        long plane = (long) width * height;
        long elements = plane * 3L;
        if (elements > Integer.MAX_VALUE) {
            throw new OcrException(OcrErrorCode.RESOURCE_LIMIT, "DET tensor is too large");
        }
        float[] output = new float[(int) elements];
        byte[] pixels = source.pixels();
        for (int outputY = 0; outputY < height; outputY++) {
            double sourceY = ((double) outputY + 0.5) * source.height() / height - 0.5;
            int sourceY0Raw = (int) Math.floor(sourceY);
            int sourceY1Raw = sourceY0Raw + 1;
            int sourceY0 = clamp(sourceY0Raw, source.height());
            int sourceY1 = clamp(sourceY1Raw, source.height());
            double weightY = sourceY - sourceY0Raw;
            for (int outputX = 0; outputX < width; outputX++) {
                double sourceX = ((double) outputX + 0.5) * source.width() / width - 0.5;
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
                    double value = (top + (bottom - top) * weightY) / 255.0;
                    int index = (int) (channel * plane + (long) outputY * width + outputX);
                    output[index] = (float) ((value - MEAN[channel]) * INVERSE_STD[channel]);
                }
            }
        }
        return new DetPreprocessResult(output, width, height,
                (float) ((double) width / source.width()),
                (float) ((double) height / source.height()));
    }

    private static long roundedMultipleOf32(double value) {
        return (long) Math.floor(value / 32.0 + 0.5) * 32L;
    }

    private static int clamp(int coordinate, int limit) {
        if (coordinate < 0) return 0;
        if (coordinate >= limit) return limit - 1;
        return coordinate;
    }
}
