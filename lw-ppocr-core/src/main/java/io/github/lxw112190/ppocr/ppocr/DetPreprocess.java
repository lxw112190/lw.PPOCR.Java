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
        resizeNormalizeInto(source, width, height, output, 0);
        return new DetPreprocessResult(output, width, height,
                (float) ((double) width / source.width()),
                (float) ((double) height / source.height()));
    }

    /** Resizes and normalizes directly into an existing CHW FP32 tensor. */
    public static void resizeNormalizeInto(BgrImage source, int width, int height,
                                           float[] output, int outputOffset) {
        if (source == null || width <= 0 || height <= 0 || output == null) {
            throw new OcrException(OcrErrorCode.INVALID_ARGUMENT,
                    "DET preprocess inputs are required");
        }
        long plane = (long) width * height;
        long elements = plane * 3L;
        if (elements > Integer.MAX_VALUE || outputOffset < 0
                || outputOffset > output.length
                || elements > output.length - (long) outputOffset) {
            throw new OcrException(OcrErrorCode.RESOURCE_LIMIT,
                    "DET destination is too small");
        }
        byte[] pixels = source.pixels();
        int sourceOffset = source.offset();
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
                    double topLeft = pixels[sourceOffset + sourceY0 * source.stride() + sourceX0 * 3 + channel] & 0xff;
                    double topRight = pixels[sourceOffset + sourceY0 * source.stride() + sourceX1 * 3 + channel] & 0xff;
                    double bottomLeft = pixels[sourceOffset + sourceY1 * source.stride() + sourceX0 * 3 + channel] & 0xff;
                    double bottomRight = pixels[sourceOffset + sourceY1 * source.stride() + sourceX1 * 3 + channel] & 0xff;
                    double top = topLeft + (topRight - topLeft) * weightX;
                    double bottom = bottomLeft + (bottomRight - bottomLeft) * weightX;
                    double value = (top + (bottom - top) * weightY) / 255.0;
                    int index = outputOffset
                            + (int) (channel * plane + (long) outputY * width + outputX);
                    output[index] = (float) ((value - MEAN[channel]) * INVERSE_STD[channel]);
                }
            }
        }
    }

    /** Reusable fixed-size DET preprocessing buffer; callers must not invoke it concurrently. */
    public static final class Workspace {
        private final int width;
        private final int height;
        private final float[] output;
        private final int outputOffset;
        private float widthRatio;
        private float heightRatio;

        public Workspace(int width, int height) {
            if (width <= 0 || height <= 0) {
                throw new OcrException(OcrErrorCode.INVALID_ARGUMENT,
                        "DET dimensions are required");
            }
            long elements = 3L * width * height;
            if (elements > Integer.MAX_VALUE) {
                throw new OcrException(OcrErrorCode.RESOURCE_LIMIT, "DET tensor is too large");
            }
            this.width = width;
            this.height = height;
            this.output = new float[(int) elements];
            this.outputOffset = 0;
        }

        /** Package-private view constructor used by a bound DET session. */
        Workspace(int width, int height, float[] output, int outputOffset) {
            if (width <= 0 || height <= 0 || output == null || outputOffset < 0
                    || outputOffset > output.length
                    || 3L * width * height > output.length - (long) outputOffset) {
                throw new OcrException(OcrErrorCode.RESOURCE_LIMIT,
                        "DET workspace destination is too small");
            }
            this.width = width;
            this.height = height;
            this.output = output;
            this.outputOffset = outputOffset;
        }

        public void resizeNormalize(BgrImage source) {
            if (source == null) {
                throw new OcrException(OcrErrorCode.INVALID_ARGUMENT,
                        "source image is required");
            }
            widthRatio = (float) ((double) width / source.width());
            heightRatio = (float) ((double) height / source.height());
            resizeNormalizeInto(source, width, height, output, outputOffset);
        }

        public float[] getChw() { return output; }
        public int getResizedWidth() { return width; }
        public int getResizedHeight() { return height; }
        public float getWidthRatio() { return widthRatio; }
        public float getHeightRatio() { return heightRatio; }
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
