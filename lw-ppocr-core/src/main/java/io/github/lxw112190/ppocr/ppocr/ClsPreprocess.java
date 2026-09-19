package io.github.lxw112190.ppocr.ppocr;

import io.github.lxw112190.ppocr.image.BgrImage;
import io.github.lxw112190.ppocr.model.OcrErrorCode;
import io.github.lxw112190.ppocr.model.OcrException;

/** Pure-Java CLS preprocessing matching lw.PPOCR.C's fixed BGR contract. */
public final class ClsPreprocess {
    public static final int INPUT_HEIGHT = 80;
    public static final int INPUT_WIDTH = 160;
    private static final double NORMALIZE_SCALE = 2.0 / 255.0;

    private ClsPreprocess() { }

    public static ClsPreprocessResult resizeNormalize(BgrImage source) {
        if (source == null) {
            throw new OcrException(OcrErrorCode.INVALID_ARGUMENT, "source image is required");
        }
        long scaledWidth = (long) INPUT_HEIGHT * source.width();
        int resizedWidth = (int) Math.min(INPUT_WIDTH,
                (scaledWidth + source.height() - 1L) / source.height());
        long plane = (long) INPUT_HEIGHT * INPUT_WIDTH;
        float[] output = new float[(int) (3L * plane)];
        resizeNormalizeInto(source, output, 0);
        return new ClsPreprocessResult(output, resizedWidth);
    }

    /** Resizes and normalizes directly into a fixed-size CHW FP32 tensor. */
    public static void resizeNormalizeInto(BgrImage source, float[] output, int outputOffset) {
        if (source == null || output == null) {
            throw new OcrException(OcrErrorCode.INVALID_ARGUMENT,
                    "CLS preprocess inputs are required");
        }
        long plane = (long) INPUT_HEIGHT * INPUT_WIDTH;
        long required = 3L * plane;
        if (outputOffset < 0 || outputOffset > output.length
                || required > output.length - (long) outputOffset) {
            throw new OcrException(OcrErrorCode.RESOURCE_LIMIT,
                    "CLS destination is too small");
        }
        long scaledWidth = (long) INPUT_HEIGHT * source.width();
        int resizedWidth = (int) Math.min(INPUT_WIDTH,
                (scaledWidth + source.height() - 1L) / source.height());
        if (resizedWidth < INPUT_WIDTH) {
            for (int channel = 0; channel < 3; channel++) {
                int channelOffset = outputOffset + channel * INPUT_HEIGHT * INPUT_WIDTH;
                for (int outputY = 0; outputY < INPUT_HEIGHT; outputY++) {
                    int paddingStart = channelOffset + outputY * INPUT_WIDTH + resizedWidth;
                    java.util.Arrays.fill(output, paddingStart,
                            paddingStart + INPUT_WIDTH - resizedWidth, -1.0f);
                }
            }
        }
        byte[] pixels = source.pixels();
        int sourceOffset = source.offset();
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
                    double topLeft = pixels[sourceOffset + sourceY0 * source.stride() + sourceX0 * 3 + channel] & 0xff;
                    double topRight = pixels[sourceOffset + sourceY0 * source.stride() + sourceX1 * 3 + channel] & 0xff;
                    double bottomLeft = pixels[sourceOffset + sourceY1 * source.stride() + sourceX0 * 3 + channel] & 0xff;
                    double bottomRight = pixels[sourceOffset + sourceY1 * source.stride() + sourceX1 * 3 + channel] & 0xff;
                    double top = topLeft + (topRight - topLeft) * weightX;
                    double bottom = bottomLeft + (bottomRight - bottomLeft) * weightX;
                    double value = top + (bottom - top) * weightY;
                    int index = outputOffset
                            + (int) (channel * plane + (long) outputY * INPUT_WIDTH + outputX);
                    output[index] = (float) (value * NORMALIZE_SCALE - 1.0);
                }
            }
        }
    }

    /** Reusable CLS preprocessing buffer; callers must not invoke it concurrently. */
    public static final class Workspace {
        private final float[] output;
        private int resizedWidth;

        public Workspace() {
            long plane = (long) INPUT_HEIGHT * INPUT_WIDTH;
            this.output = new float[(int) (3L * plane)];
        }

        public void resizeNormalize(BgrImage source) {
            if (source == null) {
                throw new OcrException(OcrErrorCode.INVALID_ARGUMENT,
                        "source image is required");
            }
            long scaledWidth = (long) INPUT_HEIGHT * source.width();
            resizedWidth = (int) Math.min(INPUT_WIDTH,
                    (scaledWidth + source.height() - 1L) / source.height());
            ClsPreprocess.resizeNormalizeInto(source, output, 0);
        }

        public float[] getChw() { return output; }
        public int getResizedWidth() { return resizedWidth; }
    }

    private static int clamp(int coordinate, int limit) {
        if (coordinate < 0) return 0;
        if (coordinate >= limit) return limit - 1;
        return coordinate;
    }
}
