package io.github.lxw112190.ppocr.ppocr;

import io.github.lxw112190.ppocr.image.BgrImage;
import io.github.lxw112190.ppocr.model.OcrErrorCode;
import io.github.lxw112190.ppocr.model.OcrException;
import java.util.ArrayList;
import java.util.List;

/** Perspective BGR crop matching the four-point crop contract used by PP-OCR. */
public final class PerspectiveCrop {
    private PerspectiveCrop() { }

    public static BgrImage crop(BgrImage source, DetectionBox box) {
        return crop(source, box, new float[8], new double[8], null);
    }

    /** Reusable crop geometry and optional pixel buffers; callers must not invoke it concurrently. */
    public static final class Workspace {
        private final float[] values = new float[8];
        private final double[] points = new double[8];
        private final List<byte[]> pixelBuffers = new ArrayList<byte[]>();
        private byte[] arena = new byte[0];
        private int[] cropOffsets = new int[0];
        private final int[] dimensions = new int[2];

        public BgrImage crop(BgrImage source, DetectionBox box) {
            return PerspectiveCrop.crop(source, box, values, points, null);
        }

        /** Crops all boxes into one reusable arena so line crops share one byte array. */
        public void cropAll(BgrImage source, List<DetectionBox> boxes, List<BgrImage> destination) {
            if (source == null || boxes == null || destination == null) {
                throw new OcrException(OcrErrorCode.INVALID_ARGUMENT,
                        "crop batch inputs are required");
            }
            int count = boxes.size();
            ensureCropMetadata(count);
            long totalBytes = 0L;
            for (int i = 0; i < count; i++) {
                dimensions(source, boxes.get(i), values, points, dimensions);
                if (totalBytes > Integer.MAX_VALUE) {
                    throw new OcrException(OcrErrorCode.RESOURCE_LIMIT, "crop arena is too large");
                }
                cropOffsets[i] = (int) totalBytes;
                totalBytes += (long) dimensions[0] * dimensions[1] * 3L;
            }
            if (totalBytes > Integer.MAX_VALUE) {
                throw new OcrException(OcrErrorCode.RESOURCE_LIMIT, "crop arena is too large");
            }
            if (arena.length < (int) totalBytes) {
                int next = Math.max((int) totalBytes,
                        arena.length + Math.max(1, arena.length >> 1));
                arena = new byte[next];
            }
            destination.clear();
            for (int i = 0; i < count; i++) {
                destination.add(PerspectiveCrop.crop(source, boxes.get(i), values, points,
                        arena, cropOffsets[i]));
            }
        }

        /**
         * Reuses the backing pixel array associated with {@code slot}. The returned view remains
         * valid until this workspace crops into the same slot again.
         */
        public BgrImage crop(BgrImage source, DetectionBox box, int slot) {
            if (slot < 0) {
                throw new OcrException(OcrErrorCode.INVALID_ARGUMENT,
                        "crop buffer slot must not be negative");
            }
            while (pixelBuffers.size() <= slot) pixelBuffers.add(null);
            BgrImage result = PerspectiveCrop.crop(source, box, values, points,
                    pixelBuffers.get(slot));
            pixelBuffers.set(slot, result.pixels());
            return result;
        }

        private void ensureCropMetadata(int count) {
            if (cropOffsets.length >= count) return;
            cropOffsets = new int[count];
        }
    }

    private static BgrImage crop(BgrImage source, DetectionBox box,
                                 float[] values, double[] points, byte[] reusableOutput) {
        return crop(source, box, values, points, reusableOutput, 0);
    }

    private static BgrImage crop(BgrImage source, DetectionBox box,
                                 float[] values, double[] points, byte[] reusableOutput,
                                 int outputOffset) {
        if (source == null || box == null) {
            throw new OcrException(OcrErrorCode.INVALID_ARGUMENT, "source image and detection box are required");
        }
        box.copyPointsTo(values);
        if (values.length != 8) throw invalid("detection box must have four points");
        for (int i = 0; i < values.length; i++) {
            if (!Float.isFinite(values[i])) throw invalid("detection box contains non-finite coordinates");
            points[i] = values[i];
        }
        int unrotatedWidth = roundedDistance(points[0], points[1], points[2], points[3]);
        int unrotatedHeight = roundedDistance(points[0], points[1], points[6], points[7]);
        boolean rotateVertical = (double) unrotatedHeight >= (double) unrotatedWidth * 1.5;
        int outputWidth = rotateVertical ? unrotatedHeight : unrotatedWidth;
        int outputHeight = rotateVertical ? unrotatedWidth : unrotatedHeight;
        long outputBytes = (long) outputWidth * outputHeight * 3L;
        if (outputBytes > Integer.MAX_VALUE) throw new OcrException(OcrErrorCode.RESOURCE_LIMIT, "crop is too large");
        if (outputOffset < 0 || outputBytes > Integer.MAX_VALUE - (long) outputOffset) {
            throw new OcrException(OcrErrorCode.RESOURCE_LIMIT, "crop output offset is invalid");
        }
        byte[] output = reusableOutput != null && reusableOutput.length - outputOffset >= outputBytes
                ? reusableOutput : new byte[(int) (outputOffset + outputBytes)];
        double dx1 = points[2] - points[4];
        double dx2 = points[6] - points[4];
        double dx3 = points[0] - points[2] + points[4] - points[6];
        double dy1 = points[3] - points[5];
        double dy2 = points[7] - points[5];
        double dy3 = points[1] - points[3] + points[5] - points[7];
        double g = 0.0;
        double h = 0.0;
        if (Math.abs(dx3) > 1.0e-12 || Math.abs(dy3) > 1.0e-12) {
            double denominator = dx1 * dy2 - dx2 * dy1;
            if (!Double.isFinite(denominator) || Math.abs(denominator) <= 1.0e-12) {
                throw invalid("detection quadrilateral is degenerate");
            }
            g = (dx3 * dy2 - dx2 * dy3) / denominator;
            h = (dx1 * dy3 - dx3 * dy1) / denominator;
        }
        double a = points[2] - points[0] + g * points[2];
        double b = points[6] - points[0] + h * points[6];
        double c = points[0];
        double d = points[3] - points[1] + g * points[3];
        double e = points[7] - points[1] + h * points[7];
        double f = points[1];
        byte[] sourcePixels = source.pixels();
        for (int y = 0; y < unrotatedHeight; y++) {
            double v = (double) y / unrotatedHeight;
            for (int x = 0; x < unrotatedWidth; x++) {
                double u = (double) x / unrotatedWidth;
                double mappingDenominator = g * u + h * v + 1.0;
                if (!Double.isFinite(mappingDenominator) || Math.abs(mappingDenominator) <= 1.0e-12) {
                    throw invalid("detection quadrilateral mapping is invalid");
                }
                double sourceX = (a * u + b * v + c) / mappingDenominator;
                double sourceY = (d * u + e * v + f) / mappingDenominator;
                if (!Double.isFinite(sourceX) || !Double.isFinite(sourceY)) {
                    throw invalid("detection quadrilateral mapping is invalid");
                }
                int destinationX = rotateVertical ? unrotatedHeight - 1 - y : x;
                int destinationY = rotateVertical ? x : y;
                int destination = outputOffset + (destinationY * outputWidth + destinationX) * 3;
                for (int channel = 0; channel < 3; channel++) {
                    output[destination + channel] = (byte) sample(sourcePixels, source, sourceX, sourceY, channel);
                }
            }
        }
        return new BgrImage(output, outputOffset, outputWidth, outputHeight, outputWidth * 3);
    }

    private static void dimensions(BgrImage source, DetectionBox box,
                                   float[] values, double[] points, int[] output) {
        if (source == null || box == null) {
            throw new OcrException(OcrErrorCode.INVALID_ARGUMENT,
                    "source image and detection box are required");
        }
        box.copyPointsTo(values);
        for (int i = 0; i < values.length; i++) {
            if (!Float.isFinite(values[i])) throw invalid("detection box contains non-finite coordinates");
            points[i] = values[i];
        }
        int unrotatedWidth = roundedDistance(points[0], points[1], points[2], points[3]);
        int unrotatedHeight = roundedDistance(points[0], points[1], points[6], points[7]);
        boolean rotateVertical = (double) unrotatedHeight >= (double) unrotatedWidth * 1.5;
        output[0] = rotateVertical ? unrotatedHeight : unrotatedWidth;
        output[1] = rotateVertical ? unrotatedWidth : unrotatedHeight;
    }

    private static int roundedDistance(double x0, double y0, double x1, double y1) {
        double distance = Math.hypot(x1 - x0, y1 - y0);
        if (!Double.isFinite(distance) || distance < 1.0 || distance > Integer.MAX_VALUE) {
            throw invalid("detection box side length is invalid");
        }
        return Math.max(1, (int) Math.floor(distance + 0.5));
    }

    private static int sample(byte[] pixels, BgrImage source, double x, double y, int channel) {
        int x0Raw = (int) Math.floor(x);
        int y0Raw = (int) Math.floor(y);
        int x0 = clamp(x0Raw, source.width());
        int x1 = clamp(x0Raw + 1, source.width());
        int y0 = clamp(y0Raw, source.height());
        int y1 = clamp(y0Raw + 1, source.height());
        double weightX = x - x0Raw;
        double weightY = y - y0Raw;
        int sourceOffset = source.offset();
        double topLeft = pixels[sourceOffset + y0 * source.stride() + x0 * 3 + channel] & 0xff;
        double topRight = pixels[sourceOffset + y0 * source.stride() + x1 * 3 + channel] & 0xff;
        double bottomLeft = pixels[sourceOffset + y1 * source.stride() + x0 * 3 + channel] & 0xff;
        double bottomRight = pixels[sourceOffset + y1 * source.stride() + x1 * 3 + channel] & 0xff;
        double top = topLeft + (topRight - topLeft) * weightX;
        double bottom = bottomLeft + (bottomRight - bottomLeft) * weightX;
        double value = top + (bottom - top) * weightY;
        return value <= 0.0 ? 0 : value >= 255.0 ? 255 : (int) Math.floor(value + 0.5);
    }

    private static int clamp(int coordinate, int limit) {
        if (coordinate < 0) return 0;
        return coordinate >= limit ? limit - 1 : coordinate;
    }

    private static OcrException invalid(String message) {
        return new OcrException(OcrErrorCode.INVALID_MODEL, message);
    }
}
