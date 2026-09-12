package io.github.lxw112190.ppocr.ppocr;

import io.github.lxw112190.ppocr.image.BgrImage;
import io.github.lxw112190.ppocr.model.OcrErrorCode;
import io.github.lxw112190.ppocr.model.OcrException;

/** Perspective BGR crop matching the four-point crop contract used by PP-OCR. */
public final class PerspectiveCrop {
    private PerspectiveCrop() { }

    public static BgrImage crop(BgrImage source, DetectionBox box) {
        return crop(source, box, new float[8], new double[8]);
    }

    /** Reusable crop geometry workspace; callers must not invoke it concurrently. */
    public static final class Workspace {
        private final float[] values = new float[8];
        private final double[] points = new double[8];

        public BgrImage crop(BgrImage source, DetectionBox box) {
            return PerspectiveCrop.crop(source, box, values, points);
        }
    }

    private static BgrImage crop(BgrImage source, DetectionBox box,
                                 float[] values, double[] points) {
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
        byte[] output = new byte[(int) outputBytes];
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
                int destination = (destinationY * outputWidth + destinationX) * 3;
                for (int channel = 0; channel < 3; channel++) {
                    output[destination + channel] = (byte) sample(sourcePixels, source, sourceX, sourceY, channel);
                }
            }
        }
        return new BgrImage(output, outputWidth, outputHeight, outputWidth * 3);
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
        double topLeft = pixels[y0 * source.stride() + x0 * 3 + channel] & 0xff;
        double topRight = pixels[y0 * source.stride() + x1 * 3 + channel] & 0xff;
        double bottomLeft = pixels[y1 * source.stride() + x0 * 3 + channel] & 0xff;
        double bottomRight = pixels[y1 * source.stride() + x1 * 3 + channel] & 0xff;
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
