package io.github.lxw112190.ppocr.ppocr;

import io.github.lxw112190.ppocr.model.OcrErrorCode;
import io.github.lxw112190.ppocr.model.OcrException;
import java.util.Arrays;

/** Clockwise quadrilateral in source-image coordinates. */
public final class DetectionBox {
    private final float point0;
    private final float point1;
    private final float point2;
    private final float point3;
    private final float point4;
    private final float point5;
    private final float point6;
    private final float point7;
    private final float score;

    public DetectionBox(float[] points, float score) {
        if (points == null || points.length != 8 || !Float.isFinite(score)) {
            throw new OcrException(OcrErrorCode.INVALID_ARGUMENT, "detection box is invalid");
        }
        for (float point : points) {
            if (!Float.isFinite(point)) {
                throw new OcrException(OcrErrorCode.INVALID_ARGUMENT, "detection box contains non-finite coordinates");
            }
        }
        this(point(points, 0), point(points, 1), point(points, 2), point(points, 3),
                point(points, 4), point(points, 5), point(points, 6), point(points, 7), score);
    }

    DetectionBox(float point0, float point1, float point2, float point3,
                 float point4, float point5, float point6, float point7, float score) {
        if (!Float.isFinite(score) || !Float.isFinite(point0) || !Float.isFinite(point1) ||
                !Float.isFinite(point2) || !Float.isFinite(point3) || !Float.isFinite(point4) ||
                !Float.isFinite(point5) || !Float.isFinite(point6) || !Float.isFinite(point7)) {
            throw new OcrException(OcrErrorCode.INVALID_ARGUMENT, "detection box is invalid");
        }
        this.point0 = point0;
        this.point1 = point1;
        this.point2 = point2;
        this.point3 = point3;
        this.point4 = point4;
        this.point5 = point5;
        this.point6 = point6;
        this.point7 = point7;
        this.score = score;
    }

    public float[] getPoints() {
        return new float[] {point0, point1, point2, point3, point4, point5, point6, point7};
    }
    public float getScore() { return score; }

    void copyPointsTo(float[] destination) {
        if (destination == null || destination.length != 8) {
            throw new OcrException(OcrErrorCode.INVALID_ARGUMENT,
                    "detection point destination is invalid");
        }
        destination[0] = point0;
        destination[1] = point1;
        destination[2] = point2;
        destination[3] = point3;
        destination[4] = point4;
        destination[5] = point5;
        destination[6] = point6;
        destination[7] = point7;
    }

    @Override
    public String toString() {
        return "DetectionBox{" + Arrays.toString(getPoints()) + ", score=" + score + "}";
    }

    private static float point(float[] points, int index) {
        return points[index];
    }
}
