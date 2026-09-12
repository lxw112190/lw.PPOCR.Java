package io.github.lxw112190.ppocr.ppocr;

import io.github.lxw112190.ppocr.model.OcrErrorCode;
import io.github.lxw112190.ppocr.model.OcrException;
import java.util.Arrays;

/** Clockwise quadrilateral in source-image coordinates. */
public final class DetectionBox {
    private final float[] points;
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
        this.points = points.clone();
        this.score = score;
    }

    public float[] getPoints() { return points.clone(); }
    public float getScore() { return score; }

    void copyPointsTo(float[] destination) {
        if (destination == null || destination.length != points.length) {
            throw new OcrException(OcrErrorCode.INVALID_ARGUMENT,
                    "detection point destination is invalid");
        }
        System.arraycopy(points, 0, destination, 0, points.length);
    }

    @Override
    public String toString() {
        return "DetectionBox{" + Arrays.toString(points) + ", score=" + score + "}";
    }
}
