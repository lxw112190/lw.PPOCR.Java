package io.github.lxw112190.ppocr.ppocr;

import java.util.Arrays;

/** Clockwise quadrilateral in source-image coordinates. */
public final class DetectionBox {
    private final float[] points;
    private final float score;

    DetectionBox(float[] points, float score) {
        this.points = points.clone();
        this.score = score;
    }

    public float[] getPoints() { return points.clone(); }
    public float getScore() { return score; }

    @Override
    public String toString() {
        return "DetectionBox{" + Arrays.toString(points) + ", score=" + score + "}";
    }
}
