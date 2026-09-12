package io.github.lxw112190.ppocr.ppocr;

public final class ClsClassificationResult {
    private final int label;
    private final float score;
    private final int orientationDegrees;
    private final int resizedWidth;

    ClsClassificationResult(int label, float score, int orientationDegrees, int resizedWidth) {
        this.label = label;
        this.score = score;
        this.orientationDegrees = orientationDegrees;
        this.resizedWidth = resizedWidth;
    }

    public int getLabel() { return label; }
    public float getScore() { return score; }
    public int getOrientationDegrees() { return orientationDegrees; }
    public int getResizedWidth() { return resizedWidth; }

    public boolean requiresRotation(float threshold) {
        return label != 0 && score > threshold;
    }
}
