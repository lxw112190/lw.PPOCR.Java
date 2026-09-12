package io.github.lxw112190.ppocr.ppocr;

/** C-compatible resized DET input shape and source-to-map scale. */
public final class DetInputShape {
    private final int inputWidth;
    private final int inputHeight;
    private final float scaleX;
    private final float scaleY;

    DetInputShape(int inputWidth, int inputHeight, float scaleX, float scaleY) {
        this.inputWidth = inputWidth;
        this.inputHeight = inputHeight;
        this.scaleX = scaleX;
        this.scaleY = scaleY;
    }

    public int getInputWidth() { return inputWidth; }
    public int getInputHeight() { return inputHeight; }
    public float getScaleX() { return scaleX; }
    public float getScaleY() { return scaleY; }
}
