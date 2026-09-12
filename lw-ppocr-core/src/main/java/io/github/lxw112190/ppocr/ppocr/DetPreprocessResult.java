package io.github.lxw112190.ppocr.ppocr;

public final class DetPreprocessResult {
    private final float[] chw;
    private final int resizedWidth;
    private final int resizedHeight;
    private final float widthRatio;
    private final float heightRatio;

    DetPreprocessResult(float[] chw, int resizedWidth, int resizedHeight,
                        float widthRatio, float heightRatio) {
        this.chw = chw;
        this.resizedWidth = resizedWidth;
        this.resizedHeight = resizedHeight;
        this.widthRatio = widthRatio;
        this.heightRatio = heightRatio;
    }

    public float[] getChw() { return chw; }
    public int getResizedWidth() { return resizedWidth; }
    public int getResizedHeight() { return resizedHeight; }
    public float getWidthRatio() { return widthRatio; }
    public float getHeightRatio() { return heightRatio; }
}
