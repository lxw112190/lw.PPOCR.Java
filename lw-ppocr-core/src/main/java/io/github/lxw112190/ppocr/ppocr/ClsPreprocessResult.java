package io.github.lxw112190.ppocr.ppocr;

public final class ClsPreprocessResult {
    private final float[] chw;
    private final int resizedWidth;

    ClsPreprocessResult(float[] chw, int resizedWidth) {
        this.chw = chw;
        this.resizedWidth = resizedWidth;
    }

    public float[] getChw() { return chw; }
    public int getResizedWidth() { return resizedWidth; }
}
