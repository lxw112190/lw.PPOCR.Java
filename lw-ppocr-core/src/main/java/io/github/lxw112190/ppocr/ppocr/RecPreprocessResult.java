package io.github.lxw112190.ppocr.ppocr;

public final class RecPreprocessResult {
    private final float[] chw;
    private final int resizedWidth;

    RecPreprocessResult(float[] chw, int resizedWidth) {
        this.chw = chw;
        this.resizedWidth = resizedWidth;
    }

    public float[] getChw() { return chw; }
    public int getResizedWidth() { return resizedWidth; }
}
