package io.github.lxw112190.ppocr.ppocr;

public final class CtcDecodeResult {
    private final String text;
    private final float score;
    private final int emittedCount;

    CtcDecodeResult(String text, float score, int emittedCount) {
        this.text = text;
        this.score = score;
        this.emittedCount = emittedCount;
    }

    public String getText() { return text; }
    public float getScore() { return score; }
    public int getEmittedCount() { return emittedCount; }
}
