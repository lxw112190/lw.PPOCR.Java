package io.github.lxw112190.ppocr.ppocr;

public final class RecRecognitionResult {
    private final String text;
    private final float score;
    private final int emittedCount;
    private final int resizedWidth;

    RecRecognitionResult(String text, float score, int emittedCount, int resizedWidth) {
        this.text = text;
        this.score = score;
        this.emittedCount = emittedCount;
        this.resizedWidth = resizedWidth;
    }

    public String getText() { return text; }
    public float getScore() { return score; }
    public int getEmittedCount() { return emittedCount; }
    public int getResizedWidth() { return resizedWidth; }
}
