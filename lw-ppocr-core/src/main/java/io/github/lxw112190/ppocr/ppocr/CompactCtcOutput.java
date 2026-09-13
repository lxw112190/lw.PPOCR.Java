package io.github.lxw112190.ppocr.ppocr;

/** Reusable per-time-step output for fused REC projection and CTC decoding. */
public final class CompactCtcOutput {
    private final int[] classIds;
    private final float[] logits;
    private final float[] probabilities;

    CompactCtcOutput(int timeSteps) {
        this.classIds = new int[timeSteps];
        this.logits = new float[timeSteps];
        this.probabilities = new float[timeSteps];
    }

    int[] classIds() { return classIds; }
    float[] logits() { return logits; }
    float[] probabilities() { return probabilities; }
    public int getTimeSteps() { return classIds.length; }
}
