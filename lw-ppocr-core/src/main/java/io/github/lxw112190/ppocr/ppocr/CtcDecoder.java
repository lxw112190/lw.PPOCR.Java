package io.github.lxw112190.ppocr.ppocr;

import io.github.lxw112190.ppocr.model.OcrErrorCode;
import io.github.lxw112190.ppocr.model.OcrException;

/** Greedy CTC decoder matching lw.PPOCR.C's blank/repeat behavior. */
public final class CtcDecoder {
    private CtcDecoder() { }

    public static CtcDecodeResult decodeGreedy(float[] probabilities, int timeSteps,
                                               int classCount, PaddleOcrDictionary dictionary) {
        if (probabilities == null || dictionary == null || timeSteps <= 0 || classCount <= 0 ||
                (long) timeSteps * classCount != probabilities.length || classCount != dictionary.classCount()) {
            throw new OcrException(OcrErrorCode.INVALID_ARGUMENT, "CTC probability shape or dictionary is invalid");
        }
        StringBuilder text = new StringBuilder();
        int previous = 0;
        int emitted = 0;
        double scoreSum = 0.0;
        for (int step = 0; step < timeSteps; step++) {
            int row = step * classCount;
            int best = 0;
            float bestValue = probabilities[row];
            if (!Float.isFinite(bestValue)) throw new OcrException(OcrErrorCode.INVALID_ARGUMENT, "CTC probabilities contain non-finite values");
            for (int index = 1; index < classCount; index++) {
                float value = probabilities[row + index];
                if (!Float.isFinite(value)) throw new OcrException(OcrErrorCode.INVALID_ARGUMENT, "CTC probabilities contain non-finite values");
                if (value > bestValue) {
                    best = index;
                    bestValue = value;
                }
            }
            if (best != 0 && best != previous) {
                text.append(dictionary.labelForClass(best));
                scoreSum += bestValue;
                emitted++;
            }
            previous = best;
        }
        return new CtcDecodeResult(text.toString(), emitted == 0 ? 0.0f : (float) (scoreSum / emitted), emitted);
    }
}
