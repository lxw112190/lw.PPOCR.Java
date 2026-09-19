package io.github.lxw112190.ppocr.ppocr;

import io.github.lxw112190.ppocr.model.OcrErrorCode;
import io.github.lxw112190.ppocr.model.OcrException;

/** Greedy CTC decoder matching lw.PPOCR.C's blank/repeat behavior. */
public final class CtcDecoder {
    private CtcDecoder() { }

    public static CtcDecodeResult decodeGreedy(float[] probabilities, int timeSteps,
                                               int classCount, PaddleOcrDictionary dictionary) {
        return decodeGreedy(probabilities, 0, timeSteps, classCount, dictionary);
    }

    public static CtcDecodeResult decodeGreedy(float[] probabilities, int probabilityOffset,
                                               int timeSteps, int classCount,
                                               PaddleOcrDictionary dictionary) {
        long elements = (long) timeSteps * classCount;
        if (probabilities == null || dictionary == null || probabilityOffset < 0
                || timeSteps <= 0 || classCount <= 0 || probabilityOffset > probabilities.length
                || elements > probabilities.length - (long) probabilityOffset
                || classCount != dictionary.classCount()) {
            throw new OcrException(OcrErrorCode.INVALID_ARGUMENT, "CTC probability shape or dictionary is invalid");
        }
        StringBuilder text = new StringBuilder();
        int previous = 0;
        int emitted = 0;
        double scoreSum = 0.0;
        for (int step = 0; step < timeSteps; step++) {
            int row = probabilityOffset + step * classCount;
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

    public static CtcDecodeResult decodeGreedy(CompactCtcOutput compact,
                                               PaddleOcrDictionary dictionary) {
        if (compact == null || dictionary == null || compact.getTimeSteps() <= 0) {
            throw new OcrException(OcrErrorCode.INVALID_ARGUMENT,
                    "compact CTC output or dictionary is invalid");
        }
        StringBuilder text = new StringBuilder();
        int previous = 0;
        int emitted = 0;
        double scoreSum = 0.0;
        int[] classIds = compact.classIds();
        float[] probabilities = compact.probabilities();
        for (int step = 0; step < classIds.length; step++) {
            int best = classIds[step];
            float probability = probabilities[step];
            if (best < 0 || best >= dictionary.classCount() || !Float.isFinite(probability)) {
                throw new OcrException(OcrErrorCode.INVALID_ARGUMENT,
                        "compact CTC output contains invalid values");
            }
            if (best != 0 && best != previous) {
                text.append(dictionary.labelForClass(best));
                scoreSum += probability;
                emitted++;
            }
            previous = best;
        }
        return new CtcDecodeResult(text.toString(), emitted == 0 ? 0.0f :
                (float) (scoreSum / emitted), emitted);
    }

    static String decodeGreedyInto(float[] probabilities, int probabilityOffset,
                                   int timeSteps, int classCount,
                                   PaddleOcrDictionary dictionary,
                                   float[] scoreOutput, int scoreIndex,
                                   int[] emittedOutput, int emittedIndex) {
        return decodeGreedyInto(probabilities, probabilityOffset, timeSteps, classCount,
                dictionary, scoreOutput, scoreIndex, emittedOutput, emittedIndex,
                new StringBuilder());
    }

    /** Internal overload which reuses the caller-owned text builder. */
    static String decodeGreedyInto(float[] probabilities, int probabilityOffset,
                                   int timeSteps, int classCount,
                                   PaddleOcrDictionary dictionary,
                                   float[] scoreOutput, int scoreIndex,
                                   int[] emittedOutput, int emittedIndex,
                                   StringBuilder text) {
        validateOutput(scoreOutput, scoreIndex, emittedOutput, emittedIndex);
        long elements = (long) timeSteps * classCount;
        if (probabilities == null || dictionary == null || probabilityOffset < 0
                || timeSteps <= 0 || classCount <= 0 || probabilityOffset > probabilities.length
                || elements > probabilities.length - (long) probabilityOffset
                || classCount != dictionary.classCount()) {
            throw new OcrException(OcrErrorCode.INVALID_ARGUMENT,
                    "CTC probability shape or dictionary is invalid");
        }
        if (text == null) {
            throw new OcrException(OcrErrorCode.INVALID_ARGUMENT,
                    "CTC text builder is required");
        }
        text.setLength(0);
        int previous = 0;
        int emitted = 0;
        double scoreSum = 0.0;
        for (int step = 0; step < timeSteps; step++) {
            int row = probabilityOffset + step * classCount;
            int best = 0;
            float bestValue = probabilities[row];
            if (!Float.isFinite(bestValue)) {
                throw new OcrException(OcrErrorCode.INVALID_ARGUMENT,
                        "CTC probabilities contain non-finite values");
            }
            for (int index = 1; index < classCount; index++) {
                float value = probabilities[row + index];
                if (!Float.isFinite(value)) {
                    throw new OcrException(OcrErrorCode.INVALID_ARGUMENT,
                            "CTC probabilities contain non-finite values");
                }
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
        emittedOutput[emittedIndex] = emitted;
        scoreOutput[scoreIndex] = emitted == 0 ? 0.0f : (float) (scoreSum / emitted);
        return text.toString();
    }

    static String decodeGreedyInto(CompactCtcOutput compact,
                                   PaddleOcrDictionary dictionary,
                                   float[] scoreOutput, int scoreIndex,
                                   int[] emittedOutput, int emittedIndex) {
        return decodeGreedyInto(compact, dictionary, scoreOutput, scoreIndex,
                emittedOutput, emittedIndex, new StringBuilder());
    }

    /** Internal overload which reuses the caller-owned text builder. */
    static String decodeGreedyInto(CompactCtcOutput compact,
                                   PaddleOcrDictionary dictionary,
                                   float[] scoreOutput, int scoreIndex,
                                   int[] emittedOutput, int emittedIndex,
                                   StringBuilder text) {
        validateOutput(scoreOutput, scoreIndex, emittedOutput, emittedIndex);
        if (compact == null || dictionary == null || compact.getTimeSteps() <= 0) {
            throw new OcrException(OcrErrorCode.INVALID_ARGUMENT,
                    "compact CTC output or dictionary is invalid");
        }
        if (text == null) {
            throw new OcrException(OcrErrorCode.INVALID_ARGUMENT,
                    "CTC text builder is required");
        }
        text.setLength(0);
        int previous = 0;
        int emitted = 0;
        double scoreSum = 0.0;
        int[] classIds = compact.classIds();
        float[] probabilities = compact.probabilities();
        for (int step = 0; step < classIds.length; step++) {
            int best = classIds[step];
            float probability = probabilities[step];
            if (best < 0 || best >= dictionary.classCount() || !Float.isFinite(probability)) {
                throw new OcrException(OcrErrorCode.INVALID_ARGUMENT,
                        "compact CTC output contains invalid values");
            }
            if (best != 0 && best != previous) {
                text.append(dictionary.labelForClass(best));
                scoreSum += probability;
                emitted++;
            }
            previous = best;
        }
        emittedOutput[emittedIndex] = emitted;
        scoreOutput[scoreIndex] = emitted == 0 ? 0.0f : (float) (scoreSum / emitted);
        return text.toString();
    }

    private static void validateOutput(float[] scoreOutput, int scoreIndex,
                                       int[] emittedOutput, int emittedIndex) {
        if (scoreOutput == null || emittedOutput == null || scoreIndex < 0
                || scoreIndex >= scoreOutput.length || emittedIndex < 0
                || emittedIndex >= emittedOutput.length) {
            throw new OcrException(OcrErrorCode.INVALID_ARGUMENT,
                    "CTC output staging buffers are invalid");
        }
    }
}
