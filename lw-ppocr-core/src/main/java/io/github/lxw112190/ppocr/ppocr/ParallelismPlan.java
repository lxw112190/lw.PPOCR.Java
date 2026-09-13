package io.github.lxw112190.ppocr.ppocr;

import io.github.lxw112190.ppocr.model.OcrErrorCode;
import io.github.lxw112190.ppocr.model.OcrException;

/** Immutable worker plan constrained by one visible CPU budget. */
public final class ParallelismPlan {
    private final int availableProcessors;
    private final int lineWorkers;
    private final int classifierWorkers;
    private final int recognizerWorkers;
    private final int recognizerIntraOp;
    private final int detectorIntraOp;

    private ParallelismPlan(int availableProcessors, int lineWorkers,
                            int classifierWorkers, int recognizerWorkers,
                            int recognizerIntraOp, int detectorIntraOp) {
        this.availableProcessors = availableProcessors;
        this.lineWorkers = lineWorkers;
        this.classifierWorkers = classifierWorkers;
        this.recognizerWorkers = recognizerWorkers;
        this.recognizerIntraOp = recognizerIntraOp;
        this.detectorIntraOp = detectorIntraOp;
    }

    public static ParallelismPlan automatic(int availableProcessors, int lineCount) {
        requirePositive(availableProcessors, "available processor count");
        requireNonNegative(lineCount, "line count");
        int workItems = Math.max(1, lineCount);
        int maximumLineWorkers = availableProcessors == 1 ? 1
                : availableProcessors == 2 ? 2 : 4;
        int lineWorkers = Math.min(workItems, maximumLineWorkers);
        int desiredIntraOp = availableProcessors >= 16
                ? Math.min(4, availableProcessors / 4)
                : availableProcessors >= 8 ? 2 : 1;
        int recognizerIntraOp = Math.min(desiredIntraOp,
                Math.max(1, availableProcessors / lineWorkers));
        int detectorIntraOp = Math.min(availableProcessors, 4);
        return new ParallelismPlan(availableProcessors, lineWorkers, lineWorkers,
                lineWorkers, recognizerIntraOp, detectorIntraOp);
    }

    static ParallelismPlan manual(int availableProcessors, int lineCount,
                                  int classifierWorkers, int recognizerWorkers) {
        requirePositive(availableProcessors, "available processor count");
        requireNonNegative(lineCount, "line count");
        requirePositive(classifierWorkers, "classifier workers");
        requirePositive(recognizerWorkers, "recognizer workers");
        int workItems = Math.max(1, lineCount);
        int cls = Math.min(workItems, classifierWorkers);
        int rec = Math.min(workItems, recognizerWorkers);
        return new ParallelismPlan(availableProcessors, Math.max(cls, rec), cls, rec, 1, 1);
    }

    public int getAvailableProcessors() { return availableProcessors; }
    public int getLineWorkers() { return lineWorkers; }
    public int getClassifierWorkers() { return classifierWorkers; }
    public int getRecognizerWorkers() { return recognizerWorkers; }
    public int getRecognizerIntraOp() { return recognizerIntraOp; }
    public int getDetectorIntraOp() { return detectorIntraOp; }

    private static void requirePositive(int value, String name) {
        if (value <= 0) throw invalid(name + " must be positive");
    }

    private static void requireNonNegative(int value, String name) {
        if (value < 0) throw invalid(name + " must not be negative");
    }

    private static OcrException invalid(String message) {
        return new OcrException(OcrErrorCode.INVALID_ARGUMENT, message);
    }
}
