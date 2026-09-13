package io.github.lxw112190.ppocr.ppocr;

import io.github.lxw112190.ppocr.model.OcrErrorCode;
import io.github.lxw112190.ppocr.model.OcrException;

/** Immutable thresholds, shape limits, and execution policies for the OCR pipeline. */
public final class PaddleOcrOptions {
    private final float detectionBitmapThreshold;
    private final float detectionBoxThreshold;
    private final float detectionUnclipRatio;
    private final boolean detectionDilation;
    private final int maxDetectionCandidates;
    private final float classifierThreshold;
    private final int readingOrder;
    private final int classificationParallelism;
    private final int recognitionParallelism;
    private final ParallelismPolicy parallelismPolicy;
    private final int detectionMaximumSideLength;

    private PaddleOcrOptions(float detectionBitmapThreshold, float detectionBoxThreshold,
                             float detectionUnclipRatio, boolean detectionDilation,
                             int maxDetectionCandidates, float classifierThreshold,
                             int readingOrder, int classificationParallelism,
                             int recognitionParallelism, ParallelismPolicy parallelismPolicy,
                             int detectionMaximumSideLength) {
        if (!Float.isFinite(detectionBitmapThreshold) || detectionBitmapThreshold < 0.0f ||
                detectionBitmapThreshold > 1.0f || !Float.isFinite(detectionBoxThreshold) ||
                detectionBoxThreshold < 0.0f || detectionBoxThreshold > 1.0f ||
                !Float.isFinite(detectionUnclipRatio) || detectionUnclipRatio <= 0.0f ||
                detectionUnclipRatio > 10.0f ||
                maxDetectionCandidates <= 0 || !Float.isFinite(classifierThreshold) ||
                classifierThreshold < 0.0f || classifierThreshold > 1.0f ||
                readingOrder < ReadingOrder.HORIZONTAL_LTR || readingOrder > ReadingOrder.VERTICAL_LTR ||
                classificationParallelism <= 0 || classificationParallelism > 64 ||
                recognitionParallelism <= 0 || recognitionParallelism > 64 ||
                parallelismPolicy == null ||
                detectionMaximumSideLength < 32) {
            throw new OcrException(OcrErrorCode.INVALID_ARGUMENT, "OCR options are invalid");
        }
        this.detectionBitmapThreshold = detectionBitmapThreshold;
        this.detectionBoxThreshold = detectionBoxThreshold;
        this.detectionUnclipRatio = detectionUnclipRatio;
        this.detectionDilation = detectionDilation;
        this.maxDetectionCandidates = maxDetectionCandidates;
        this.classifierThreshold = classifierThreshold;
        this.readingOrder = readingOrder;
        this.classificationParallelism = classificationParallelism;
        this.recognitionParallelism = recognitionParallelism;
        this.parallelismPolicy = parallelismPolicy;
        this.detectionMaximumSideLength = detectionMaximumSideLength;
    }

    public static PaddleOcrOptions defaults() {
        return builder().build();
    }

    public static Builder builder() {
        return new Builder();
    }

    public float getDetectionBitmapThreshold() { return detectionBitmapThreshold; }
    public float getDetectionBoxThreshold() { return detectionBoxThreshold; }
    public float getDetectionUnclipRatio() { return detectionUnclipRatio; }
    public boolean isDetectionDilation() { return detectionDilation; }
    public int getMaxDetectionCandidates() { return maxDetectionCandidates; }
    public float getClassifierThreshold() { return classifierThreshold; }
    public int getReadingOrder() { return readingOrder; }
    public int getClassificationParallelism() { return classificationParallelism; }
    public int getRecognitionParallelism() { return recognitionParallelism; }
    public ParallelismPolicy getParallelismPolicy() { return parallelismPolicy; }
    public int getDetectionMaximumSideLength() { return detectionMaximumSideLength; }

    public static final class Builder {
        private float detectionBitmapThreshold = 0.3f;
        private float detectionBoxThreshold = 0.6f;
        private float detectionUnclipRatio = 1.6f;
        private boolean detectionDilation;
        private int maxDetectionCandidates = 1000;
        private float classifierThreshold = 0.9f;
        private int readingOrder = ReadingOrder.HORIZONTAL_LTR;
        private int classificationParallelism = 1;
        private int recognitionParallelism = 1;
        private ParallelismPolicy parallelismPolicy = ParallelismPolicy.MANUAL;
        private int detectionMaximumSideLength = 960;

        public Builder setDetectionBitmapThreshold(float value) {
            detectionBitmapThreshold = value;
            return this;
        }

        public Builder setDetectionBoxThreshold(float value) {
            detectionBoxThreshold = value;
            return this;
        }

        public Builder setDetectionUnclipRatio(float value) {
            detectionUnclipRatio = value;
            return this;
        }

        public Builder setDetectionDilation(boolean value) {
            detectionDilation = value;
            return this;
        }

        public Builder setMaxDetectionCandidates(int value) {
            maxDetectionCandidates = value;
            return this;
        }

        public Builder setClassifierThreshold(float value) {
            classifierThreshold = value;
            return this;
        }

        public Builder setReadingOrder(int value) {
            readingOrder = value;
            return this;
        }

        /** Sets the maximum number of fixed-shape CLS evaluations run concurrently. */
        public Builder setClassificationParallelism(int value) {
            classificationParallelism = value;
            parallelismPolicy = ParallelismPolicy.MANUAL;
            return this;
        }

        /** Sets the maximum number of REC width groups evaluated concurrently. */
        public Builder setRecognitionParallelism(int value) {
            recognitionParallelism = value;
            parallelismPolicy = ParallelismPolicy.MANUAL;
            return this;
        }

        /** Selects automatic CPU budgeting or the explicit CLS/REC worker values. */
        public Builder setParallelismMode(ParallelismPolicy value) {
            parallelismPolicy = value;
            return this;
        }

        /** Zero selects AUTO; a positive value sets both CLS and REC worker limits. */
        public Builder setParallelism(int value) {
            if (value == 0) {
                parallelismPolicy = ParallelismPolicy.AUTO;
            } else {
                classificationParallelism = value;
                recognitionParallelism = value;
                parallelismPolicy = ParallelismPolicy.MANUAL;
            }
            return this;
        }

        /** Sets the maximum input side used by dynamic DET models before 32-pixel alignment. */
        public Builder setDetectionMaximumSideLength(int value) {
            detectionMaximumSideLength = value;
            return this;
        }

        public PaddleOcrOptions build() {
            return new PaddleOcrOptions(detectionBitmapThreshold, detectionBoxThreshold,
                    detectionUnclipRatio, detectionDilation, maxDetectionCandidates,
                    classifierThreshold, readingOrder, classificationParallelism,
                    recognitionParallelism, parallelismPolicy,
                    detectionMaximumSideLength);
        }
    }

    ParallelismPlan parallelismPlan(int lineCount) {
        int processors = Math.max(1, Runtime.getRuntime().availableProcessors());
        return parallelismPolicy == ParallelismPolicy.AUTO
                ? ParallelismPlan.automatic(processors, lineCount)
                : ParallelismPlan.manual(processors, lineCount,
                        classificationParallelism, recognitionParallelism);
    }
}
