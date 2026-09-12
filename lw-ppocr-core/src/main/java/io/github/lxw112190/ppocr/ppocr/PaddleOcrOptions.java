package io.github.lxw112190.ppocr.ppocr;

import io.github.lxw112190.ppocr.model.OcrErrorCode;
import io.github.lxw112190.ppocr.model.OcrException;

/** Immutable thresholds and policies for the fixed-shape OCR pipeline. */
public final class PaddleOcrOptions {
    private final float detectionBitmapThreshold;
    private final float detectionBoxThreshold;
    private final float detectionUnclipRatio;
    private final boolean detectionDilation;
    private final int maxDetectionCandidates;
    private final float classifierThreshold;
    private final int readingOrder;
    private final int recognitionParallelism;

    private PaddleOcrOptions(float detectionBitmapThreshold, float detectionBoxThreshold,
                             float detectionUnclipRatio, boolean detectionDilation,
                             int maxDetectionCandidates, float classifierThreshold,
                             int readingOrder, int recognitionParallelism) {
        if (!Float.isFinite(detectionBitmapThreshold) || detectionBitmapThreshold < 0.0f ||
                detectionBitmapThreshold > 1.0f || !Float.isFinite(detectionBoxThreshold) ||
                detectionBoxThreshold < 0.0f || detectionBoxThreshold > 1.0f ||
                !Float.isFinite(detectionUnclipRatio) || detectionUnclipRatio <= 0.0f ||
                detectionUnclipRatio > 10.0f ||
                maxDetectionCandidates <= 0 || !Float.isFinite(classifierThreshold) ||
                classifierThreshold < 0.0f || classifierThreshold > 1.0f ||
                readingOrder < ReadingOrder.HORIZONTAL_LTR || readingOrder > ReadingOrder.VERTICAL_LTR ||
                recognitionParallelism <= 0 || recognitionParallelism > 64) {
            throw new OcrException(OcrErrorCode.INVALID_ARGUMENT, "OCR options are invalid");
        }
        this.detectionBitmapThreshold = detectionBitmapThreshold;
        this.detectionBoxThreshold = detectionBoxThreshold;
        this.detectionUnclipRatio = detectionUnclipRatio;
        this.detectionDilation = detectionDilation;
        this.maxDetectionCandidates = maxDetectionCandidates;
        this.classifierThreshold = classifierThreshold;
        this.readingOrder = readingOrder;
        this.recognitionParallelism = recognitionParallelism;
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
    public int getRecognitionParallelism() { return recognitionParallelism; }

    public static final class Builder {
        private float detectionBitmapThreshold = 0.3f;
        private float detectionBoxThreshold = 0.6f;
        private float detectionUnclipRatio = 1.6f;
        private boolean detectionDilation;
        private int maxDetectionCandidates = 1000;
        private float classifierThreshold = 0.9f;
        private int readingOrder = ReadingOrder.HORIZONTAL_LTR;
        private int recognitionParallelism = 1;

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

        /** Sets the maximum number of REC width groups evaluated concurrently. */
        public Builder setRecognitionParallelism(int value) {
            recognitionParallelism = value;
            return this;
        }

        public PaddleOcrOptions build() {
            return new PaddleOcrOptions(detectionBitmapThreshold, detectionBoxThreshold,
                    detectionUnclipRatio, detectionDilation, maxDetectionCandidates,
                    classifierThreshold, readingOrder, recognitionParallelism);
        }
    }
}
