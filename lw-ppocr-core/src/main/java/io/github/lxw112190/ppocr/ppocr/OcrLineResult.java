package io.github.lxw112190.ppocr.ppocr;

import io.github.lxw112190.ppocr.model.OcrErrorCode;
import io.github.lxw112190.ppocr.model.OcrException;

/** One detected text line and the optional CLS/REC metadata associated with it. */
public final class OcrLineResult {
    private final DetectionBox box;
    private final String text;
    private final float recognitionScore;
    private final ClsClassificationResult classification;
    private final boolean rotated;

    public OcrLineResult(DetectionBox box, String text, float recognitionScore,
                         ClsClassificationResult classification, boolean rotated) {
        if (box == null || text == null || !Float.isFinite(recognitionScore)) {
            throw new OcrException(OcrErrorCode.INVALID_ARGUMENT, "OCR line result is invalid");
        }
        this.box = box;
        this.text = text;
        this.recognitionScore = recognitionScore;
        this.classification = classification;
        this.rotated = rotated;
    }

    public DetectionBox getBox() { return box; }
    public String getText() { return text; }
    public float getRecognitionScore() { return recognitionScore; }
    public ClsClassificationResult getClassification() { return classification; }
    public boolean isRotated() { return rotated; }
}
