package io.github.lxw112190.ppocr.ppocr;

import io.github.lxw112190.ppocr.model.OcrErrorCode;
import io.github.lxw112190.ppocr.model.OcrException;

/** CLS argmax and rotation policy matching lw.PPOCR.C. */
public final class ClsPostprocess {
    private ClsPostprocess() { }

    public static ClsClassificationResult decode(float[] probabilities, int resizedWidth) {
        return decode(probabilities, 0, 2, resizedWidth);
    }

    public static ClsClassificationResult decode(float[] probabilities, int offset,
                                                 int length, int resizedWidth) {
        if (probabilities == null || offset < 0 || length < 2
                || offset > probabilities.length || length > probabilities.length - offset
                || resizedWidth <= 0 || !Float.isFinite(probabilities[offset])
                || !Float.isFinite(probabilities[offset + 1])) {
            throw new OcrException(OcrErrorCode.INVALID_ARGUMENT, "CLS output or resized width is invalid");
        }
        int label = probabilities[offset + 1] > probabilities[offset] ? 1 : 0;
        return new ClsClassificationResult(label, probabilities[offset + label],
                label == 0 ? 0 : 180, resizedWidth);
    }
}
