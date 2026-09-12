package io.github.lxw112190.ppocr.ppocr;

import io.github.lxw112190.ppocr.model.OcrErrorCode;
import io.github.lxw112190.ppocr.model.OcrException;

/** CLS argmax and rotation policy matching lw.PPOCR.C. */
public final class ClsPostprocess {
    private ClsPostprocess() { }

    public static ClsClassificationResult decode(float[] probabilities, int resizedWidth) {
        if (probabilities == null || probabilities.length != 2 || resizedWidth <= 0 ||
                !Float.isFinite(probabilities[0]) || !Float.isFinite(probabilities[1])) {
            throw new OcrException(OcrErrorCode.INVALID_ARGUMENT, "CLS output or resized width is invalid");
        }
        int label = probabilities[1] > probabilities[0] ? 1 : 0;
        return new ClsClassificationResult(label, probabilities[label], label == 0 ? 0 : 180, resizedWidth);
    }
}
