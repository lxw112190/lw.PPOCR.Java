package io.github.lxw112190.ppocr.imageio;

import io.github.lxw112190.ppocr.image.BgrImage;
import io.github.lxw112190.ppocr.model.OcrErrorCode;
import io.github.lxw112190.ppocr.model.OcrException;
import io.github.lxw112190.ppocr.ppocr.OcrResult;
import io.github.lxw112190.ppocr.ppocr.PaddleOcr;
import java.awt.image.BufferedImage;
import java.io.InputStream;
import java.nio.file.Path;

/** Convenience bridge from standard Java images to the core OCR pipeline. */
public final class PaddleOcrImageIo {
    private PaddleOcrImageIo() { }

    public static OcrResult recognize(PaddleOcr ocr, Path path) {
        requirePipeline(ocr);
        return ocr.recognize(ImageIoLoader.load(path));
    }

    public static OcrResult recognize(PaddleOcr ocr, InputStream input) {
        requirePipeline(ocr);
        return ocr.recognize(ImageIoLoader.load(input));
    }

    public static OcrResult recognize(PaddleOcr ocr, BufferedImage image) {
        requirePipeline(ocr);
        BgrImage bgr = BufferedImageAdapter.toBgr(image);
        return ocr.recognize(bgr);
    }

    private static void requirePipeline(PaddleOcr ocr) {
        if (ocr == null) throw new OcrException(OcrErrorCode.INVALID_ARGUMENT, "OCR pipeline is required");
    }
}
