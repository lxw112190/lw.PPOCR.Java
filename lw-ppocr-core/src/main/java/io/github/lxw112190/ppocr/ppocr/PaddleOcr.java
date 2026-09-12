package io.github.lxw112190.ppocr.ppocr;

import io.github.lxw112190.ppocr.image.BgrImage;
import io.github.lxw112190.ppocr.image.BgrTransforms;
import io.github.lxw112190.ppocr.model.OcrErrorCode;
import io.github.lxw112190.ppocr.model.OcrException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

/** Fixed-shape pure-Java OCR pipeline over decoded BGR images. */
public final class PaddleOcr implements AutoCloseable {
    private final PaddleOcrDetector detector;
    private final PaddleOcrClassifier classifier;
    private final PaddleOcrRecognizer recognizer;
    private final PerspectiveCrop.Workspace cropper;
    private final PaddleOcrOptions options;
    private boolean closed;

    /** Takes ownership of all supplied components; classifier may be null to disable CLS. */
    public PaddleOcr(PaddleOcrDetector detector, PaddleOcrClassifier classifier,
                     PaddleOcrRecognizer recognizer) {
        this(detector, classifier, recognizer, PaddleOcrOptions.defaults());
    }

    public PaddleOcr(PaddleOcrDetector detector, PaddleOcrClassifier classifier,
                     PaddleOcrRecognizer recognizer, float classifierThreshold,
                     int readingOrder) {
        this(detector, classifier, recognizer, PaddleOcrOptions.builder()
                .setClassifierThreshold(classifierThreshold)
                .setReadingOrder(readingOrder)
                .build());
    }

    public PaddleOcr(PaddleOcrDetector detector, PaddleOcrClassifier classifier,
                     PaddleOcrRecognizer recognizer, PaddleOcrOptions options) {
        if (detector == null || recognizer == null || options == null) {
            throw new OcrException(OcrErrorCode.INVALID_ARGUMENT, "OCR pipeline options are invalid");
        }
        this.detector = detector;
        this.classifier = classifier;
        this.recognizer = recognizer;
        this.cropper = new PerspectiveCrop.Workspace();
        this.options = options;
    }

    public static PaddleOcr load(Path detectorPath, Path classifierPath,
                                 Path recognizerPath, Path dictionaryPath) {
        return load(detectorPath, classifierPath, recognizerPath, dictionaryPath,
                PaddleOcrOptions.defaults());
    }

    public static PaddleOcr load(Path detectorPath, Path classifierPath,
                                 Path recognizerPath, Path dictionaryPath,
                                 PaddleOcrOptions options) {
        PaddleOcrDetector detector = PaddleOcrDetector.load(detectorPath);
        PaddleOcrClassifier classifier = null;
        PaddleOcrRecognizer recognizer = null;
        try {
            if (classifierPath != null) classifier = PaddleOcrClassifier.load(classifierPath);
            recognizer = PaddleOcrRecognizer.load(recognizerPath, dictionaryPath);
            return new PaddleOcr(detector, classifier, recognizer, options);
        } catch (RuntimeException e) {
            if (recognizer != null) recognizer.close();
            if (classifier != null) classifier.close();
            detector.close();
            throw e;
        }
    }

    public OcrResult recognize(BgrImage source) {
        ensureOpen();
        List<DetectionBox> boxes = detector.detect(source,
                options.getDetectionBitmapThreshold(), options.getDetectionBoxThreshold(),
                options.getDetectionUnclipRatio(), options.isDetectionDilation(),
                options.getMaxDetectionCandidates());
        List<OcrLineResult> lines = new ArrayList<OcrLineResult>(boxes.size());
        for (DetectionBox box : boxes) {
            BgrImage crop = cropper.crop(source, box);
            ClsClassificationResult classification = null;
            boolean rotated = false;
            if (classifier != null) {
                classification = classifier.classify(crop);
                if (classification.requiresRotation(options.getClassifierThreshold())) {
                    crop = BgrTransforms.rotate180(crop);
                    rotated = true;
                }
            }
            RecRecognitionResult recognition = recognizer.recognize(crop);
            lines.add(new OcrLineResult(box, recognition.getText(), recognition.getScore(),
                    classification, rotated));
        }
        return new OcrResult(lines).sorted(options.getReadingOrder());
    }

    @Override
    public void close() {
        if (!closed) {
            closed = true;
            recognizer.close();
            if (classifier != null) classifier.close();
            detector.close();
        }
    }

    private void ensureOpen() {
        if (closed) throw new OcrException(OcrErrorCode.INVALID_ARGUMENT, "OCR pipeline is closed");
    }
}
