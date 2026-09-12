package io.github.lxw112190.ppocr.ppocr;

import io.github.lxw112190.ppocr.image.BgrImage;
import io.github.lxw112190.ppocr.image.BgrTransforms;
import io.github.lxw112190.ppocr.kernels.KernelBackend;
import io.github.lxw112190.ppocr.kernels.ScalarBackend;
import io.github.lxw112190.ppocr.model.OcrErrorCode;
import io.github.lxw112190.ppocr.model.OcrException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

/** Pure-Java DET/CLS/REC OCR pipeline over decoded BGR images. */
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
                PaddleOcrOptions.defaults(), new ScalarBackend());
    }

    public static PaddleOcr load(Path detectorPath, Path classifierPath,
                                 Path recognizerPath, Path dictionaryPath,
                                 PaddleOcrOptions options) {
        return load(detectorPath, classifierPath, recognizerPath, dictionaryPath,
                options, new ScalarBackend());
    }

    /** Loads a complete OCR pipeline using one shared stateless kernel backend. */
    public static PaddleOcr load(Path detectorPath, Path classifierPath,
                                 Path recognizerPath, Path dictionaryPath,
                                 PaddleOcrOptions options, KernelBackend backend) {
        if (options == null || backend == null) {
            throw new OcrException(OcrErrorCode.INVALID_ARGUMENT,
                    "OCR pipeline options and backend are required");
        }
        PaddleOcrDetector detector = PaddleOcrDetector.load(detectorPath,
                options.getDetectionMaximumSideLength(), backend);
        PaddleOcrClassifier classifier = null;
        PaddleOcrRecognizer recognizer = null;
        try {
            if (classifierPath != null) classifier = PaddleOcrClassifier.load(classifierPath, backend);
            recognizer = PaddleOcrRecognizer.load(recognizerPath, dictionaryPath, backend);
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
        List<BgrImage> crops = new ArrayList<BgrImage>(boxes.size());
        List<Boolean> rotations = new ArrayList<Boolean>(boxes.size());
        for (int i = 0; i < boxes.size(); i++) {
            crops.add(cropper.crop(source, boxes.get(i), i));
        }
        List<ClsClassificationResult> classifications;
        if (classifier == null) {
            classifications = new ArrayList<ClsClassificationResult>(boxes.size());
            for (int i = 0; i < boxes.size(); i++) classifications.add(null);
        } else {
            classifications = classifier.classifyAll(crops,
                    options.getClassificationParallelism());
        }
        for (int i = 0; i < boxes.size(); i++) {
            BgrImage crop = crops.get(i);
            ClsClassificationResult classification = classifications.get(i);
            boolean rotated = false;
            if (classification != null &&
                    classification.requiresRotation(options.getClassifierThreshold())) {
                crop = BgrTransforms.rotate180(crop);
                crops.set(i, crop);
                rotated = true;
            }
            rotations.add(rotated);
        }
        List<RecRecognitionResult> recognitions = recognizer.recognizeAll(
                crops, options.getRecognitionParallelism());
        for (int i = 0; i < boxes.size(); i++) {
            RecRecognitionResult recognition = recognitions.get(i);
            ClsClassificationResult classification = classifications.get(i);
            boolean rotated = rotations.get(i);
            DetectionBox box = boxes.get(i);
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
