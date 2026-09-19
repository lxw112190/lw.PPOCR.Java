package io.github.lxw112190.ppocr.ppocr;

import io.github.lxw112190.ppocr.image.BgrImage;
import io.github.lxw112190.ppocr.image.BgrTransforms;
import io.github.lxw112190.ppocr.kernels.KernelBackend;
import io.github.lxw112190.ppocr.kernels.ScalarBackend;
import io.github.lxw112190.ppocr.model.OcrErrorCode;
import io.github.lxw112190.ppocr.model.OcrException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/** Pure-Java DET/CLS/REC OCR pipeline over decoded BGR images. */
public final class PaddleOcr implements AutoCloseable {
    private final PaddleOcrDetector detector;
    private final PaddleOcrClassifier classifier;
    private final PaddleOcrRecognizer recognizer;
    private final PerspectiveCrop.Workspace cropper;
    private final PaddleOcrOptions options;
    private final ArrayList<OcrLineResult> lineStaging;
    private final ArrayList<BgrImage> cropStaging;
    private ClsClassificationResult[] classificationStaging;
    private RecRecognitionResult[] recognitionStaging;
    private boolean[] rotationStaging;
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
        this.lineStaging = new ArrayList<OcrLineResult>();
        this.cropStaging = new ArrayList<BgrImage>();
        this.classificationStaging = new ClsClassificationResult[0];
        this.recognitionStaging = new RecRecognitionResult[0];
        this.rotationStaging = new boolean[0];
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
        ParallelismPlan parallelism = options.parallelismPlan(boxes.size());
        prepareStaging(boxes.size());
        try {
            cropper.cropAll(source, boxes, cropStaging);
            if (classifier == null) {
                Arrays.fill(classificationStaging, 0, boxes.size(), null);
            } else {
                classifier.classifyAllInto(cropStaging,
                        parallelism.getClassifierWorkers(), classificationStaging);
            }
            for (int i = 0; i < boxes.size(); i++) {
                BgrImage crop = cropStaging.get(i);
                ClsClassificationResult classification = classificationStaging[i];
                boolean rotated = false;
                if (classification != null &&
                        classification.requiresRotation(options.getClassifierThreshold())) {
                    crop = BgrTransforms.rotate180(crop);
                    cropStaging.set(i, crop);
                    rotated = true;
                }
                rotationStaging[i] = rotated;
            }
            recognizer.recognizeAllInto(cropStaging,
                    parallelism.getRecognizerWorkers(), recognitionStaging);
            for (int i = 0; i < boxes.size(); i++) {
                RecRecognitionResult recognition = recognitionStaging[i];
                ClsClassificationResult classification = classificationStaging[i];
                DetectionBox box = boxes.get(i);
                lineStaging.add(new OcrLineResult(box, recognition.getText(), recognition.getScore(),
                        classification, rotationStaging[i]));
            }
            return new OcrResult(lineStaging).sorted(options.getReadingOrder());
        } finally {
            clearStaging(boxes.size());
        }
    }

    private void prepareStaging(int size) {
        lineStaging.clear();
        cropStaging.clear();
        lineStaging.ensureCapacity(size);
        cropStaging.ensureCapacity(size);
        if (classificationStaging.length < size) {
            classificationStaging = new ClsClassificationResult[size];
        }
        if (recognitionStaging.length < size) {
            recognitionStaging = new RecRecognitionResult[size];
        }
        if (rotationStaging.length < size) rotationStaging = new boolean[size];
    }

    private void clearStaging(int size) {
        lineStaging.clear();
        cropStaging.clear();
        Arrays.fill(classificationStaging, 0, size, null);
        Arrays.fill(recognitionStaging, 0, size, null);
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
