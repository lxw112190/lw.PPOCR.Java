package io.github.lxw112190.ppocr.ppocr;

import io.github.lxw112190.ppocr.image.BgrImage;
import io.github.lxw112190.ppocr.image.BgrTransforms;
import io.github.lxw112190.ppocr.kernels.KernelBackend;
import io.github.lxw112190.ppocr.kernels.ScalarBackend;
import io.github.lxw112190.ppocr.model.OcrErrorCode;
import io.github.lxw112190.ppocr.model.OcrException;
import io.github.lxw112190.ppocr.runtime.WorkspaceDiagnostics;
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
    private final ArrayList<DetectionBox> detectionStaging;
    private final ArrayList<OcrLineResult> lineStaging;
    private final ArrayList<BgrImage> cropStaging;
    private ClsClassificationResult[] classificationStaging;
    private String[] recognitionTexts;
    private float[] recognitionScores;
    private int[] recognitionEmittedCounts;
    private int[] recognitionResizedWidths;
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
        this.detectionStaging = new ArrayList<DetectionBox>();
        this.lineStaging = new ArrayList<OcrLineResult>();
        this.cropStaging = new ArrayList<BgrImage>();
        this.classificationStaging = new ClsClassificationResult[0];
        this.recognitionTexts = new String[0];
        this.recognitionScores = new float[0];
        this.recognitionEmittedCounts = new int[0];
        this.recognitionResizedWidths = new int[0];
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
        detectionStaging.clear();
        detector.detectInto(source,
                options.getDetectionBitmapThreshold(), options.getDetectionBoxThreshold(),
                options.getDetectionUnclipRatio(), options.isDetectionDilation(),
                options.getMaxDetectionCandidates(), detectionStaging);
        List<DetectionBox> boxes = detectionStaging;
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
                    // The crop arena is owned by this pipeline and CLS has finished
                    // reading it, so rotate the slot in place instead of allocating
                    // another full crop-sized byte array.
                    BgrTransforms.rotate180InPlace(crop);
                    rotated = true;
                }
                rotationStaging[i] = rotated;
            }
            recognizer.recognizeAllInto(cropStaging,
                    parallelism.getRecognizerWorkers(), recognitionTexts,
                    recognitionScores, recognitionEmittedCounts, recognitionResizedWidths);
            for (int i = 0; i < boxes.size(); i++) {
                ClsClassificationResult classification = classificationStaging[i];
                DetectionBox box = boxes.get(i);
                lineStaging.add(new OcrLineResult(box, recognitionTexts[i], recognitionScores[i],
                        classification, rotationStaging[i]));
            }
            ReadingOrder.sortLinesInPlace(lineStaging, options.getReadingOrder());
            return new OcrResult(lineStaging);
        } finally {
            clearStaging(boxes.size());
        }
    }

    /** Returns the combined workspace measurements of the prepared DET/CLS/REC sessions. */
    public WorkspaceDiagnostics workspaceDiagnostics() {
        ensureOpen();
        return WorkspaceDiagnostics.aggregate(
                detector.workspaceDiagnostics(),
                classifier == null ? null : classifier.workspaceDiagnostics(),
                recognizer.workspaceDiagnostics());
    }

    /** Returns the capacity of reusable DB postprocess buffers in bytes. */
    public long dbScratchBytes() {
        ensureOpen();
        return detector.dbScratchBytes();
    }

    /** Returns the capacity of the reusable perspective-crop workspace in bytes. */
    public long cropWorkspaceBytes() {
        ensureOpen();
        return cropper.workspaceBytes();
    }

    /** Returns canonical FP32 constants materialized by the three OCR models. */
    public long decodedConstantBytes() {
        ensureOpen();
        return detector.decodedConstantBytes()
                + (classifier == null ? 0L : classifier.decodedConstantBytes())
                + recognizer.decodedConstantBytes();
    }

    /** Returns prepared REC projection matrices retained by the pipeline. */
    public long packedWeightBytes() {
        ensureOpen();
        return recognizer.packedWeightBytes();
    }

    /** Returns prepared DET workspace bytes across currently cached shapes. */
    public long detectorWorkspaceBytes() {
        ensureOpen();
        return detector.workspaceDiagnostics().getWorkspaceBytes();
    }

    /** Returns prepared CLS workspace bytes across currently prepared workers. */
    public long classifierWorkspaceBytes() {
        ensureOpen();
        return classifier == null ? 0L : classifier.workspaceDiagnostics().getWorkspaceBytes();
    }

    /** Returns prepared REC workspace bytes across currently prepared widths. */
    public long recognizerWorkspaceBytes() {
        ensureOpen();
        return recognizer.workspaceDiagnostics().getWorkspaceBytes();
    }

    /** Returns REC bucket workspace bytes in 192/320/480/640/960 policy order. */
    public long[] recognizerWorkspaceBytesByWidth() {
        ensureOpen();
        return recognizer.workspaceBytesByWidth();
    }

    /** Returns workspace bytes for a non-bucket fallback REC session, if any. */
    public long recognizerFallbackWorkspaceBytes() {
        ensureOpen();
        return recognizer.fallbackWorkspaceBytes();
    }

    private void prepareStaging(int size) {
        lineStaging.clear();
        cropStaging.clear();
        lineStaging.ensureCapacity(size);
        cropStaging.ensureCapacity(size);
        if (classificationStaging.length < size) {
            classificationStaging = new ClsClassificationResult[size];
        }
        if (recognitionTexts.length < size) {
            recognitionTexts = new String[size];
            recognitionScores = new float[size];
            recognitionEmittedCounts = new int[size];
            recognitionResizedWidths = new int[size];
        }
        if (rotationStaging.length < size) rotationStaging = new boolean[size];
    }

    private void clearStaging(int size) {
        detectionStaging.clear();
        lineStaging.clear();
        cropStaging.clear();
        Arrays.fill(classificationStaging, 0, size, null);
        Arrays.fill(recognitionTexts, 0, size, null);
        // Primitive result buffers are overwritten for every valid crop on the
        // next invocation; clearing them only adds memory traffic.  Object
        // buffers are cleared so a large previous result is not retained.
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
