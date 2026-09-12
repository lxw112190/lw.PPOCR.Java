package io.github.lxw112190.ppocr.benchmark;

import io.github.lxw112190.ppocr.image.BgrImage;
import io.github.lxw112190.ppocr.image.BgrTransforms;
import io.github.lxw112190.ppocr.imageio.ImageIoLoader;
import io.github.lxw112190.ppocr.kernels.KernelBackend;
import io.github.lxw112190.ppocr.kernels.ScalarBackend;
import io.github.lxw112190.ppocr.model.LwmLoader;
import io.github.lxw112190.ppocr.model.LwmModel;
import io.github.lxw112190.ppocr.model.OperatorType;
import io.github.lxw112190.ppocr.ppocr.ClsClassificationResult;
import io.github.lxw112190.ppocr.ppocr.DetectionBox;
import io.github.lxw112190.ppocr.ppocr.OcrLineResult;
import io.github.lxw112190.ppocr.ppocr.OcrResult;
import io.github.lxw112190.ppocr.ppocr.PaddleOcrClassifier;
import io.github.lxw112190.ppocr.ppocr.PaddleOcrDetector;
import io.github.lxw112190.ppocr.ppocr.PaddleOcrDictionary;
import io.github.lxw112190.ppocr.ppocr.PaddleOcrOptions;
import io.github.lxw112190.ppocr.ppocr.PaddleOcrRecognizer;
import io.github.lxw112190.ppocr.ppocr.PerspectiveCrop;
import io.github.lxw112190.ppocr.ppocr.RecRecognitionResult;
import io.github.lxw112190.ppocr.runtime.InferenceProfiler;
import java.io.IOException;
import java.io.InputStream;
import java.lang.management.GarbageCollectorMXBean;
import java.lang.management.ManagementFactory;
import java.lang.management.MemoryMXBean;
import java.lang.management.MemoryUsage;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Locale;

/** CI-friendly end-to-end OCR benchmark over the committed Tiny Golden assets. */
public final class FullOcrPerformanceMain {
    private static final String DET_MODEL = "/golden/det/det.lwm";
    private static final String CLS_MODEL = "/golden/cls/cls.lwm";
    private static final String REC_MODEL = "/golden/rec/rec.lwm";
    private static final String DICTIONARY = "/golden/rec/ppocr_keys.txt";
    private static final String IMAGE = "/golden/ocr/sample.jpg";
    private static final int DEFAULT_WARMUP = 1;
    private static final int DEFAULT_ITERATIONS = 5;
    private static final int DEFAULT_DETECTOR_LIMIT = 960;

    private FullOcrPerformanceMain() { }

    public static void main(String[] args) throws Exception {
        int warmup = args.length > 0 ? positive(args[0], "warmup") : DEFAULT_WARMUP;
        int iterations = args.length > 1 ? positive(args[1], "iterations") : DEFAULT_ITERATIONS;
        int detectorLimit = args.length > 2
                ? positive(args[2], "detector limit") : DEFAULT_DETECTOR_LIMIT;
        String backendName = args.length > 3 ? args[3] : "scalar";
        if (detectorLimit < 32) throw new IllegalArgumentException("detector limit must be at least 32");
        KernelBackend backend = createBackend(backendName);

        BgrImage image = loadImage();
        MemoryMXBean memory = ManagementFactory.getMemoryMXBean();
        long heapBefore = usedHeap(memory);
        long loadStart = System.nanoTime();
        try (ProfiledPipeline pipeline = loadPipeline(detectorLimit, backend)) {
            long modelLoadNanos = System.nanoTime() - loadStart;
            long heapAfterLoad = usedHeap(memory);
            StageSample cold = pipeline.recognize(image);
            long peakHeap = Math.max(heapAfterLoad, usedHeap(memory));
            for (int i = 1; i < warmup; i++) {
                pipeline.recognize(image);
                peakHeap = Math.max(peakHeap, usedHeap(memory));
            }

            long gcCountBefore = gcCount();
            long gcTimeBefore = gcTimeMillis();
            long[] total = new long[iterations];
            long[] detection = new long[iterations];
            long[] crop = new long[iterations];
            long[] classification = new long[iterations];
            long[] rotation = new long[iterations];
            long[] recognition = new long[iterations];
            long[] sorting = new long[iterations];
            int lineCount = -1;
            InferenceProfiler.Profile operatorProfile;
            try (InferenceProfiler profiler = InferenceProfiler.start()) {
                for (int i = 0; i < iterations; i++) {
                    StageSample sample = pipeline.recognize(image);
                    total[i] = sample.totalNanos;
                    detection[i] = sample.detectionNanos;
                    crop[i] = sample.cropNanos;
                    classification[i] = sample.classificationNanos;
                    rotation[i] = sample.rotationNanos;
                    recognition[i] = sample.recognitionNanos;
                    sorting[i] = sample.sortingNanos;
                    lineCount = sample.lines;
                    peakHeap = Math.max(peakHeap, usedHeap(memory));
                }
                operatorProfile = profiler.snapshot();
            }
            long gcCountDelta = nonNegativeDelta(gcCount(), gcCountBefore);
            long gcTimeDelta = nonNegativeDelta(gcTimeMillis(), gcTimeBefore);
            long heapAfter = usedHeap(memory);
            long peakDelta = Math.max(0L, peakHeap - heapBefore);
            Arrays.sort(total);
            String benchmark = detectorLimit == DEFAULT_DETECTOR_LIMIT
                    ? "full-ocr-default" : "full-ocr-det" + detectorLimit;
            System.out.printf(Locale.ROOT,
                    "{\"benchmark\":\"%s\",\"backend\":\"%s\",\"image_width\":%d,"
                            + "\"image_height\":%d,\"detector_limit_side\":%d,"
                            + "\"lines\":%d,\"warmup\":%d,\"iterations\":%d,"
                            + "\"model_load_ms\":%.3f,\"cold_ms\":%.3f,"
                            + "\"mean_ms\":%.3f,\"median_ms\":%.3f,\"p95_ms\":%.3f,"
                            + "\"detection_mean_ms\":%.3f,\"crop_mean_ms\":%.3f,"
                            + "\"classification_mean_ms\":%.3f,\"rotation_mean_ms\":%.3f,"
                            + "\"recognition_mean_ms\":%.3f,\"sorting_mean_ms\":%.3f,"
                            + "\"heap_before_bytes\":%d,\"heap_after_load_bytes\":%d,"
                            + "\"heap_after_bytes\":%d,\"peak_heap_bytes\":%d,"
                            + "\"peak_heap_delta_bytes\":%d,\"gc_count_delta\":%d,"
                            + "\"gc_time_ms_delta\":%d,\"operators\":%s}%n",
                    benchmark, backendName, image.width(), image.height(), detectorLimit, lineCount,
                    warmup, iterations, milliseconds(modelLoadNanos), milliseconds(cold.totalNanos),
                    milliseconds(mean(total)), milliseconds(percentile(total, 0.50)),
                    milliseconds(percentile(total, 0.95)), milliseconds(mean(detection)),
                    milliseconds(mean(crop)), milliseconds(mean(classification)),
                    milliseconds(mean(rotation)), milliseconds(mean(recognition)),
                    milliseconds(mean(sorting)), heapBefore, heapAfterLoad, heapAfter,
                    peakHeap, peakDelta, gcCountDelta, gcTimeDelta,
                    operatorJson(operatorProfile, iterations));
        }
    }

    private static KernelBackend createBackend(String name) {
        if ("scalar".equals(name)) return new ScalarBackend();
        if (!"vector".equals(name)) throw new IllegalArgumentException("backend must be scalar or vector");
        try {
            return (KernelBackend) Class.forName("io.github.lxw112190.ppocr.vector.VectorBackend")
                    .getDeclaredConstructor().newInstance();
        } catch (ReflectiveOperationException e) {
            throw new IllegalStateException("Vector backend is unavailable", e);
        }
    }

    private static ProfiledPipeline loadPipeline(int detectorLimit, KernelBackend backend) throws IOException {
        LwmModel detectorModel = loadModel(DET_MODEL);
        PaddleOcrDetector detector = null;
        LwmModel classifierModel = null;
        PaddleOcrClassifier classifier = null;
        LwmModel recognizerModel = null;
        PaddleOcrDictionary dictionary = null;
        PaddleOcrRecognizer recognizer = null;
        try {
            detector = new PaddleOcrDetector(detectorModel, detectorLimit, backend);
            classifierModel = loadModel(CLS_MODEL);
            classifier = new PaddleOcrClassifier(classifierModel, backend);
            recognizerModel = loadModel(REC_MODEL);
            dictionary = loadDictionary();
            recognizer = new PaddleOcrRecognizer(recognizerModel, dictionary, backend);
            return new ProfiledPipeline(detector, classifier, recognizer);
        } catch (RuntimeException e) {
            if (recognizer != null) recognizer.close();
            else {
                if (dictionary != null) dictionary.close();
                if (recognizerModel != null) recognizerModel.close();
            }
            if (classifier != null) classifier.close();
            else if (classifierModel != null) classifierModel.close();
            if (detector != null) detector.close();
            else detectorModel.close();
            throw e;
        }
    }

    private static LwmModel loadModel(String resource) throws IOException {
        try (InputStream input = resource(resource)) {
            return LwmLoader.load(input);
        }
    }

    private static PaddleOcrDictionary loadDictionary() throws IOException {
        try (InputStream input = resource(DICTIONARY)) {
            return PaddleOcrDictionary.load(input);
        }
    }

    private static BgrImage loadImage() throws IOException {
        try (InputStream input = resource(IMAGE)) {
            return ImageIoLoader.load(input);
        }
    }

    private static InputStream resource(String name) throws IOException {
        InputStream input = FullOcrPerformanceMain.class.getResourceAsStream(name);
        if (input == null) throw new IOException("missing benchmark resource: " + name);
        return input;
    }

    private static long usedHeap(MemoryMXBean memory) {
        MemoryUsage usage = memory.getHeapMemoryUsage();
        return usage == null ? 0L : usage.getUsed();
    }

    private static long gcCount() {
        long total = 0L;
        for (GarbageCollectorMXBean collector : ManagementFactory.getGarbageCollectorMXBeans()) {
            long count = collector.getCollectionCount();
            if (count >= 0L) total += count;
        }
        return total;
    }

    private static long gcTimeMillis() {
        long total = 0L;
        for (GarbageCollectorMXBean collector : ManagementFactory.getGarbageCollectorMXBeans()) {
            long time = collector.getCollectionTime();
            if (time >= 0L) total += time;
        }
        return total;
    }

    private static long nonNegativeDelta(long after, long before) {
        return Math.max(0L, after - before);
    }

    private static int positive(String value, String name) {
        try {
            int parsed = Integer.parseInt(value);
            if (parsed > 0) return parsed;
        } catch (NumberFormatException ignored) { }
        throw new IllegalArgumentException(name + " must be positive");
    }

    private static double mean(long[] values) {
        long total = 0L;
        for (long value : values) total += value;
        return (double) total / values.length;
    }

    private static long percentile(long[] sortedValues, double fraction) {
        return sortedValues[(int) Math.min(sortedValues.length - 1,
                Math.ceil(sortedValues.length * fraction) - 1)];
    }

    private static double milliseconds(double nanos) {
        return nanos / 1_000_000.0;
    }

    private static String operatorJson(InferenceProfiler.Profile profile, int iterations) {
        StringBuilder json = new StringBuilder("{");
        boolean first = true;
        for (OperatorType operator : OperatorType.values()) {
            long calls = profile.getInvocations(operator);
            if (calls == 0L) continue;
            if (!first) json.append(',');
            first = false;
            json.append('\"').append(operator.name().toLowerCase(Locale.ROOT)).append("\":{")
                    .append("\"calls\":").append(calls)
                    .append(",\"mean_per_ocr_ms\":")
                    .append(String.format(Locale.ROOT, "%.3f",
                            milliseconds((double) profile.getElapsedNanos(operator) / iterations)))
                    .append('}');
        }
        return json.append('}').toString();
    }

    /** Benchmark-only copy of the public pipeline orchestration with stage boundaries. */
    private static final class ProfiledPipeline implements AutoCloseable {
        private final PaddleOcrDetector detector;
        private final PaddleOcrClassifier classifier;
        private final PaddleOcrRecognizer recognizer;
        private final PerspectiveCrop.Workspace cropper = new PerspectiveCrop.Workspace();
        private final PaddleOcrOptions options = PaddleOcrOptions.defaults();

        private ProfiledPipeline(PaddleOcrDetector detector, PaddleOcrClassifier classifier,
                                 PaddleOcrRecognizer recognizer) {
            this.detector = detector;
            this.classifier = classifier;
            this.recognizer = recognizer;
        }

        private StageSample recognize(BgrImage source) {
            StageSample sample = new StageSample();
            long totalStart = System.nanoTime();
            long start = System.nanoTime();
            List<DetectionBox> boxes = detector.detect(source,
                    options.getDetectionBitmapThreshold(), options.getDetectionBoxThreshold(),
                    options.getDetectionUnclipRatio(), options.isDetectionDilation(),
                    options.getMaxDetectionCandidates());
            sample.detectionNanos = System.nanoTime() - start;

            List<OcrLineResult> lines = new ArrayList<OcrLineResult>(boxes.size());
            for (DetectionBox box : boxes) {
                start = System.nanoTime();
                BgrImage crop = cropper.crop(source, box);
                sample.cropNanos += System.nanoTime() - start;

                start = System.nanoTime();
                ClsClassificationResult classification = classifier.classify(crop);
                sample.classificationNanos += System.nanoTime() - start;
                boolean rotated = false;
                if (classification.requiresRotation(options.getClassifierThreshold())) {
                    start = System.nanoTime();
                    crop = BgrTransforms.rotate180(crop);
                    sample.rotationNanos += System.nanoTime() - start;
                    rotated = true;
                }

                start = System.nanoTime();
                RecRecognitionResult recognition = recognizer.recognize(crop);
                sample.recognitionNanos += System.nanoTime() - start;
                lines.add(new OcrLineResult(box, recognition.getText(), recognition.getScore(),
                        classification, rotated));
            }
            start = System.nanoTime();
            OcrResult result = new OcrResult(lines).sorted(options.getReadingOrder());
            sample.sortingNanos = System.nanoTime() - start;
            sample.lines = result.getLines().size();
            sample.totalNanos = System.nanoTime() - totalStart;
            return sample;
        }

        @Override
        public void close() {
            recognizer.close();
            classifier.close();
            detector.close();
        }
    }

    private static final class StageSample {
        private long totalNanos;
        private long detectionNanos;
        private long cropNanos;
        private long classificationNanos;
        private long rotationNanos;
        private long recognitionNanos;
        private long sortingNanos;
        private int lines;
    }
}
