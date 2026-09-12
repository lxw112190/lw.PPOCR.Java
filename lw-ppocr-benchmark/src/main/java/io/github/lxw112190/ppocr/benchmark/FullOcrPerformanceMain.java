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
import java.lang.management.MemoryPoolMXBean;
import java.lang.management.MemoryType;
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
        int recognitionParallelism = args.length > 4
                ? positive(args[4], "recognition parallelism") : 1;
        int classificationParallelism = args.length > 5
                ? positive(args[5], "classification parallelism") : 1;
        if (detectorLimit < 32) throw new IllegalArgumentException("detector limit must be at least 32");
        KernelBackend backend = createBackend(backendName);

        BgrImage image = loadImage();
        MemoryMXBean memory = ManagementFactory.getMemoryMXBean();
        long heapBefore = stabilizedHeap(memory);
        long loadStart = System.nanoTime();
        try (ProfiledPipeline pipeline = loadPipeline(detectorLimit, backend,
                recognitionParallelism, classificationParallelism)) {
            long modelLoadNanos = System.nanoTime() - loadStart;
            long heapAfterLoad = stabilizedHeap(memory);
            StageSample cold = pipeline.recognize(image);
            for (int i = 1; i < warmup; i++) {
                pipeline.recognize(image);
            }

            resetHeapPeaks();
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
            }
            long peakHeap = Math.max(heapPeakUsed(), usedHeap(memory));
            long gcCountDelta = nonNegativeDelta(gcCount(), gcCountBefore);
            long gcTimeDelta = nonNegativeDelta(gcTimeMillis(), gcTimeBefore);
            long heapAfter = usedHeap(memory);
            long heapAfterGc = stabilizedHeap(memory);
            long peakDelta = Math.max(0L, peakHeap - heapBefore);
            long modelHeap = Math.max(0L, heapAfterLoad - heapBefore);
            long retainedHeap = Math.max(0L, heapAfterGc - heapBefore);
            long transientHeap = Math.max(0L, peakHeap - heapAfterGc);
            long heapMax = maximumHeap(memory);

            // Keep node-level atomic accounting out of the measured samples. A separate,
            // already-warmed invocation supplies diagnostics without biasing wall time or GC.
            StageSample profiledSample = pipeline.profile(image);
            if (profiledSample.lines != lineCount) {
                throw new IllegalStateException("profile invocation changed OCR line count: measured="
                        + lineCount + ", profiled=" + profiledSample.lines);
            }

            Arrays.sort(total);
            String benchmark = detectorLimit == DEFAULT_DETECTOR_LIMIT
                    ? "full-ocr-default" : "full-ocr-det" + detectorLimit;
            String profileScope = "all-threads";
            System.out.printf(Locale.ROOT,
                    "{\"schema\":3,\"benchmark\":\"%s\",\"backend\":\"%s\","
                            + "\"classification_parallelism\":%d,"
                            + "\"recognition_parallelism\":%d,"
                            + "\"operator_profile_scope\":\"%s\","
                            + "\"operator_profile_iterations\":1,\"operator_profile_ms\":%.3f,"
                            + "\"image_width\":%d,"
                            + "\"image_height\":%d,\"detector_limit_side\":%d,"
                            + "\"lines\":%d,\"warmup\":%d,\"iterations\":%d,"
                            + "\"model_load_ms\":%.3f,\"cold_ms\":%.3f,"
                            + "\"mean_ms\":%.3f,\"median_ms\":%.3f,\"p95_ms\":%.3f,"
                            + "\"detection_mean_ms\":%.3f,\"crop_mean_ms\":%.3f,"
                            + "\"classification_mean_ms\":%.3f,\"rotation_mean_ms\":%.3f,"
                            + "\"recognition_mean_ms\":%.3f,\"sorting_mean_ms\":%.3f,"
                            + "\"available_processors\":%d,\"heap_max_bytes\":%d,"
                            + "\"heap_before_bytes\":%d,\"heap_after_load_bytes\":%d,"
                            + "\"heap_after_load_gc_bytes\":%d,\"model_heap_bytes\":%d,"
                            + "\"heap_after_bytes\":%d,\"heap_before_gc_bytes\":%d,"
                            + "\"heap_after_gc_bytes\":%d,\"retained_heap_delta_bytes\":%d,"
                            + "\"peak_heap_bytes\":%d,\"peak_heap_delta_bytes\":%d,"
                            + "\"transient_heap_bytes\":%d,"
                            + "\"heap_peak_method\":\"mxbean-pool-sum\",\"gc_count_delta\":%d,"
                            + "\"gc_time_ms_delta\":%d,\"operators\":%s,"
                            + "\"stage_operators\":%s,\"stage_hot_nodes\":%s}%n",
                    benchmark, backendName, classificationParallelism,
                    recognitionParallelism, profileScope,
                    milliseconds(profiledSample.totalNanos),
                    image.width(), image.height(), detectorLimit, lineCount,
                    warmup, iterations, milliseconds(modelLoadNanos), milliseconds(cold.totalNanos),
                    milliseconds(mean(total)), milliseconds(percentile(total, 0.50)),
                    milliseconds(percentile(total, 0.95)), milliseconds(mean(detection)),
                    milliseconds(mean(crop)), milliseconds(mean(classification)),
                    milliseconds(mean(rotation)), milliseconds(mean(recognition)),
                    milliseconds(mean(sorting)), Runtime.getRuntime().availableProcessors(), heapMax,
                    heapBefore, heapAfterLoad, heapAfterLoad, modelHeap, heapAfter, heapAfter,
                    heapAfterGc, retainedHeap,
                    peakHeap, peakDelta, transientHeap, gcCountDelta, gcTimeDelta,
                    operatorJson(profiledSample.detectionProfile,
                            profiledSample.classificationProfile,
                            profiledSample.recognitionProfile),
                    stageOperatorJson(profiledSample), hotNodeStageJson(profiledSample));
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

    private static ProfiledPipeline loadPipeline(int detectorLimit, KernelBackend backend,
                                                 int recognitionParallelism,
                                                 int classificationParallelism) throws IOException {
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
            return new ProfiledPipeline(detector, classifier, recognizer,
                    recognitionParallelism, classificationParallelism);
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

    private static long maximumHeap(MemoryMXBean memory) {
        MemoryUsage usage = memory.getHeapMemoryUsage();
        return usage == null ? 0L : Math.max(0L, usage.getMax());
    }

    private static long stabilizedHeap(MemoryMXBean memory) {
        long minimum = usedHeap(memory);
        for (int attempt = 0; attempt < 3; attempt++) {
            System.gc();
            try {
                Thread.sleep(25L);
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                break;
            }
            minimum = Math.min(minimum, usedHeap(memory));
        }
        return minimum;
    }

    private static void resetHeapPeaks() {
        for (MemoryPoolMXBean pool : ManagementFactory.getMemoryPoolMXBeans()) {
            if (pool.getType() == MemoryType.HEAP) pool.resetPeakUsage();
        }
    }

    private static long heapPeakUsed() {
        long total = 0L;
        for (MemoryPoolMXBean pool : ManagementFactory.getMemoryPoolMXBeans()) {
            if (pool.getType() != MemoryType.HEAP) continue;
            MemoryUsage usage = pool.getPeakUsage();
            if (usage != null) total += Math.max(0L, usage.getUsed());
        }
        return total;
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

    private static String operatorJson(InferenceProfiler.Profile... profiles) {
        StringBuilder json = new StringBuilder("{");
        boolean first = true;
        for (OperatorType operator : OperatorType.values()) {
            long calls = 0L;
            long elapsedNanos = 0L;
            for (InferenceProfiler.Profile profile : profiles) {
                if (profile == null) continue;
                calls += profile.getInvocations(operator);
                elapsedNanos += profile.getElapsedNanos(operator);
            }
            if (calls == 0L) continue;
            if (!first) json.append(',');
            first = false;
            json.append('\"').append(operator.name().toLowerCase(Locale.ROOT)).append("\":{")
                    .append("\"calls\":").append(calls)
                    .append(",\"summed_thread_ms_per_ocr\":")
                    .append(String.format(Locale.ROOT, "%.3f",
                            milliseconds(elapsedNanos)))
                    .append('}');
        }
        return json.append('}').toString();
    }

    private static String stageOperatorJson(StageSample sample) {
        return "{\"detection\":" + operatorJson(sample.detectionProfile)
                + ",\"classification\":" + operatorJson(sample.classificationProfile)
                + ",\"recognition\":" + operatorJson(sample.recognitionProfile) + '}';
    }

    private static String hotNodeStageJson(StageSample sample) {
        return "{\"detection\":" + hotNodeJson(sample.detectionProfile, 8)
                + ",\"classification\":" + hotNodeJson(sample.classificationProfile, 8)
                + ",\"recognition\":" + hotNodeJson(sample.recognitionProfile, 12) + '}';
    }

    private static String hotNodeJson(InferenceProfiler.Profile profile, int limit) {
        StringBuilder json = new StringBuilder("[");
        List<InferenceProfiler.NodeProfile> nodes = profile.getNodes();
        int count = Math.min(limit, nodes.size());
        for (int i = 0; i < count; i++) {
            if (i != 0) json.append(',');
            InferenceProfiler.NodeProfile node = nodes.get(i);
            json.append("{\"node\":").append(node.getNodeIndex())
                    .append(",\"operator\":\"")
                    .append(node.getOperator().name().toLowerCase(Locale.ROOT))
                    .append("\",\"calls\":").append(node.getInvocations())
                    .append(",\"summed_thread_ms_per_ocr\":")
                    .append(String.format(Locale.ROOT, "%.3f",
                            milliseconds(node.getElapsedNanos())))
                    .append(",\"shape\":\"").append(jsonEscape(node.getDescription()))
                    .append("\"}");
        }
        return json.append(']').toString();
    }

    private static String jsonEscape(String value) {
        StringBuilder escaped = new StringBuilder(value.length());
        for (int i = 0; i < value.length(); i++) {
            char character = value.charAt(i);
            if (character == '\\' || character == '\"') escaped.append('\\');
            escaped.append(character);
        }
        return escaped.toString();
    }

    /** Benchmark-only copy of the public pipeline orchestration with stage boundaries. */
    private static final class ProfiledPipeline implements AutoCloseable {
        private final PaddleOcrDetector detector;
        private final PaddleOcrClassifier classifier;
        private final PaddleOcrRecognizer recognizer;
        private final PerspectiveCrop.Workspace cropper = new PerspectiveCrop.Workspace();
        private final PaddleOcrOptions options = PaddleOcrOptions.defaults();
        private final int recognitionParallelism;
        private final int classificationParallelism;

        private ProfiledPipeline(PaddleOcrDetector detector, PaddleOcrClassifier classifier,
                                 PaddleOcrRecognizer recognizer, int recognitionParallelism,
                                 int classificationParallelism) {
            this.detector = detector;
            this.classifier = classifier;
            this.recognizer = recognizer;
            this.recognitionParallelism = recognitionParallelism;
            this.classificationParallelism = classificationParallelism;
        }

        private StageSample recognize(BgrImage source) {
            return recognize(source, false);
        }

        private StageSample profile(BgrImage source) {
            return recognize(source, true);
        }

        private StageSample recognize(BgrImage source, boolean profileOperators) {
            StageSample sample = new StageSample();
            long totalStart = System.nanoTime();
            long start = System.nanoTime();
            List<DetectionBox> boxes;
            if (profileOperators) {
                try (InferenceProfiler profiler = InferenceProfiler.startShared()) {
                    boxes = detect(source);
                    sample.detectionProfile = profiler.snapshot();
                }
            } else {
                boxes = detect(source);
            }
            sample.detectionNanos = System.nanoTime() - start;

            List<OcrLineResult> lines = new ArrayList<OcrLineResult>(boxes.size());
            List<BgrImage> crops = new ArrayList<BgrImage>(boxes.size());
            List<Boolean> rotations = new ArrayList<Boolean>(boxes.size());
            for (int line = 0; line < boxes.size(); line++) {
                DetectionBox box = boxes.get(line);
                start = System.nanoTime();
                BgrImage crop = cropper.crop(source, box, line);
                sample.cropNanos += System.nanoTime() - start;
                crops.add(crop);
            }
            start = System.nanoTime();
            List<ClsClassificationResult> classifications;
            if (profileOperators) {
                try (InferenceProfiler profiler = InferenceProfiler.startShared()) {
                    classifications = classifier.classifyAll(crops, classificationParallelism);
                    sample.classificationProfile = profiler.snapshot();
                }
            } else {
                classifications = classifier.classifyAll(crops, classificationParallelism);
            }
            sample.classificationNanos = System.nanoTime() - start;
            for (int line = 0; line < boxes.size(); line++) {
                BgrImage crop = crops.get(line);
                ClsClassificationResult classification = classifications.get(line);
                boolean rotated = false;
                if (classification.requiresRotation(options.getClassifierThreshold())) {
                    start = System.nanoTime();
                    crop = BgrTransforms.rotate180(crop);
                    sample.rotationNanos += System.nanoTime() - start;
                    rotated = true;
                }
                crops.set(line, crop);
                rotations.add(rotated);
            }
            start = System.nanoTime();
            List<RecRecognitionResult> recognitions;
            if (profileOperators) {
                try (InferenceProfiler profiler = InferenceProfiler.startShared()) {
                    recognitions = recognizer.recognizeAll(crops, recognitionParallelism);
                    sample.recognitionProfile = profiler.snapshot();
                }
            } else {
                recognitions = recognizer.recognizeAll(crops, recognitionParallelism);
            }
            sample.recognitionNanos = System.nanoTime() - start;
            for (int i = 0; i < boxes.size(); i++) {
                RecRecognitionResult recognition = recognitions.get(i);
                ClsClassificationResult classification = classifications.get(i);
                boolean rotated = rotations.get(i);
                DetectionBox box = boxes.get(i);
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

        private List<DetectionBox> detect(BgrImage source) {
            return detector.detect(source,
                    options.getDetectionBitmapThreshold(), options.getDetectionBoxThreshold(),
                    options.getDetectionUnclipRatio(), options.isDetectionDilation(),
                    options.getMaxDetectionCandidates());
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
        private InferenceProfiler.Profile detectionProfile;
        private InferenceProfiler.Profile classificationProfile;
        private InferenceProfiler.Profile recognitionProfile;
    }
}
