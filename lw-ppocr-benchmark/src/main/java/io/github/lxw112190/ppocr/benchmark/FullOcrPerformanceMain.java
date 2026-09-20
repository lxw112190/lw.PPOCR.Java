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
import io.github.lxw112190.ppocr.ppocr.ParallelismPlan;
import io.github.lxw112190.ppocr.ppocr.PerspectiveCrop;
import io.github.lxw112190.ppocr.runtime.InferenceProfiler;
import io.github.lxw112190.ppocr.runtime.WorkspaceDiagnostics;
import java.io.IOException;
import java.io.InputStream;
import java.lang.management.GarbageCollectorMXBean;
import java.lang.management.ManagementFactory;
import java.lang.management.MemoryMXBean;
import java.lang.management.MemoryPoolMXBean;
import java.lang.management.MemoryType;
import java.lang.management.MemoryUsage;
import java.nio.file.Path;
import java.nio.file.Paths;
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
        boolean automaticParallelism = args.length > 4 && "auto".equalsIgnoreCase(args[4]);
        ParallelismPlan automaticPlan = automaticParallelism
                ? ParallelismPlan.automatic(
                        Math.max(1, Runtime.getRuntime().availableProcessors()), 16) : null;
        int recognitionParallelism = automaticParallelism
                ? automaticPlan.getRecognizerWorkers()
                : args.length > 4 ? positive(args[4], "recognition parallelism") : 1;
        int classificationParallelism = automaticParallelism
                ? automaticPlan.getClassifierWorkers()
                : args.length > 5 ? positive(args[5], "classification parallelism") : 1;
        String parallelismPolicy = automaticParallelism ? "auto" : "manual";
        boolean processMemoryEnabled = Boolean.parseBoolean(
                System.getProperty("lwppocr.benchmark.processMemory", "false"));
        if (detectorLimit < 32) throw new IllegalArgumentException("detector limit must be at least 32");
        KernelBackend backend = createBackend(backendName);

        BgrImage image = loadImage(System.getProperty("lwppocr.benchmark.image"));
        MemoryMXBean memory = ManagementFactory.getMemoryMXBean();
        ThreadAllocationProbe allocationProbe = ThreadAllocationProbe.create();
        long heapBefore = stabilizedHeap(memory);
        ProcessMemoryProbe processMemory = ProcessMemoryProbe.open(processMemoryEnabled);
        long loadStart = System.nanoTime();
        try (ProcessMemoryProbe ignored = processMemory;
             ProfiledPipeline pipeline = loadPipeline(detectorLimit, backend,
                recognitionParallelism, classificationParallelism, allocationProbe)) {
            long modelLoadNanos = System.nanoTime() - loadStart;
            processMemory.refresh();
            long processRssLoaded = processMemory.rssBytes();
            long processRssSampledPeak = processRssLoaded;
            long processRssWarmupPeak = processRssLoaded;
            long heapAfterLoad = stabilizedHeap(memory);
            processMemory.refresh();
            long processRssLoadedAfterGc = processMemory.rssBytes();
            processRssSampledPeak = maxSupported(processRssSampledPeak,
                    processRssLoadedAfterGc);
            processRssWarmupPeak = maxSupported(processRssWarmupPeak,
                    processRssLoadedAfterGc);
            StageSample cold = pipeline.recognize(image);
            processMemory.refresh();
            long warmupRss = processMemory.rssBytes();
            processRssSampledPeak = maxSupported(processRssSampledPeak, warmupRss);
            processRssWarmupPeak = maxSupported(processRssWarmupPeak, warmupRss);
            for (int i = 1; i < warmup; i++) {
                pipeline.recognize(image);
                processMemory.refresh();
                warmupRss = processMemory.rssBytes();
                processRssSampledPeak = maxSupported(processRssSampledPeak, warmupRss);
                processRssWarmupPeak = maxSupported(processRssWarmupPeak, warmupRss);
            }

            processMemory.refresh();
            long processRssTimedStart = processMemory.rssBytes();
            long processRssTimedPeak = processRssTimedStart;
            processRssSampledPeak = maxSupported(processRssSampledPeak, processRssTimedStart);

            resetHeapPeaks();
            long gcCountBefore = gcCount();
            long gcTimeBefore = gcTimeMillis();
            ThreadAllocationProbe.Snapshot allocatedBefore = allocationProbe.snapshot();
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
                processMemory.refresh();
                long timedRss = processMemory.rssBytes();
                processRssTimedPeak = maxSupported(processRssTimedPeak, timedRss);
                processRssSampledPeak = maxSupported(processRssSampledPeak, timedRss);
            }
            processMemory.refresh();
            long processRssLast = processMemory.rssBytes();
            processRssTimedPeak = maxSupported(processRssTimedPeak, processRssLast);
            processRssSampledPeak = maxSupported(processRssSampledPeak, processRssLast);
            long processRssHwmAfterRun = processMemory.highWaterBytes();
            long processRssTimedDelta = signedDelta(processRssLast, processRssTimedStart);
            long processRssTimedPeakDelta = signedDelta(processRssTimedPeak,
                    processRssTimedStart);
            long allocatedBytes = allocationProbe.deltaSince(allocatedBefore);
            long peakHeap = Math.max(heapPeakUsed(), usedHeap(memory));
            long gcCountDelta = nonNegativeDelta(gcCount(), gcCountBefore);
            long gcTimeDelta = nonNegativeDelta(gcTimeMillis(), gcTimeBefore);
            long heapAfter = usedHeap(memory);
            long heapAfterGc = stabilizedHeap(memory);
            processMemory.refresh();
            long processRssAfterGc = processMemory.rssBytes();
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
            WorkspaceDiagnostics workspace = pipeline.workspaceDiagnostics();

            Arrays.sort(total);
            String benchmark = detectorLimit == DEFAULT_DETECTOR_LIMIT
                    ? "full-ocr-default" : "full-ocr-det" + detectorLimit;
            String profileScope = "stable-live-threads";
            String vectorBits = vectorBitsSetting(backendName);
            String pointwiseBlock = pointwiseBlockSetting(backendName);
            System.out.printf(Locale.ROOT,
                    "{\"schema\":5,\"benchmark\":\"%s\",\"backend\":\"%s\","
                            + "\"vector_bits\":\"%s\","
                            + "\"pointwise_block\":\"%s\","
                            + "\"features\":{\"rec_projection_fusion\":%s,"
                            + "\"auto_parallelism\":%s},"
                            + "\"parallelism_policy\":\"%s\","
                            + "\"classification_parallelism\":%d,"
                            + "\"recognition_parallelism\":%d,"
                            + "\"operator_profile_scope\":\"%s\","
                            + "\"operator_profile_iterations\":1,\"operator_profile_ms\":%.3f,"
                            + "\"image_width\":%d,"
                            + "\"image_height\":%d,\"detector_limit_side\":%d,"
                            + "\"lines\":%d,\"warmup\":%d,\"iterations\":%d,"
                            + "\"total_ocr_calls\":%d,"
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
                            + "\"gc_time_ms_delta\":%d,"
                            + "\"process_memory_source\":\"%s\","
                            + "\"process_rss_loaded_bytes\":%d,"
                            + "\"process_rss_loaded_after_gc_bytes\":%d,"
                            + "\"process_rss_warmup_peak_bytes\":%d,"
                            + "\"process_rss_timed_start_bytes\":%d,"
                            + "\"process_rss_timed_peak_bytes\":%d,"
                            + "\"process_rss_last_bytes\":%d,"
                            + "\"process_rss_sampled_peak_bytes\":%d,"
                            + "\"process_rss_hwm_bytes\":%d,"
                            + "\"process_rss_after_gc_bytes\":%d,"
                            + "\"process_rss_delta_bytes\":%d,"
                            + "\"process_rss_peak_delta_bytes\":%d,"
                            + "\"process_rss_timed_delta_bytes\":%d,"
                            + "\"process_rss_timed_peak_delta_bytes\":%d,"
                            + "\"allocation_measurement\":\"%s\","
                            + "\"allocated_bytes_total\":%d,"
                            + "\"allocated_bytes_per_ocr\":%d,"
                            + "\"allocated_bytes_per_line\":%d,"
                            + "\"workspace_bytes\":%d,"
                            + "\"workspace_old_bytes\":%d,"
                            + "\"workspace_live_lower_bound_bytes\":%d,"
                            + "\"workspace_efficiency\":%.6f,"
                            + "\"db_scratch_bytes\":%d,\"crop_arena_bytes\":%d,"
                            + "\"det_workspace_bytes\":%d,\"cls_workspace_bytes\":%d,"
                            + "\"rec_workspace_bytes\":%d,\"rec_workspace_by_width\":%s,"
                            + "\"rec_fallback_workspace_bytes\":%d,"
                            + "\"decoded_constant_bytes\":%d,\"packed_weight_bytes\":%d,"
                            + "\"workspace_sessions\":{\"detector\":%d,\"classifier\":%d,\"recognizer\":%d},"
                            + "\"profile_stage_allocated_bytes\":{"
                            + "\"detection\":%d,\"crop\":%d,\"classification\":%d,"
                            + "\"rotation\":%d,\"recognition\":%d,\"sorting\":%d},"
                            + "\"operators\":%s,"
                            + "\"stage_operators\":%s,\"stage_hot_nodes\":%s}%n",
                    benchmark, backendName, vectorBits, pointwiseBlock,
                    Boolean.toString(pipeline.isProjectionFusionActive()),
                    Boolean.toString(automaticParallelism), parallelismPolicy,
                    classificationParallelism,
                    recognitionParallelism, profileScope,
                    milliseconds(profiledSample.totalNanos),
                    image.width(), image.height(), detectorLimit, lineCount,
                    warmup, iterations, warmup + iterations,
                    milliseconds(modelLoadNanos), milliseconds(cold.totalNanos),
                    milliseconds(mean(total)), milliseconds(percentile(total, 0.50)),
                    milliseconds(percentile(total, 0.95)), milliseconds(mean(detection)),
                    milliseconds(mean(crop)), milliseconds(mean(classification)),
                    milliseconds(mean(rotation)), milliseconds(mean(recognition)),
                    milliseconds(mean(sorting)), Runtime.getRuntime().availableProcessors(), heapMax,
                    heapBefore, heapAfterLoad, heapAfterLoad, modelHeap, heapAfter, heapAfter,
                    heapAfterGc, retainedHeap,
                    peakHeap, peakDelta, transientHeap, gcCountDelta, gcTimeDelta,
                    processMemory.source(), processRssLoaded, processRssLoadedAfterGc,
                    processRssWarmupPeak, processRssTimedStart, processRssTimedPeak,
                    processRssLast, processRssSampledPeak, processRssHwmAfterRun,
                    processRssAfterGc, signedDelta(processRssLast, processRssLoaded),
                    signedDelta(processRssSampledPeak, processRssLoaded), processRssTimedDelta,
                    processRssTimedPeakDelta,
                    allocationProbe.method(), allocatedBytes,
                    perOperation(allocatedBytes, iterations),
                    perOperation(allocatedBytes, (long) iterations * Math.max(1, lineCount)),
                    workspace.getWorkspaceBytes(), workspace.getOldWorkspaceBytes(),
                    workspace.getLiveLowerBoundBytes(), workspace.getEfficiency(),
                    pipeline.dbScratchBytes(), pipeline.cropWorkspaceBytes(),
                    pipeline.detectorWorkspaceBytes(), pipeline.classifierWorkspaceBytes(),
                    pipeline.recognizerWorkspaceBytes(), pipeline.recognizerWorkspaceJson(),
                    pipeline.recognizerFallbackWorkspaceBytes(),
                    pipeline.decodedConstantBytes(), pipeline.packedWeightBytes(),
                    pipeline.detectorSessionCount(), pipeline.classifierSessionCount(),
                    pipeline.recognizerSessionCount(),
                    profiledSample.detectionAllocatedBytes, profiledSample.cropAllocatedBytes,
                    profiledSample.classificationAllocatedBytes, profiledSample.rotationAllocatedBytes,
                    profiledSample.recognitionAllocatedBytes, profiledSample.sortingAllocatedBytes,
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

    private static String vectorBitsSetting(String backendName) {
        if (!"vector".equals(backendName)) return "n/a";
        String value = System.getProperty("lwppocr.vectorBits", "preferred");
        if ("128".equals(value) || "256".equals(value)
                || "512".equals(value) || "preferred".equals(value)) {
            return value;
        }
        return "unknown";
    }

    private static String pointwiseBlockSetting(String backendName) {
        if (!"vector".equals(backendName)) return "n/a";
        return System.getProperty("lwppocr.vectorPointwiseBlock", "auto");
    }

    private static ProfiledPipeline loadPipeline(int detectorLimit, KernelBackend backend,
                                                 int recognitionParallelism,
                                                 int classificationParallelism,
                                                 ThreadAllocationProbe allocationProbe) throws IOException {
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
                    recognitionParallelism, classificationParallelism, allocationProbe);
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

    private static BgrImage loadImage(String configuredPath) throws IOException {
        if (configuredPath != null && !configuredPath.trim().isEmpty()) {
            Path path = Paths.get(configuredPath);
            if (!java.nio.file.Files.isRegularFile(path)) {
                throw new IOException("benchmark image does not exist: " + path);
            }
            return ImageIoLoader.load(path);
        }
        try (InputStream input = resource(IMAGE)) {
            return ImageIoLoader.load(input);
        }
    }

    private static long signedDelta(long after, long before) {
        if (after < 0L || before < 0L) return -1L;
        return after - before;
    }

    private static long maxSupported(long first, long second) {
        if (first < 0L) return second;
        if (second < 0L) return first;
        return Math.max(first, second);
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

    private static long perOperation(long total, long operations) {
        if (total < 0L || operations <= 0L) return -1L;
        return (total + operations / 2L) / operations;
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

    private static String recWorkspaceJson(long[] values) {
        StringBuilder json = new StringBuilder("{\"192\":");
        json.append(values.length > 0 ? values[0] : 0L)
                .append(",\"320\":").append(values.length > 1 ? values[1] : 0L)
                .append(",\"480\":").append(values.length > 2 ? values[2] : 0L)
                .append(",\"640\":").append(values.length > 3 ? values[3] : 0L)
                .append(",\"960\":").append(values.length > 4 ? values[4] : 0L)
                .append('}');
        return json.toString();
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
        private final ThreadAllocationProbe allocationProbe;
        private final ArrayList<DetectionBox> boxes = new ArrayList<DetectionBox>();
        private final ArrayList<OcrLineResult> lines = new ArrayList<OcrLineResult>();
        private final ArrayList<BgrImage> crops = new ArrayList<BgrImage>();
        private ClsClassificationResult[] classifications = new ClsClassificationResult[0];
        private String[] recognitionTexts = new String[0];
        private float[] recognitionScores = new float[0];
        private int[] recognitionEmittedCounts = new int[0];
        private int[] recognitionResizedWidths = new int[0];
        private boolean[] rotations = new boolean[0];

        private ProfiledPipeline(PaddleOcrDetector detector, PaddleOcrClassifier classifier,
                                 PaddleOcrRecognizer recognizer, int recognitionParallelism,
                                 int classificationParallelism,
                                 ThreadAllocationProbe allocationProbe) {
            this.detector = detector;
            this.classifier = classifier;
            this.recognizer = recognizer;
            this.recognitionParallelism = recognitionParallelism;
            this.classificationParallelism = classificationParallelism;
            this.allocationProbe = allocationProbe;
        }

        private StageSample recognize(BgrImage source) {
            return recognize(source, false);
        }

        private StageSample profile(BgrImage source) {
            return recognize(source, true);
        }

        private boolean isProjectionFusionActive() {
            return recognizer.isProjectionFusionActive();
        }

        private WorkspaceDiagnostics workspaceDiagnostics() {
            return WorkspaceDiagnostics.aggregate(detector.workspaceDiagnostics(),
                    classifier.workspaceDiagnostics(), recognizer.workspaceDiagnostics());
        }

        private long dbScratchBytes() { return detector.dbScratchBytes(); }
        private long cropWorkspaceBytes() { return cropper.workspaceBytes(); }
        private long detectorWorkspaceBytes() {
            return detector.workspaceDiagnostics().getWorkspaceBytes();
        }
        private long classifierWorkspaceBytes() {
            return classifier.workspaceDiagnostics().getWorkspaceBytes();
        }
        private long recognizerWorkspaceBytes() {
            return recognizer.workspaceDiagnostics().getWorkspaceBytes();
        }
        private String recognizerWorkspaceJson() {
            return recWorkspaceJson(recognizer.workspaceBytesByWidth());
        }
        private long recognizerFallbackWorkspaceBytes() {
            return recognizer.fallbackWorkspaceBytes();
        }
        private long decodedConstantBytes() {
            return detector.decodedConstantBytes()
                    + classifier.decodedConstantBytes() + recognizer.decodedConstantBytes();
        }
        private long packedWeightBytes() { return recognizer.packedWeightBytes(); }

        private int detectorSessionCount() { return detector.preparedSessionCount(); }
        private int classifierSessionCount() { return classifier.preparedSessionCount(); }
        private int recognizerSessionCount() { return recognizer.preparedSessionCount(); }

        private StageSample recognize(BgrImage source, boolean profileOperators) {
            StageSample sample = new StageSample();
            long totalStart = System.nanoTime();
            long start = System.nanoTime();
            ThreadAllocationProbe.Snapshot allocationStart = profileOperators
                    ? allocationProbe.snapshot() : null;
            boxes.clear();
            if (profileOperators) {
                try (InferenceProfiler profiler = InferenceProfiler.startShared()) {
                    detect(source, boxes);
                    sample.detectionProfile = profiler.snapshot();
                }
            } else {
                detect(source, boxes);
            }
            sample.detectionNanos = System.nanoTime() - start;
            sample.detectionAllocatedBytes = profileOperators
                    ? allocationProbe.deltaSince(allocationStart) : -1L;

            lines.clear();
            crops.clear();
            lines.ensureCapacity(boxes.size());
            crops.ensureCapacity(boxes.size());
            ensureStaging(boxes.size());
            if (rotations.length < boxes.size()) rotations = new boolean[boxes.size()];
            try {
                start = System.nanoTime();
                allocationStart = profileOperators ? allocationProbe.snapshot() : null;
                cropper.cropAll(source, boxes, crops);
                sample.cropNanos = System.nanoTime() - start;
                sample.cropAllocatedBytes = profileOperators
                        ? allocationProbe.deltaSince(allocationStart) : -1L;
                start = System.nanoTime();
                allocationStart = profileOperators ? allocationProbe.snapshot() : null;
                if (profileOperators) {
                    try (InferenceProfiler profiler = InferenceProfiler.startShared()) {
                        classifier.classifyAllInto(crops, classificationParallelism, classifications);
                        sample.classificationProfile = profiler.snapshot();
                    }
                } else {
                    classifier.classifyAllInto(crops, classificationParallelism, classifications);
                }
                sample.classificationNanos = System.nanoTime() - start;
                sample.classificationAllocatedBytes = profileOperators
                        ? allocationProbe.deltaSince(allocationStart) : -1L;
                allocationStart = profileOperators ? allocationProbe.snapshot() : null;
                for (int line = 0; line < boxes.size(); line++) {
                    BgrImage crop = crops.get(line);
                    ClsClassificationResult classification = classifications[line];
                    boolean rotated = false;
                    if (classification.requiresRotation(options.getClassifierThreshold())) {
                        start = System.nanoTime();
                        BgrTransforms.rotate180InPlace(crop);
                        sample.rotationNanos += System.nanoTime() - start;
                        rotated = true;
                    }
                    rotations[line] = rotated;
                }
                sample.rotationAllocatedBytes = profileOperators
                        ? allocationProbe.deltaSince(allocationStart) : -1L;
                start = System.nanoTime();
                allocationStart = profileOperators ? allocationProbe.snapshot() : null;
                if (profileOperators) {
                    try (InferenceProfiler profiler = InferenceProfiler.startShared()) {
                        recognizer.recognizeAllInto(crops, recognitionParallelism,
                                recognitionTexts, recognitionScores, recognitionEmittedCounts,
                                recognitionResizedWidths);
                        sample.recognitionProfile = profiler.snapshot();
                    }
                } else {
                    recognizer.recognizeAllInto(crops, recognitionParallelism,
                            recognitionTexts, recognitionScores, recognitionEmittedCounts,
                            recognitionResizedWidths);
                }
                sample.recognitionNanos = System.nanoTime() - start;
                sample.recognitionAllocatedBytes = profileOperators
                        ? allocationProbe.deltaSince(allocationStart) : -1L;
                for (int i = 0; i < boxes.size(); i++) {
                    ClsClassificationResult classification = classifications[i];
                    DetectionBox box = boxes.get(i);
                    lines.add(new OcrLineResult(box, recognitionTexts[i], recognitionScores[i],
                            classification, rotations[i]));
                }
                start = System.nanoTime();
                allocationStart = profileOperators ? allocationProbe.snapshot() : null;
                OcrResult result = new OcrResult(lines).sorted(options.getReadingOrder());
                sample.sortingNanos = System.nanoTime() - start;
                sample.sortingAllocatedBytes = profileOperators
                        ? allocationProbe.deltaSince(allocationStart) : -1L;
                sample.lines = result.getLines().size();
                sample.totalNanos = System.nanoTime() - totalStart;
                return sample;
            } finally {
                Arrays.fill(classifications, null);
                Arrays.fill(recognitionTexts, null);
                // Primitive staging is overwritten on the next successful run;
                // only reference arrays need clearing to avoid retention.
                boxes.clear();
                lines.clear();
                crops.clear();
            }
        }

        private void detect(BgrImage source, List<DetectionBox> destination) {
            detector.detectInto(source,
                    options.getDetectionBitmapThreshold(), options.getDetectionBoxThreshold(),
                    options.getDetectionUnclipRatio(), options.isDetectionDilation(),
                    options.getMaxDetectionCandidates(), destination);
        }

        private void ensureStaging(int size) {
            if (classifications.length < size) classifications = new ClsClassificationResult[size];
            if (recognitionTexts.length < size) {
                recognitionTexts = new String[size];
                recognitionScores = new float[size];
                recognitionEmittedCounts = new int[size];
                recognitionResizedWidths = new int[size];
            }
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
        private long detectionAllocatedBytes = -1L;
        private long cropAllocatedBytes = -1L;
        private long classificationAllocatedBytes = -1L;
        private long rotationAllocatedBytes = -1L;
        private long recognitionAllocatedBytes = -1L;
        private long sortingAllocatedBytes = -1L;
    }

    /** Optional HotSpot allocation counter; unavailable JVMs report -1 without failing CI. */
    private static final class ThreadAllocationProbe {
        private final com.sun.management.ThreadMXBean bean;

        private ThreadAllocationProbe(com.sun.management.ThreadMXBean bean) {
            this.bean = bean;
        }

        private static ThreadAllocationProbe create() {
            java.lang.management.ThreadMXBean base = ManagementFactory.getThreadMXBean();
            if (!(base instanceof com.sun.management.ThreadMXBean)) {
                return new ThreadAllocationProbe(null);
            }
            com.sun.management.ThreadMXBean bean = (com.sun.management.ThreadMXBean) base;
            if (!bean.isThreadAllocatedMemorySupported()) return new ThreadAllocationProbe(null);
            if (!bean.isThreadAllocatedMemoryEnabled()) bean.setThreadAllocatedMemoryEnabled(true);
            return new ThreadAllocationProbe(bean);
        }

        private Snapshot snapshot() {
            if (bean == null) return Snapshot.unavailable();
            java.lang.management.ThreadMXBean threadBean = ManagementFactory.getThreadMXBean();
            long[] ids = threadBean.getAllThreadIds();
            return new Snapshot(ids, bean.getThreadAllocatedBytes(ids));
        }

        private long deltaSince(Snapshot before) {
            if (before == null || before.isUnavailable()) return -1L;
            Snapshot after = snapshot();
            if (after.isUnavailable()) return -1L;

            long total = 0L;
            for (int i = 0; i < before.threadIds.length; i++) {
                long beforeBytes = before.allocatedBytes[i];
                if (beforeBytes < 0L) continue;
                int afterIndex = indexOf(after.threadIds, before.threadIds[i]);
                if (afterIndex < 0) continue;
                long afterBytes = after.allocatedBytes[afterIndex];
                if (afterBytes >= 0L) total += nonNegativeDelta(afterBytes, beforeBytes);
            }
            return total;
        }

        private static int indexOf(long[] values, long value) {
            for (int i = 0; i < values.length; i++) {
                if (values[i] == value) return i;
            }
            return -1;
        }

        private static final class Snapshot {
            private final long[] threadIds;
            private final long[] allocatedBytes;

            private Snapshot(long[] threadIds, long[] allocatedBytes) {
                this.threadIds = threadIds;
                this.allocatedBytes = allocatedBytes;
            }

            private static Snapshot unavailable() {
                return new Snapshot(null, null);
            }

            private boolean isUnavailable() {
                return threadIds == null;
            }
        }

        private String method() {
            return bean == null ? "unavailable" : "hotspot-thread-mxbean-stable-live-threads";
        }
    }
}
