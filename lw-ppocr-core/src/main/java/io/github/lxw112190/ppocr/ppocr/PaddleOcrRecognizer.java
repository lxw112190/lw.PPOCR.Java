package io.github.lxw112190.ppocr.ppocr;

import io.github.lxw112190.ppocr.image.BgrImage;
import io.github.lxw112190.ppocr.kernels.KernelBackend;
import io.github.lxw112190.ppocr.kernels.ScalarBackend;
import io.github.lxw112190.ppocr.model.DataType;
import io.github.lxw112190.ppocr.model.LwmLoader;
import io.github.lxw112190.ppocr.model.LwmModel;
import io.github.lxw112190.ppocr.model.OcrErrorCode;
import io.github.lxw112190.ppocr.model.OcrException;
import io.github.lxw112190.ppocr.model.TensorInfo;
import io.github.lxw112190.ppocr.runtime.WorkspaceDiagnostics;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.nio.file.Path;
import java.util.concurrent.ThreadFactory;
import java.util.concurrent.atomic.AtomicInteger;

/** Dynamic-width REC facade: BGR preprocessing, graph execution, and CTC decoding. */
public final class PaddleOcrRecognizer implements AutoCloseable {
    private static final int DEFAULT_MAXIMUM_WIDTH = 960;
    private final LwmModel model;
    private final PaddleOcrDictionary dictionary;
    private final int maximumWidth;
    private final boolean dynamicWidth;
    private final KernelBackend backend;
    private final RecSessionContext[] sessions;
    private RecSessionContext fallbackSession;
    private final RecognitionGroup[] reusableGroups;
    private RecognitionGroup fallbackGroup;
    private final RecognitionGroup[] activeGroups;
    private int activeGroupCount;
    private final ReusableParallelExecutor parallelExecutor;
    private final GroupWorkerAction groupWorkerAction;
    private boolean closed;

    /** Takes ownership of the model and dictionary and closes both on close(). */
    public PaddleOcrRecognizer(LwmModel model, PaddleOcrDictionary dictionary) {
        this(model, dictionary, new ScalarBackend());
    }

    /** Creates a recognizer using the supplied stateless kernel backend. */
    public PaddleOcrRecognizer(LwmModel model, PaddleOcrDictionary dictionary,
                               KernelBackend backend) {
        if (model == null || dictionary == null || backend == null) {
            throw new OcrException(OcrErrorCode.INVALID_ARGUMENT,
                    "REC model, dictionary, and backend are required");
        }
        int inputIndex = validateModel(model, dictionary);
        TensorInfo input = model.getTensors().get(inputIndex);
        this.model = model;
        this.dictionary = dictionary;
        int declaredWidth = input.getDimensions()[3];
        this.dynamicWidth = declaredWidth == -1;
        this.maximumWidth = dynamicWidth ? DEFAULT_MAXIMUM_WIDTH : declaredWidth;
        this.backend = backend;
        this.sessions = new RecSessionContext[RecWidthPolicy.bucketCount()];
        this.reusableGroups = new RecognitionGroup[RecWidthPolicy.bucketCount()];
        this.activeGroups = new RecognitionGroup[RecWidthPolicy.bucketCount()];
        this.parallelExecutor = new ReusableParallelExecutor(DaemonThreadFactory.INSTANCE);
        this.groupWorkerAction = new GroupWorkerAction();
    }

    public static PaddleOcrRecognizer load(Path modelPath, Path dictionaryPath) {
        return load(modelPath, dictionaryPath, new ScalarBackend());
    }

    /** Loads a recognizer and dictionary using the supplied stateless kernel backend. */
    public static PaddleOcrRecognizer load(Path modelPath, Path dictionaryPath,
                                           KernelBackend backend) {
        LwmModel model = LwmLoader.load(modelPath);
        PaddleOcrDictionary dictionary = null;
        try {
            dictionary = PaddleOcrDictionary.load(dictionaryPath);
            return new PaddleOcrRecognizer(model, dictionary, backend);
        } catch (RuntimeException e) {
            if (dictionary != null) dictionary.close();
            model.close();
            throw e;
        }
    }

    public RecRecognitionResult recognize(BgrImage source) {
        ensureOpen();
        if (source == null) {
            throw new OcrException(OcrErrorCode.INVALID_ARGUMENT, "REC source image is required");
        }
        int slot = dynamicWidth
                ? RecWidthPolicy.chooseBucketIndex(source, maximumWidth)
                : RecWidthPolicy.bucketIndex(maximumWidth);
        int targetWidth = RecWidthPolicy.widthForIndex(slot, maximumWidth);
        return recognize(source, context(slot, targetWidth));
    }

    /** Reports whether at least one prepared dynamic-width session uses terminal fusion. */
    public boolean isProjectionFusionActive() {
        ensureOpen();
        for (RecSessionContext context : sessions) {
            if (context != null && context.usesProjectionFusion()) return true;
        }
        if (fallbackSession != null && fallbackSession.usesProjectionFusion()) return true;
        return false;
    }

    /** Returns the combined workspace measurements of currently prepared REC width sessions. */
    public WorkspaceDiagnostics workspaceDiagnostics() {
        ensureOpen();
        WorkspaceDiagnostics[] values = new WorkspaceDiagnostics[sessions.length + 1];
        int count = 0;
        for (RecSessionContext context : sessions) {
            if (context != null) values[count++] = context.workspaceDiagnostics();
        }
        if (fallbackSession != null) values[count++] = fallbackSession.workspaceDiagnostics();
        return WorkspaceDiagnostics.aggregate(Arrays.copyOf(values, count));
    }

    /** Returns canonical FP32 constants materialized by the REC model. */
    public long decodedConstantBytes() {
        ensureOpen();
        for (RecSessionContext context : sessions) {
            if (context != null) return context.decodedConstantBytes();
        }
        return fallbackSession == null ? 0L : fallbackSession.decodedConstantBytes();
    }

    /** Returns prepared projection matrices retained by REC sessions in bytes. */
    public long packedWeightBytes() {
        ensureOpen();
        long total = 0L;
        for (RecSessionContext context : sessions) {
            if (context != null) total += context.packedWeightBytes();
        }
        if (fallbackSession != null) total += fallbackSession.packedWeightBytes();
        return total;
    }

    /** Returns workspace bytes for the fixed REC buckets in policy order. */
    public long[] workspaceBytesByWidth() {
        ensureOpen();
        long[] values = new long[sessions.length];
        for (int i = 0; i < sessions.length; i++) {
            if (sessions[i] != null) values[i] = sessions[i].workspaceDiagnostics().getWorkspaceBytes();
        }
        return values;
    }

    /** Returns workspace bytes for a non-bucket fallback REC session, if any. */
    public long fallbackWorkspaceBytes() {
        ensureOpen();
        return fallbackSession == null ? 0L : fallbackSession.workspaceDiagnostics().getWorkspaceBytes();
    }

    /** Returns the number of currently prepared REC width sessions. */
    public int preparedSessionCount() {
        ensureOpen();
        int count = fallbackSession == null ? 0 : 1;
        for (RecSessionContext context : sessions) if (context != null) count++;
        return count;
    }

    /** Recognizes images in input order while evaluating independent width groups concurrently. */
    public List<RecRecognitionResult> recognizeAll(List<BgrImage> sources, int parallelism) {
        validateBatch(sources, parallelism);
        if (sources.isEmpty()) return Collections.emptyList();
        RecRecognitionResult[] results = new RecRecognitionResult[sources.size()];
        recognizeAllInto(sources, parallelism, results);
        return Arrays.asList(results);
    }

    /** Internal low-allocation batch path used by the owning OCR pipeline. */
    void recognizeAllInto(List<BgrImage> sources, int parallelism,
                          RecRecognitionResult[] results) {
        validateBatch(sources, parallelism);
        validateObjectResults(sources.size(), results);
        if (sources.isEmpty()) return;

        prepareGroups(sources);
        try {
            executeGroups(parallelism, results, null, null, null, null);
        } finally {
            resetGroups();
        }
    }

    /** Recognizes into caller-owned primitive result storage without allocating a result list. */
    public void recognizeAllInto(List<BgrImage> sources, int parallelism,
                                 String[] texts, float[] scores, int[] emittedCounts,
                                 int[] resizedWidths) {
        validateBatch(sources, parallelism);
        validatePrimitiveResults(sources.size(), texts, scores, emittedCounts, resizedWidths);
        if (sources.isEmpty()) return;

        prepareGroups(sources);
        try {
            executeGroups(parallelism, null, texts, scores, emittedCounts, resizedWidths);
        } finally {
            resetGroups();
        }
    }

    private void validateBatch(List<BgrImage> sources, int parallelism) {
        ensureOpen();
        if (sources == null || parallelism <= 0 || parallelism > 64) {
            throw new OcrException(OcrErrorCode.INVALID_ARGUMENT,
                    "REC sources and parallelism are invalid");
        }
        for (BgrImage source : sources) {
            if (source == null) {
                throw new OcrException(OcrErrorCode.INVALID_ARGUMENT, "REC source image is required");
            }
        }
    }

    private static void validateObjectResults(int size, RecRecognitionResult[] results) {
        if (results == null || results.length < size) {
            throw new OcrException(OcrErrorCode.INVALID_ARGUMENT,
                    "REC result buffer is smaller than the source batch");
        }
    }

    private static void validatePrimitiveResults(int size, String[] texts, float[] scores,
                                                 int[] emittedCounts, int[] resizedWidths) {
        if (texts == null || scores == null || emittedCounts == null || resizedWidths == null
                || texts.length < size || scores.length < size
                || emittedCounts.length < size || resizedWidths.length < size) {
            throw new OcrException(OcrErrorCode.INVALID_ARGUMENT,
                    "REC primitive result buffers are smaller than the source batch");
        }
    }

    private void prepareGroups(List<BgrImage> sources) {
        resetGroups();
        for (int i = 0; i < sources.size(); i++) {
            BgrImage source = sources.get(i);
            int slot = dynamicWidth
                    ? RecWidthPolicy.chooseBucketIndex(source, maximumWidth)
                    : RecWidthPolicy.bucketIndex(maximumWidth);
            int width = RecWidthPolicy.widthForIndex(slot, maximumWidth);
            RecognitionGroup group = group(slot, width);
            if (group.isEmpty()) activeGroups[activeGroupCount++] = group;
            group.add(i, source);
        }
    }

    private void executeGroups(int parallelism, RecRecognitionResult[] objectResults,
                               String[] texts, float[] scores, int[] emittedCounts,
                               int[] resizedWidths) {
        int workers = Math.min(parallelism, activeGroupCount);
        if (workers == 1) {
            for (int i = 0; i < activeGroupCount; i++) {
                recognize(activeGroups[i], objectResults, texts, scores,
                        emittedCounts, resizedWidths);
            }
            return;
        }

        sortGroupsLargestFirst();
        groupWorkerAction.prepare(objectResults, texts, scores, emittedCounts, resizedWidths);
        try {
            parallelExecutor.run(workers, groupWorkerAction);
        } finally {
            groupWorkerAction.clear();
        }
    }

    private void resetGroups() {
        for (int i = 0; i < activeGroupCount; i++) {
            activeGroups[i].clear();
            activeGroups[i] = null;
        }
        activeGroupCount = 0;
    }

    private RecSessionContext context(int slot, int targetWidth) {
        if (slot >= 0) {
            RecSessionContext context = sessions[slot];
            if (context == null) {
                context = new RecSessionContext(model, targetWidth, dictionary.classCount(), backend);
                sessions[slot] = context;
            }
            return context;
        }
        if (fallbackSession == null) {
            fallbackSession = new RecSessionContext(model, targetWidth, dictionary.classCount(), backend);
        }
        return fallbackSession;
    }

    private RecognitionGroup group(int slot, int targetWidth) {
        if (slot >= 0) {
            RecognitionGroup group = reusableGroups[slot];
            if (group == null) {
                group = new RecognitionGroup(targetWidth, context(slot, targetWidth));
                reusableGroups[slot] = group;
            }
            return group;
        }
        if (fallbackGroup == null) {
            fallbackGroup = new RecognitionGroup(targetWidth, context(slot, targetWidth));
        }
        return fallbackGroup;
    }

    private void sortGroupsLargestFirst() {
        for (int i = 1; i < activeGroupCount; i++) {
            RecognitionGroup value = activeGroups[i];
            long work = value.estimatedWork();
            int j = i - 1;
            while (j >= 0 && activeGroups[j].estimatedWork() < work) {
                activeGroups[j + 1] = activeGroups[j];
                j--;
            }
            activeGroups[j + 1] = value;
        }
    }

    private RecRecognitionResult recognize(BgrImage source, RecSessionContext context) {
        context.preprocess(source);
        CtcDecodeResult decoded = context.run(dictionary);
        return new RecRecognitionResult(decoded.getText(), decoded.getScore(),
                decoded.getEmittedCount(), context.resizedWidth());
    }

    private void recognize(RecognitionGroup group, RecRecognitionResult[] objectResults,
                           String[] texts, float[] scores, int[] emittedCounts,
                           int[] resizedWidths) {
        for (int i = 0; i < group.size; i++) {
            int resultIndex = group.indexes[i];
            if (objectResults != null) {
                objectResults[resultIndex] = recognize(group.sources[i], group.context);
            } else {
                recognizeInto(group.sources[i], group.context, resultIndex,
                        texts, scores, emittedCounts, resizedWidths);
            }
        }
    }

    private void recognizeInto(BgrImage source, RecSessionContext context, int resultIndex,
                                String[] texts, float[] scores, int[] emittedCounts,
                                int[] resizedWidths) {
        context.preprocess(source);
        texts[resultIndex] = context.runInto(dictionary, scores, resultIndex,
                emittedCounts, resultIndex);
        resizedWidths[resultIndex] = context.resizedWidth();
    }

    @Override
    public void close() {
        if (!closed) {
            closed = true;
            parallelExecutor.close();
            for (int i = 0; i < sessions.length; i++) {
                if (sessions[i] != null) {
                    sessions[i].close();
                    sessions[i] = null;
                }
            }
            if (fallbackSession != null) {
                fallbackSession.close();
                fallbackSession = null;
            }
            resetGroups();
            Arrays.fill(reusableGroups, null);
            fallbackGroup = null;
            dictionary.close();
            model.close();
        }
    }

    private static int validateModel(LwmModel model, PaddleOcrDictionary dictionary) {
        if (model.getGraphInputs().size() != 1 || model.getGraphOutputs().size() != 1) {
            throw invalid("REC model must have one input and one output");
        }
        TensorInfo input = model.getTensors().get(model.getGraphInputs().get(0));
        int[] dimensions = input.getDimensions();
        if (input.getDataType() != DataType.F32 || dimensions.length != 4 ||
                (dimensions[0] != -1 && dimensions[0] != 1) ||
                dimensions[1] != 3 || dimensions[2] != RecPreprocess.INPUT_HEIGHT ||
                (dimensions[3] != -1 && dimensions[3] <= 0)) {
            throw invalid("REC input must be FP32 [1,3,48,W]");
        }
        TensorInfo output = model.getTensors().get(model.getGraphOutputs().get(0));
        int[] outputDimensions = output.getDimensions();
        if (output.getDataType() != DataType.F32 ||
                (outputDimensions.length != 2 && outputDimensions.length != 3)) {
            throw invalid("REC output must be FP32 [T,C] or [1,T,C]");
        }
        int classAxis = outputDimensions.length - 1;
        if (outputDimensions[classAxis] != dictionary.classCount() ||
                (outputDimensions.length == 3 && outputDimensions[0] != -1 && outputDimensions[0] != 1) ||
                (outputDimensions[outputDimensions.length - 2] != -1 &&
                        outputDimensions[outputDimensions.length - 2] <= 0)) {
            throw invalid("REC output class count does not match dictionary");
        }
        return model.getGraphInputs().get(0);
    }

    private static OcrException invalid(String message) {
        return new OcrException(OcrErrorCode.INVALID_MODEL, message);
    }

    private void ensureOpen() {
        if (closed) throw new OcrException(OcrErrorCode.INVALID_ARGUMENT, "REC recognizer is closed");
    }

    private static final class RecognitionGroup {
        private final int targetWidth;
        private final RecSessionContext context;
        private int[] indexes = new int[0];
        private BgrImage[] sources = new BgrImage[0];
        private int size;

        private RecognitionGroup(int targetWidth, RecSessionContext context) {
            this.targetWidth = targetWidth;
            this.context = context;
        }

        private long estimatedWork() {
            return (long) targetWidth * size;
        }

        private boolean isEmpty() {
            return size == 0;
        }

        private void add(int index, BgrImage source) {
            ensureCapacity(size + 1);
            indexes[size] = index;
            sources[size] = source;
            size++;
        }

        private void clear() {
            Arrays.fill(sources, 0, size, null);
            size = 0;
        }

        private void ensureCapacity(int required) {
            if (indexes.length >= required) return;
            int capacity = Math.max(required, Math.max(4, indexes.length * 2));
            indexes = Arrays.copyOf(indexes, capacity);
            sources = Arrays.copyOf(sources, capacity);
        }
    }

    /** Reusable multi-worker dispatcher; one recognizer owns one instance. */
    private final class GroupWorkerAction implements ReusableParallelExecutor.WorkerAction {
        private RecRecognitionResult[] objectResults;
        private String[] texts;
        private float[] scores;
        private int[] emittedCounts;
        private int[] resizedWidths;
        private int nextGroup;

        private void prepare(RecRecognitionResult[] objectResults, String[] texts,
                             float[] scores, int[] emittedCounts, int[] resizedWidths) {
            this.objectResults = objectResults;
            this.texts = texts;
            this.scores = scores;
            this.emittedCounts = emittedCounts;
            this.resizedWidths = resizedWidths;
            this.nextGroup = 0;
        }

        private void clear() {
            objectResults = null;
            texts = null;
            scores = null;
            emittedCounts = null;
            resizedWidths = null;
            nextGroup = 0;
        }

        private synchronized int claimGroup() {
            return nextGroup < activeGroupCount ? nextGroup++ : -1;
        }

        @Override
        public void run(int workerIndex) {
            int groupIndex;
            while ((groupIndex = claimGroup()) >= 0) {
                recognize(activeGroups[groupIndex], objectResults, texts, scores,
                        emittedCounts, resizedWidths);
            }
        }
    }

    private static final class DaemonThreadFactory implements ThreadFactory {
        private static final DaemonThreadFactory INSTANCE = new DaemonThreadFactory();
        private static final AtomicInteger IDS = new AtomicInteger();

        @Override
        public Thread newThread(Runnable runnable) {
            Thread thread = new Thread(runnable, "lw-ppocr-rec-" + IDS.incrementAndGet());
            thread.setDaemon(true);
            return thread;
        }
    }
}
