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
import java.util.HashMap;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;
import java.util.Map;
import java.nio.file.Path;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
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
    private final Map<Integer, RecSessionContext> sessions;
    private final Map<Integer, RecognitionGroup> reusableGroups;
    private final List<RecognitionGroup> activeGroups;
    private final List<Future<Void>> taskFutures;
    private ExecutorService parallelExecutor;
    private int parallelExecutorSize;
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
        this.sessions = new HashMap<Integer, RecSessionContext>();
        this.reusableGroups = new HashMap<Integer, RecognitionGroup>();
        this.activeGroups = new ArrayList<RecognitionGroup>();
        this.taskFutures = new ArrayList<Future<Void>>();
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
        int targetWidth = dynamicWidth ? RecWidthPolicy.chooseTargetWidth(source, maximumWidth) : maximumWidth;
        return recognize(source, context(targetWidth));
    }

    /** Reports whether at least one prepared dynamic-width session uses terminal fusion. */
    public boolean isProjectionFusionActive() {
        ensureOpen();
        for (RecSessionContext context : sessions.values()) {
            if (context.usesProjectionFusion()) return true;
        }
        return false;
    }

    /** Recognizes images in input order while evaluating independent width groups concurrently. */
    public List<RecRecognitionResult> recognizeAll(List<BgrImage> sources, int parallelism) {
        validateBatch(sources, parallelism, null);
        if (sources.isEmpty()) return Collections.emptyList();
        RecRecognitionResult[] results = new RecRecognitionResult[sources.size()];
        recognizeAllInto(sources, parallelism, results);
        return Arrays.asList(results);
    }

    /** Internal low-allocation batch path used by the owning OCR pipeline. */
    void recognizeAllInto(List<BgrImage> sources, int parallelism,
                          RecRecognitionResult[] results) {
        validateBatch(sources, parallelism, results);
        if (sources.isEmpty()) return;

        resetGroups();
        try {
            for (int i = 0; i < sources.size(); i++) {
                BgrImage source = sources.get(i);
                int width = dynamicWidth
                        ? RecWidthPolicy.chooseTargetWidth(source, maximumWidth) : maximumWidth;
                RecognitionGroup group = reusableGroups.get(width);
                if (group == null) {
                    group = new RecognitionGroup(width, context(width));
                    reusableGroups.put(width, group);
                }
                if (group.isEmpty()) activeGroups.add(group);
                group.add(i, source);
            }
            int workers = Math.min(parallelism, activeGroups.size());
            if (workers == 1) {
                for (RecognitionGroup group : activeGroups) recognize(group, results);
                return;
            }

            ExecutorService executor = executor(workers);
            Collections.sort(activeGroups, RecognitionGroup.LARGEST_WORK_FIRST);
            taskFutures.clear();
            for (RecognitionGroup scheduled : activeGroups) {
                final RecognitionGroup group = scheduled;
                taskFutures.add(executor.submit(new Callable<Void>() {
                    @Override
                    public Void call() {
                        recognize(group, results);
                        return null;
                    }
                }));
            }
            await(taskFutures);
        } finally {
            taskFutures.clear();
            resetGroups();
        }
    }

    private void validateBatch(List<BgrImage> sources, int parallelism,
                               RecRecognitionResult[] results) {
        ensureOpen();
        if (sources == null || parallelism <= 0 || parallelism > 64) {
            throw new OcrException(OcrErrorCode.INVALID_ARGUMENT,
                    "REC sources and parallelism are invalid");
        }
        if (results != null && results.length < sources.size()) {
            throw new OcrException(OcrErrorCode.INVALID_ARGUMENT,
                    "REC result buffer is smaller than the source batch");
        }
        for (BgrImage source : sources) {
            if (source == null) {
                throw new OcrException(OcrErrorCode.INVALID_ARGUMENT, "REC source image is required");
            }
        }
    }

    private void resetGroups() {
        for (RecognitionGroup group : activeGroups) group.clear();
        activeGroups.clear();
    }

    private static void await(List<Future<Void>> futures) {
        RuntimeException failure = null;
        for (Future<Void> future : futures) {
            try {
                future.get();
            } catch (InterruptedException e) {
                for (Future<Void> pending : futures) pending.cancel(true);
                Thread.currentThread().interrupt();
                throw new OcrException(OcrErrorCode.RESOURCE_LIMIT,
                        "parallel REC execution was interrupted", e);
            } catch (ExecutionException e) {
                Throwable cause = e.getCause();
                RuntimeException current = cause instanceof RuntimeException
                        ? (RuntimeException) cause
                        : new OcrException(OcrErrorCode.RESOURCE_LIMIT,
                                "parallel REC execution failed", cause);
                if (failure == null) failure = current;
                else failure.addSuppressed(current);
            }
        }
        if (failure != null) throw failure;
    }

    private RecSessionContext context(int targetWidth) {
        RecSessionContext context = sessions.get(targetWidth);
        if (context == null) {
            context = new RecSessionContext(model, targetWidth, dictionary.classCount(), backend);
            sessions.put(targetWidth, context);
        }
        return context;
    }

    private RecRecognitionResult recognize(BgrImage source, RecSessionContext context) {
        context.preprocess.resizeNormalize(source);
        CtcDecodeResult decoded = context.run(context.preprocess.getChw(), dictionary);
        return new RecRecognitionResult(decoded.getText(), decoded.getScore(),
                decoded.getEmittedCount(), context.preprocess.getResizedWidth());
    }

    private void recognize(RecognitionGroup group, RecRecognitionResult[] results) {
        for (int i = 0; i < group.size; i++) {
            results[group.indexes[i]] = recognize(group.sources[i], group.context);
        }
    }

    private ExecutorService executor(int size) {
        if (parallelExecutor == null || parallelExecutorSize < size) {
            if (parallelExecutor != null) parallelExecutor.shutdown();
            parallelExecutor = Executors.newFixedThreadPool(size, DaemonThreadFactory.INSTANCE);
            parallelExecutorSize = size;
        }
        return parallelExecutor;
    }

    @Override
    public void close() {
        if (!closed) {
            closed = true;
            if (parallelExecutor != null) parallelExecutor.shutdown();
            for (RecSessionContext context : sessions.values()) context.close();
            sessions.clear();
            resetGroups();
            reusableGroups.clear();
            taskFutures.clear();
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
        private static final Comparator<RecognitionGroup> LARGEST_WORK_FIRST =
                new Comparator<RecognitionGroup>() {
                    @Override
                    public int compare(RecognitionGroup left, RecognitionGroup right) {
                        return Long.compare(right.estimatedWork(), left.estimatedWork());
                    }
                };

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
