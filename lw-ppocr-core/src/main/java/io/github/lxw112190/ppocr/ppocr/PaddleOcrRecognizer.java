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
import java.util.LinkedHashMap;
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

    /** Recognizes images in input order while evaluating independent width groups concurrently. */
    public List<RecRecognitionResult> recognizeAll(List<BgrImage> sources, int parallelism) {
        ensureOpen();
        if (sources == null || parallelism <= 0 || parallelism > 64) {
            throw new OcrException(OcrErrorCode.INVALID_ARGUMENT,
                    "REC sources and parallelism are invalid");
        }
        if (sources.isEmpty()) return Collections.emptyList();
        Map<Integer, RecognitionGroup> byWidth = new LinkedHashMap<Integer, RecognitionGroup>();
        for (int i = 0; i < sources.size(); i++) {
            BgrImage source = sources.get(i);
            if (source == null) {
                throw new OcrException(OcrErrorCode.INVALID_ARGUMENT, "REC source image is required");
            }
            int width = dynamicWidth ? RecWidthPolicy.chooseTargetWidth(source, maximumWidth) : maximumWidth;
            RecognitionGroup group = byWidth.get(width);
            if (group == null) {
                group = new RecognitionGroup(width, context(width));
                byWidth.put(width, group);
            }
            group.indexes.add(i);
            group.sources.add(source);
        }
        final RecRecognitionResult[] results = new RecRecognitionResult[sources.size()];
        final List<RecognitionGroup> groups = new ArrayList<RecognitionGroup>(byWidth.values());
        int workers = Math.min(parallelism, groups.size());
        if (workers == 1) {
            for (RecognitionGroup group : groups) recognize(group, results);
            return Arrays.asList(results);
        }

        ExecutorService executor = executor(workers);
        Collections.sort(groups, RecognitionGroup.LARGEST_WORK_FIRST);
        List<Future<Void>> futures = new ArrayList<Future<Void>>(groups.size());
        for (RecognitionGroup scheduled : groups) {
            final RecognitionGroup group = scheduled;
            futures.add(executor.submit(new Callable<Void>() {
                @Override
                public Void call() {
                    recognize(group, results);
                    return null;
                }
            }));
        }
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
        return Arrays.asList(results);
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
        context.session.run(context.preprocess.getChw(), context.logits);
        CtcDecodeResult decoded = CtcDecoder.decodeGreedy(context.logits, context.timeSteps,
                dictionary.classCount(), dictionary);
        return new RecRecognitionResult(decoded.getText(), decoded.getScore(),
                decoded.getEmittedCount(), context.preprocess.getResizedWidth());
    }

    private void recognize(RecognitionGroup group, RecRecognitionResult[] results) {
        for (int i = 0; i < group.sources.size(); i++) {
            results[group.indexes.get(i)] = recognize(group.sources.get(i), group.context);
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
        private final List<Integer> indexes = new ArrayList<Integer>();
        private final List<BgrImage> sources = new ArrayList<BgrImage>();

        private RecognitionGroup(int targetWidth, RecSessionContext context) {
            this.targetWidth = targetWidth;
            this.context = context;
        }

        private long estimatedWork() {
            return (long) targetWidth * sources.size();
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
