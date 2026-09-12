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
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.ThreadFactory;
import java.util.concurrent.atomic.AtomicInteger;

/** Fixed-shape CLS facade: BGR preprocessing, graph execution, and decision postprocess. */
public final class PaddleOcrClassifier implements AutoCloseable {
    private static final int[] INPUT_DIMENSIONS = {1, 3, ClsPreprocess.INPUT_HEIGHT, ClsPreprocess.INPUT_WIDTH};
    private final LwmModel model;
    private final KernelBackend backend;
    private final List<ClsSessionContext> contexts;
    private ExecutorService parallelExecutor;
    private int parallelExecutorSize;
    private boolean closed;

    public PaddleOcrClassifier(LwmModel model) {
        this(model, new ScalarBackend());
    }

    /** Creates a classifier using the supplied stateless kernel backend. */
    public PaddleOcrClassifier(LwmModel model, KernelBackend backend) {
        if (model == null || backend == null) {
            throw new OcrException(OcrErrorCode.INVALID_ARGUMENT, "CLS model and backend are required");
        }
        validateModel(model);
        this.model = model;
        this.backend = backend;
        this.contexts = new ArrayList<ClsSessionContext>();
        try {
            contexts.add(new ClsSessionContext(model, backend));
        } catch (RuntimeException e) {
            model.close();
            throw e;
        }
    }

    public static PaddleOcrClassifier load(Path path) {
        return load(path, new ScalarBackend());
    }

    /** Loads a classifier using the supplied stateless kernel backend. */
    public static PaddleOcrClassifier load(Path path, KernelBackend backend) {
        LwmModel model = LwmLoader.load(path);
        try {
            return new PaddleOcrClassifier(model, backend);
        } catch (RuntimeException e) {
            model.close();
            throw e;
        }
    }

    public ClsClassificationResult classify(BgrImage source) {
        ensureOpen();
        if (source == null) {
            throw new OcrException(OcrErrorCode.INVALID_ARGUMENT, "CLS source image is required");
        }
        return contexts.get(0).classify(source);
    }

    /** Classifies images concurrently while preserving input order. */
    public List<ClsClassificationResult> classifyAll(List<BgrImage> sources, int parallelism) {
        ensureOpen();
        if (sources == null || parallelism <= 0 || parallelism > 64) {
            throw new OcrException(OcrErrorCode.INVALID_ARGUMENT,
                    "CLS sources and parallelism are invalid");
        }
        if (sources.isEmpty()) return Collections.emptyList();
        for (BgrImage source : sources) {
            if (source == null) {
                throw new OcrException(OcrErrorCode.INVALID_ARGUMENT,
                        "CLS source image is required");
            }
        }

        final int workers = Math.min(parallelism, sources.size());
        ensureContexts(workers);
        final ClsClassificationResult[] results = new ClsClassificationResult[sources.size()];
        if (workers == 1) {
            classifyWorker(sources, results, 0, 1);
            return Arrays.asList(results);
        }

        ExecutorService executor = executor(workers);
        List<Future<Void>> futures = new ArrayList<Future<Void>>(workers);
        for (int workerIndex = 0; workerIndex < workers; workerIndex++) {
            final int index = workerIndex;
            futures.add(executor.submit(new Callable<Void>() {
                @Override
                public Void call() {
                    classifyWorker(sources, results, index, workers);
                    return null;
                }
            }));
        }
        await(futures);
        return Arrays.asList(results);
    }

    @Override
    public void close() {
        if (!closed) {
            closed = true;
            if (parallelExecutor != null) parallelExecutor.shutdown();
            for (ClsSessionContext context : contexts) context.close();
            contexts.clear();
            model.close();
        }
    }

    private void ensureContexts(int size) {
        while (contexts.size() < size) contexts.add(new ClsSessionContext(model, backend));
    }

    private void classifyWorker(List<BgrImage> sources, ClsClassificationResult[] results,
                                int first, int stride) {
        ClsSessionContext context = contexts.get(first);
        for (int i = first; i < sources.size(); i += stride) {
            results[i] = context.classify(sources.get(i));
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

    private static void await(List<Future<Void>> futures) {
        RuntimeException failure = null;
        for (Future<Void> future : futures) {
            try {
                future.get();
            } catch (InterruptedException e) {
                for (Future<Void> pending : futures) pending.cancel(true);
                Thread.currentThread().interrupt();
                throw new OcrException(OcrErrorCode.RESOURCE_LIMIT,
                        "parallel CLS execution was interrupted", e);
            } catch (ExecutionException e) {
                Throwable cause = e.getCause();
                RuntimeException current = cause instanceof RuntimeException
                        ? (RuntimeException) cause
                        : new OcrException(OcrErrorCode.RESOURCE_LIMIT,
                                "parallel CLS execution failed", cause);
                if (failure == null) failure = current;
                else failure.addSuppressed(current);
            }
        }
        if (failure != null) throw failure;
    }

    private static void validateModel(LwmModel model) {
        if (model.getGraphInputs().size() != 1 || model.getGraphOutputs().size() != 1) {
            throw invalid("CLS model must have one input and one output");
        }
        TensorInfo input = model.getTensors().get(model.getGraphInputs().get(0));
        if (input.getDataType() != DataType.F32 || !Arrays.equals(input.getDimensions(), INPUT_DIMENSIONS)) {
            throw invalid("CLS input must be FP32 [1,3,80,160]");
        }
        TensorInfo output = model.getTensors().get(model.getGraphOutputs().get(0));
        if (output.getDataType() != DataType.F32 || output.getRank() == 0 || elementCount(output) != 2) {
            throw invalid("CLS output must contain two FP32 class scores");
        }
    }

    private static long elementCount(TensorInfo tensor) {
        long result = 1;
        for (int dimension : tensor.getDimensions()) result *= dimension;
        return result;
    }

    private static OcrException invalid(String message) {
        return new OcrException(OcrErrorCode.INVALID_MODEL, message);
    }

    private void ensureOpen() {
        if (closed) throw new OcrException(OcrErrorCode.INVALID_ARGUMENT, "CLS classifier is closed");
    }

    private static final class DaemonThreadFactory implements ThreadFactory {
        private static final DaemonThreadFactory INSTANCE = new DaemonThreadFactory();
        private static final AtomicInteger IDS = new AtomicInteger();

        @Override
        public Thread newThread(Runnable runnable) {
            Thread thread = new Thread(runnable, "lw-ppocr-cls-" + IDS.incrementAndGet());
            thread.setDaemon(true);
            return thread;
        }
    }
}
