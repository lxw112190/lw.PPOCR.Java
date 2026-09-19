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
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.concurrent.ThreadFactory;
import java.util.concurrent.atomic.AtomicInteger;

/** Fixed-shape CLS facade: BGR preprocessing, graph execution, and decision postprocess. */
public final class PaddleOcrClassifier implements AutoCloseable {
    private static final int[] INPUT_DIMENSIONS = {1, 3, ClsPreprocess.INPUT_HEIGHT, ClsPreprocess.INPUT_WIDTH};
    private final LwmModel model;
    private final KernelBackend backend;
    private final List<ClsSessionContext> contexts;
    private final ReusableParallelExecutor parallelExecutor;
    private final ClassifyWorkerAction workerAction;
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
        this.parallelExecutor = new ReusableParallelExecutor(DaemonThreadFactory.INSTANCE);
        this.workerAction = new ClassifyWorkerAction();
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

    /** Returns the combined workspace measurements of currently prepared CLS workers. */
    public WorkspaceDiagnostics workspaceDiagnostics() {
        ensureOpen();
        WorkspaceDiagnostics[] values = new WorkspaceDiagnostics[contexts.size()];
        for (int i = 0; i < contexts.size(); i++) values[i] = contexts.get(i).workspaceDiagnostics();
        return WorkspaceDiagnostics.aggregate(values);
    }

    /** Returns canonical FP32 constants materialized by the CLS model. */
    public long decodedConstantBytes() {
        ensureOpen();
        return contexts.isEmpty() ? 0L : contexts.get(0).decodedConstantBytes();
    }

    /** Returns the number of currently prepared CLS worker sessions. */
    public int preparedSessionCount() {
        ensureOpen();
        return contexts.size();
    }

    /** Classifies images concurrently while preserving input order. */
    public List<ClsClassificationResult> classifyAll(List<BgrImage> sources, int parallelism) {
        validateBatch(sources, parallelism, null);
        if (sources.isEmpty()) return Collections.emptyList();
        ClsClassificationResult[] results = new ClsClassificationResult[sources.size()];
        classifyAllInto(sources, parallelism, results);
        return Arrays.asList(results);
    }

    /** Classifies into caller-owned result storage without allocating a result list. */
    public void classifyAllInto(List<BgrImage> sources, int parallelism,
                                ClsClassificationResult[] results) {
        validateBatch(sources, parallelism, results);
        if (sources.isEmpty()) return;

        final int workers = Math.min(parallelism, sources.size());
        ensureContexts(workers);
        if (workers == 1) {
            classifyWorker(sources, results, 0, 1);
            return;
        }

        workerAction.prepare(sources, results, workers);
        try {
            parallelExecutor.run(workers, workerAction);
        } finally {
            workerAction.clear();
        }
    }

    private void validateBatch(List<BgrImage> sources, int parallelism,
                               ClsClassificationResult[] results) {
        ensureOpen();
        if (sources == null || parallelism <= 0 || parallelism > 64) {
            throw new OcrException(OcrErrorCode.INVALID_ARGUMENT,
                    "CLS sources and parallelism are invalid");
        }
        if (results != null && results.length < sources.size()) {
            throw new OcrException(OcrErrorCode.INVALID_ARGUMENT,
                    "CLS result buffer is smaller than the source batch");
        }
        for (BgrImage source : sources) {
            if (source == null) {
                throw new OcrException(OcrErrorCode.INVALID_ARGUMENT,
                        "CLS source image is required");
            }
        }
    }

    @Override
    public void close() {
        if (!closed) {
            closed = true;
            parallelExecutor.close();
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

    /** Reusable classifier batch dispatcher; one classifier owns one instance. */
    private final class ClassifyWorkerAction implements ReusableParallelExecutor.WorkerAction {
        private List<BgrImage> sources;
        private ClsClassificationResult[] results;
        private int workers;

        private void prepare(List<BgrImage> sources, ClsClassificationResult[] results,
                             int workers) {
            this.sources = sources;
            this.results = results;
            this.workers = workers;
        }

        private void clear() {
            sources = null;
            results = null;
            workers = 0;
        }

        @Override
        public void run(int workerIndex) {
            classifyWorker(sources, results, workerIndex, workers);
        }
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
