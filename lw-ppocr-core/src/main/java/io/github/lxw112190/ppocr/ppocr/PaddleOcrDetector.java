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
import java.util.List;

/** DET facade with static compatibility and cached C-compatible dynamic shapes. */
public final class PaddleOcrDetector implements AutoCloseable {
    private static final int DEFAULT_DYNAMIC_LIMIT_SIDE = 960;
    private static final int DYNAMIC_CACHE_CAPACITY = 3;

    private final LwmModel model;
    private final int fixedInputHeight;
    private final int fixedInputWidth;
    private final boolean dynamicInput;
    private final int maximumSideLength;
    private final KernelBackend backend;
    private final DetSessionCache sessions;
    private boolean closed;

    /** Uses C-compatible defaults: bitmap 0.3, box 0.6, unclip 1.6, no dilation. */
    public List<DetectionBox> detect(BgrImage source) {
        return detect(source, 0.3f, 0.6f, 1.6f, false, 1000);
    }

    public List<DetectionBox> detect(BgrImage source, float bitmapThreshold, float boxThreshold,
                                     float unclipRatio, boolean useDilation, int maxCandidates) {
        ensureOpen();
        DetSessionContext context = contextFor(source);
        context.preprocess.resizeNormalize(source);
        context.session.run(context.preprocess.getChw(), context.probabilityMap);
        return context.postprocessor.decodeToSource(
                context.probabilityMap, bitmapThreshold, boxThreshold,
                context.preprocess.getWidthRatio(), context.preprocess.getHeightRatio(),
                maxCandidates, unclipRatio, useDilation, source.width(), source.height());
    }

    public PaddleOcrDetector(LwmModel model) {
        this(model, DEFAULT_DYNAMIC_LIMIT_SIDE, new ScalarBackend());
    }

    /**
     * Creates a detector with an explicit maximum side length for dynamic models.
     * Static models keep their declared dimensions; the value is retained for API symmetry.
     */
    public PaddleOcrDetector(LwmModel model, int maximumSideLength) {
        this(model, maximumSideLength, new ScalarBackend());
    }

    /** Creates a detector using the supplied stateless kernel backend. */
    public PaddleOcrDetector(LwmModel model, int maximumSideLength, KernelBackend backend) {
        if (model == null || backend == null) {
            throw new OcrException(OcrErrorCode.INVALID_ARGUMENT, "DET model and backend are required");
        }
        if (maximumSideLength < 32) {
            throw new OcrException(OcrErrorCode.INVALID_ARGUMENT,
                    "DET maximum side length must be at least 32");
        }
        int inputIndex = model.getGraphInputs().size() == 1 ? model.getGraphInputs().get(0) : -1;
        int outputIndex = model.getGraphOutputs().size() == 1 ? model.getGraphOutputs().get(0) : -1;
        if (inputIndex < 0 || outputIndex < 0) throw invalid("DET model must have one input and one output");
        TensorInfo input = model.getTensors().get(inputIndex);
        int[] inputDimensions = input.getDimensions();
        if (input.getDataType() != DataType.F32 || inputDimensions.length != 4 ||
                (inputDimensions[0] != -1 && inputDimensions[0] != 1) ||
                inputDimensions[1] != 3 || !validDetectorDimension(inputDimensions[2]) ||
                !validDetectorDimension(inputDimensions[3])) {
            throw invalid("DET input must be FP32 [1,3,H,W] with 32-pixel dimensions");
        }
        TensorInfo output = model.getTensors().get(outputIndex);
        int[] outputDimensions = output.getDimensions();
        if (output.getDataType() != DataType.F32 || outputDimensions.length < 2) {
            throw invalid("DET output must be an FP32 probability map");
        }
        for (int axis = 0; axis < outputDimensions.length - 2; axis++) {
            if (outputDimensions[axis] != -1 && outputDimensions[axis] != 1) {
                throw invalid("DET probability map has unsupported leading dimensions");
            }
        }
        boolean dynamicInput = inputDimensions[2] == -1 || inputDimensions[3] == -1;
        int resolvedInputHeight = dynamicInput ? 0 : inputDimensions[2];
        int resolvedInputWidth = dynamicInput ? 0 : inputDimensions[3];
        this.model = model;
        this.fixedInputHeight = resolvedInputHeight;
        this.fixedInputWidth = resolvedInputWidth;
        this.dynamicInput = dynamicInput;
        this.maximumSideLength = dynamicInput ? maximumSideLength : 0;
        this.backend = backend;
        this.sessions = new DetSessionCache(DYNAMIC_CACHE_CAPACITY);
        if (!dynamicInput) {
            try {
                sessions.getOrCreate(new DetShapeKey(fixedInputHeight, fixedInputWidth), model, backend);
            } catch (RuntimeException e) {
                model.close();
                throw e;
            }
        }
    }

    public static PaddleOcrDetector load(Path path) {
        return load(path, DEFAULT_DYNAMIC_LIMIT_SIDE, new ScalarBackend());
    }

    /** Loads a detector using an explicit dynamic-shape limit and stateless backend. */
    public static PaddleOcrDetector load(Path path, int maximumSideLength,
                                         KernelBackend backend) {
        LwmModel model = LwmLoader.load(path);
        try {
            return new PaddleOcrDetector(model, maximumSideLength, backend);
        } catch (RuntimeException e) {
            model.close();
            throw e;
        }
    }

    @Override
    public void close() {
        if (!closed) {
            closed = true;
            sessions.close();
            model.close();
        }
    }

    private static OcrException invalid(String message) {
        return new OcrException(OcrErrorCode.INVALID_MODEL, message);
    }

    private static boolean validDetectorDimension(int dimension) {
        return dimension == -1 || (dimension >= 32 && dimension % 32 == 0);
    }

    private void ensureOpen() {
        if (closed) throw new OcrException(OcrErrorCode.INVALID_ARGUMENT, "DET detector is closed");
    }

    private DetSessionContext contextFor(BgrImage source) {
        if (source == null) throw new OcrException(OcrErrorCode.INVALID_ARGUMENT, "source image is required");
        if (!dynamicInput) return sessions.get(new DetShapeKey(fixedInputHeight, fixedInputWidth));
        DetInputShape shape = DetInputShapePolicy.choose(source, maximumSideLength);
        return sessions.getOrCreate(new DetShapeKey(shape.getInputHeight(), shape.getInputWidth()), model, backend);
    }
}
