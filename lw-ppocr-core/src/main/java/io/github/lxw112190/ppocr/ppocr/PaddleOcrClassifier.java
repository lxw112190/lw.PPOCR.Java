package io.github.lxw112190.ppocr.ppocr;

import io.github.lxw112190.ppocr.image.BgrImage;
import io.github.lxw112190.ppocr.model.DataType;
import io.github.lxw112190.ppocr.model.LwmLoader;
import io.github.lxw112190.ppocr.model.LwmModel;
import io.github.lxw112190.ppocr.model.OcrErrorCode;
import io.github.lxw112190.ppocr.model.OcrException;
import io.github.lxw112190.ppocr.model.TensorInfo;
import io.github.lxw112190.ppocr.runtime.InferenceSession;
import io.github.lxw112190.ppocr.runtime.TensorShape;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.Collections;

/** Fixed-shape CLS facade: BGR preprocessing, scalar graph execution, and decision postprocess. */
public final class PaddleOcrClassifier implements AutoCloseable {
    private static final int[] INPUT_DIMENSIONS = {1, 3, ClsPreprocess.INPUT_HEIGHT, ClsPreprocess.INPUT_WIDTH};
    private final LwmModel model;
    private final InferenceSession session;
    private boolean closed;

    public PaddleOcrClassifier(LwmModel model) {
        if (model == null) throw new OcrException(OcrErrorCode.INVALID_ARGUMENT, "CLS model is required");
        validateModel(model);
        this.model = model;
        this.session = new InferenceSession(model,
                Collections.singletonList(new TensorShape(INPUT_DIMENSIONS)));
    }

    public static PaddleOcrClassifier load(Path path) {
        LwmModel model = LwmLoader.load(path);
        try {
            return new PaddleOcrClassifier(model);
        } catch (RuntimeException e) {
            model.close();
            throw e;
        }
    }

    public ClsClassificationResult classify(BgrImage source) {
        ensureOpen();
        ClsPreprocessResult preprocessed = ClsPreprocess.resizeNormalize(source);
        float[] probabilities = new float[2];
        session.run(preprocessed.getChw(), probabilities);
        return ClsPostprocess.decode(probabilities, preprocessed.getResizedWidth());
    }

    @Override
    public void close() {
        if (!closed) {
            closed = true;
            session.close();
            model.close();
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
}
