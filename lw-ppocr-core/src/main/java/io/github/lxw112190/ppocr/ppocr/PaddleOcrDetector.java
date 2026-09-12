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
import java.util.Collections;
import java.util.List;

/** Fixed-shape DET facade: preprocessing, LWM execution, and DB postprocess. */
public final class PaddleOcrDetector implements AutoCloseable {
    private static final int DEFAULT_DYNAMIC_INPUT_SIZE = 320;

    private final LwmModel model;
    private final InferenceSession session;
    private final int inputHeight;
    private final int inputWidth;
    private final int mapHeight;
    private final int mapWidth;
    private final DetPreprocess.Workspace preprocess;
    private final DbPostprocess.Decoder postprocessor;
    private final float[] probabilities;
    private boolean closed;

    /** Uses C-compatible defaults: bitmap 0.3, box 0.6, unclip 1.6, no dilation. */
    public List<DetectionBox> detect(BgrImage source) {
        return detect(source, 0.3f, 0.6f, 1.6f, false, 1000);
    }

    public List<DetectionBox> detect(BgrImage source, float bitmapThreshold, float boxThreshold,
                                     float unclipRatio, boolean useDilation, int maxCandidates) {
        ensureOpen();
        preprocess.resizeNormalize(source);
        session.run(preprocess.getChw(), probabilities);
        return postprocessor.decodeToSource(probabilities, bitmapThreshold, boxThreshold,
                preprocess.getWidthRatio(), preprocess.getHeightRatio(),
                maxCandidates, unclipRatio, useDilation, source.width(), source.height());
    }

    public PaddleOcrDetector(LwmModel model) {
        if (model == null) throw new OcrException(OcrErrorCode.INVALID_ARGUMENT, "DET model is required");
        int inputIndex = model.getGraphInputs().size() == 1 ? model.getGraphInputs().get(0) : -1;
        int outputIndex = model.getGraphOutputs().size() == 1 ? model.getGraphOutputs().get(0) : -1;
        if (inputIndex < 0 || outputIndex < 0) throw invalid("DET model must have one input and one output");
        TensorInfo input = model.getTensors().get(inputIndex);
        int[] inputDimensions = input.getDimensions();
        if (input.getDataType() != DataType.F32 || inputDimensions.length != 4 || inputDimensions[0] != 1 ||
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
        int resolvedInputHeight = inputDimensions[2] == -1 ? DEFAULT_DYNAMIC_INPUT_SIZE : inputDimensions[2];
        int resolvedInputWidth = inputDimensions[3] == -1 ? DEFAULT_DYNAMIC_INPUT_SIZE : inputDimensions[3];
        this.model = model;
        try {
            this.session = new InferenceSession(model,
                    Collections.singletonList(new TensorShape(1, 3, resolvedInputHeight, resolvedInputWidth)));
        } catch (RuntimeException e) {
            model.close();
            throw e;
        }
        TensorShape resolvedOutput = this.session.execution().shapes().get(outputIndex);
        if (resolvedOutput.getRank() < 2) {
            this.session.close();
            model.close();
            throw invalid("DET probability map rank is invalid");
        }
        for (int axis = 0; axis < resolvedOutput.getRank() - 2; axis++) {
            if (resolvedOutput.get(axis) != 1) {
                this.session.close();
                model.close();
                throw invalid("DET probability map has unsupported leading dimensions");
            }
        }
        int mapHeight = resolvedOutput.get(resolvedOutput.getRank() - 2);
        int mapWidth = resolvedOutput.get(resolvedOutput.getRank() - 1);
        if (mapHeight <= 0 || mapWidth <= 0 || (long) mapHeight * mapWidth > Integer.MAX_VALUE) {
            this.session.close();
            model.close();
            throw invalid("DET probability map dimensions are invalid");
        }
        this.inputHeight = resolvedInputHeight;
        this.inputWidth = resolvedInputWidth;
        this.mapHeight = mapHeight;
        this.mapWidth = mapWidth;
        this.preprocess = new DetPreprocess.Workspace(inputWidth, inputHeight);
        this.postprocessor = DbPostprocess.createDecoder(mapWidth, mapHeight);
        this.probabilities = new float[mapHeight * mapWidth];
    }

    public static PaddleOcrDetector load(Path path) {
        LwmModel model = LwmLoader.load(path);
        try {
            return new PaddleOcrDetector(model);
        } catch (RuntimeException e) {
            model.close();
            throw e;
        }
    }

    @Override
    public void close() {
        if (!closed) {
            closed = true;
            session.close();
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
}
