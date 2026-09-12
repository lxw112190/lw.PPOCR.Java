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

/** Fixed-width REC facade: BGR preprocessing, scalar graph execution, and CTC decoding. */
public final class PaddleOcrRecognizer implements AutoCloseable {
    private final LwmModel model;
    private final PaddleOcrDictionary dictionary;
    private final InferenceSession session;
    private final int timeSteps;
    private boolean closed;

    /** Takes ownership of the model and dictionary and closes both on close(). */
    public PaddleOcrRecognizer(LwmModel model, PaddleOcrDictionary dictionary) {
        if (model == null || dictionary == null) {
            throw new OcrException(OcrErrorCode.INVALID_ARGUMENT, "REC model and dictionary are required");
        }
        int inputIndex = validateModel(model, dictionary);
        TensorInfo input = model.getTensors().get(inputIndex);
        TensorShape inputShape = new TensorShape(input.getDimensions());
        this.model = model;
        this.dictionary = dictionary;
        this.session = new InferenceSession(model, Collections.singletonList(inputShape));
        this.timeSteps = outputTimeSteps(model, dictionary);
    }

    public static PaddleOcrRecognizer load(Path modelPath, Path dictionaryPath) {
        LwmModel model = LwmLoader.load(modelPath);
        PaddleOcrDictionary dictionary = null;
        try {
            dictionary = PaddleOcrDictionary.load(dictionaryPath);
            return new PaddleOcrRecognizer(model, dictionary);
        } catch (RuntimeException e) {
            if (dictionary != null) dictionary.close();
            model.close();
            throw e;
        }
    }

    public RecRecognitionResult recognize(BgrImage source) {
        ensureOpen();
        TensorInfo input = model.getTensors().get(model.getGraphInputs().get(0));
        int targetWidth = input.getDimensions()[3];
        RecPreprocessResult preprocessed = RecPreprocess.resizeNormalize(source, targetWidth);
        float[] logits = new float[timeSteps * dictionary.classCount()];
        session.run(preprocessed.getChw(), logits);
        CtcDecodeResult decoded = CtcDecoder.decodeGreedy(logits, timeSteps,
                dictionary.classCount(), dictionary);
        return new RecRecognitionResult(decoded.getText(), decoded.getScore(),
                decoded.getEmittedCount(), preprocessed.getResizedWidth());
    }

    @Override
    public void close() {
        if (!closed) {
            closed = true;
            session.close();
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
        if (input.getDataType() != DataType.F32 || dimensions.length != 4 || dimensions[0] != 1 ||
                dimensions[1] != 3 || dimensions[2] != RecPreprocess.INPUT_HEIGHT || dimensions[3] <= 0) {
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
                (outputDimensions.length == 3 && outputDimensions[0] != 1) ||
                outputDimensions[outputDimensions.length - 2] <= 0) {
            throw invalid("REC output class count does not match dictionary");
        }
        return model.getGraphInputs().get(0);
    }

    private static int outputTimeSteps(LwmModel model, PaddleOcrDictionary dictionary) {
        int[] dimensions = model.getTensors().get(model.getGraphOutputs().get(0)).getDimensions();
        int timeSteps = dimensions[dimensions.length - 2];
        if (timeSteps <= 0 || dimensions[dimensions.length - 1] != dictionary.classCount()) {
            throw invalid("REC output dimensions are invalid");
        }
        return timeSteps;
    }

    private static OcrException invalid(String message) {
        return new OcrException(OcrErrorCode.INVALID_MODEL, message);
    }

    private void ensureOpen() {
        if (closed) throw new OcrException(OcrErrorCode.INVALID_ARGUMENT, "REC recognizer is closed");
    }
}
