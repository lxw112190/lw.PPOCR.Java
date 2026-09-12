package io.github.lxw112190.ppocr.ppocr;

import io.github.lxw112190.ppocr.kernels.KernelBackend;
import io.github.lxw112190.ppocr.model.OcrErrorCode;
import io.github.lxw112190.ppocr.model.OcrException;
import io.github.lxw112190.ppocr.runtime.InferenceSession;
import io.github.lxw112190.ppocr.runtime.TensorShape;
import java.util.Collections;

/** Mutable, width-specialized state owned by one non-thread-safe recognizer. */
final class RecSessionContext implements AutoCloseable {
    final int width;
    final InferenceSession session;
    final RecPreprocess.Workspace preprocess;
    final float[] logits;
    final int timeSteps;

    RecSessionContext(io.github.lxw112190.ppocr.model.LwmModel model, int width, int classCount,
                      KernelBackend backend) {
        this.width = width;
        this.session = new InferenceSession(model,
                Collections.singletonList(new TensorShape(1, 3, RecPreprocess.INPUT_HEIGHT, width)), backend);
        int outputIndex = model.getGraphOutputs().get(0);
        TensorShape output = session.execution().shapes().get(outputIndex);
        if ((output.getRank() != 2 && output.getRank() != 3) ||
                output.get(output.getRank() - 1) != classCount ||
                (output.getRank() == 3 && output.get(0) != 1)) {
            session.close();
            throw new OcrException(OcrErrorCode.INVALID_MODEL, "REC output shape is incompatible with dictionary");
        }
        int resolvedTimeSteps = output.get(output.getRank() - 2);
        if (resolvedTimeSteps <= 0 || (long) resolvedTimeSteps * classCount > Integer.MAX_VALUE) {
            session.close();
            throw new OcrException(OcrErrorCode.INVALID_MODEL, "REC output time dimension is invalid");
        }
        this.timeSteps = resolvedTimeSteps;
        this.logits = new float[resolvedTimeSteps * classCount];
        this.preprocess = new RecPreprocess.Workspace(width);
    }

    @Override
    public void close() {
        session.close();
    }
}
