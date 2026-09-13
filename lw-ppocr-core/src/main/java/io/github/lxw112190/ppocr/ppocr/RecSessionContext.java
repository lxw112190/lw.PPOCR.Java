package io.github.lxw112190.ppocr.ppocr;

import io.github.lxw112190.ppocr.kernels.KernelBackend;
import io.github.lxw112190.ppocr.model.OcrErrorCode;
import io.github.lxw112190.ppocr.model.OcrException;
import io.github.lxw112190.ppocr.runtime.InferenceSession;
import io.github.lxw112190.ppocr.runtime.CtcProjectionSession;
import io.github.lxw112190.ppocr.runtime.TensorShape;
import java.util.Collections;

/** Mutable, width-specialized state owned by one non-thread-safe recognizer. */
final class RecSessionContext implements AutoCloseable {
    final int width;
    final InferenceSession session;
    final CtcProjectionSession projectionSession;
    final RecPreprocess.Workspace preprocess;
    final float[] logits;
    final CompactCtcOutput compactOutput;
    final int timeSteps;

    RecSessionContext(io.github.lxw112190.ppocr.model.LwmModel model, int width, int classCount,
                      KernelBackend backend) {
        this.width = width;
        java.util.List<TensorShape> inputShapes = Collections.singletonList(
                new TensorShape(1, 3, RecPreprocess.INPUT_HEIGHT, width));
        CtcProjectionSession compactSession = CtcProjectionSession.tryCreate(model,
                inputShapes, backend, classCount);
        this.projectionSession = compactSession;
        this.session = compactSession == null ? new InferenceSession(model, inputShapes, backend) : null;
        int resolvedTimeSteps;
        if (compactSession != null) {
            resolvedTimeSteps = compactSession.getTimeSteps();
        } else {
            int outputIndex = model.getGraphOutputs().get(0);
            TensorShape output = session.execution().shapes().get(outputIndex);
            if ((output.getRank() != 2 && output.getRank() != 3) ||
                    output.get(output.getRank() - 1) != classCount ||
                    (output.getRank() == 3 && output.get(0) != 1)) {
                closeSessions();
                throw new OcrException(OcrErrorCode.INVALID_MODEL,
                        "REC output shape is incompatible with dictionary");
            }
            resolvedTimeSteps = output.get(output.getRank() - 2);
        }
        if (resolvedTimeSteps <= 0 || (long) resolvedTimeSteps * classCount > Integer.MAX_VALUE) {
            closeSessions();
            throw new OcrException(OcrErrorCode.INVALID_MODEL, "REC output time dimension is invalid");
        }
        this.timeSteps = resolvedTimeSteps;
        this.logits = compactSession == null ? new float[resolvedTimeSteps * classCount] : null;
        this.compactOutput = compactSession == null ? null : new CompactCtcOutput(resolvedTimeSteps);
        this.preprocess = new RecPreprocess.Workspace(width);
    }

    CtcDecodeResult run(float[] input, PaddleOcrDictionary dictionary) {
        if (projectionSession != null) {
            projectionSession.run(input, compactOutput.classIds(), compactOutput.logits(),
                    compactOutput.probabilities());
            return CtcDecoder.decodeGreedy(compactOutput, dictionary);
        }
        session.run(input, logits);
        return CtcDecoder.decodeGreedy(logits, timeSteps, dictionary.classCount(), dictionary);
    }

    boolean usesProjectionFusion() { return projectionSession != null; }

    @Override
    public void close() {
        closeSessions();
    }

    private void closeSessions() {
        if (projectionSession != null) projectionSession.close();
        if (session != null) session.close();
    }
}
