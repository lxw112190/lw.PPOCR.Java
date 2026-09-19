package io.github.lxw112190.ppocr.ppocr;

import io.github.lxw112190.ppocr.kernels.KernelBackend;
import io.github.lxw112190.ppocr.model.OcrErrorCode;
import io.github.lxw112190.ppocr.model.OcrException;
import io.github.lxw112190.ppocr.runtime.InferenceSession;
import io.github.lxw112190.ppocr.runtime.CtcProjectionSession;
import io.github.lxw112190.ppocr.runtime.FloatTensorView;
import io.github.lxw112190.ppocr.runtime.TensorShape;
import io.github.lxw112190.ppocr.runtime.WorkspaceDiagnostics;
import java.util.Collections;

/** Mutable, width-specialized state owned by one non-thread-safe recognizer. */
final class RecSessionContext implements AutoCloseable {
    final int width;
    final InferenceSession session;
    final CtcProjectionSession projectionSession;
    final CompactCtcOutput compactOutput;
    final int timeSteps;
    private final StringBuilder textScratch = new StringBuilder(128);
    private int resizedWidth;

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
        this.compactOutput = compactSession == null ? null : new CompactCtcOutput(resolvedTimeSteps);
    }

    void preprocess(io.github.lxw112190.ppocr.image.BgrImage source) {
        FloatTensorView input = projectionSession != null
                ? projectionSession.inputView() : session.inputView();
        RecPreprocess.resizeNormalizeInto(source, width, input.array(), input.offset());
        long scaledWidth = (long) RecPreprocess.INPUT_HEIGHT * source.width();
        resizedWidth = (int) Math.min(width,
                (scaledWidth + source.height() - 1L) / source.height());
    }

    CtcDecodeResult run(PaddleOcrDictionary dictionary) {
        if (projectionSession != null) {
            projectionSession.runBound(compactOutput.classIds(), compactOutput.logits(),
                    compactOutput.probabilities());
            return CtcDecoder.decodeGreedy(compactOutput, dictionary);
        }
        session.runBound();
        FloatTensorView output = session.outputView();
        return CtcDecoder.decodeGreedy(output.array(), output.offset(), timeSteps,
                dictionary.classCount(), dictionary);
    }

    String runInto(PaddleOcrDictionary dictionary, float[] scores, int scoreIndex,
                   int[] emittedCounts, int emittedIndex) {
        if (projectionSession != null) {
            projectionSession.runBound(compactOutput.classIds(), compactOutput.logits(),
                    compactOutput.probabilities());
            return CtcDecoder.decodeGreedyInto(compactOutput, dictionary,
                    scores, scoreIndex, emittedCounts, emittedIndex, textScratch);
        }
        session.runBound();
        FloatTensorView output = session.outputView();
        return CtcDecoder.decodeGreedyInto(output.array(), output.offset(), timeSteps,
                dictionary.classCount(), dictionary, scores, scoreIndex,
                emittedCounts, emittedIndex, textScratch);
    }

    int resizedWidth() { return resizedWidth; }

    boolean usesProjectionFusion() { return projectionSession != null; }

    WorkspaceDiagnostics workspaceDiagnostics() {
        return projectionSession != null
                ? projectionSession.workspaceDiagnostics() : session.workspaceDiagnostics();
    }

    long decodedConstantBytes() {
        return projectionSession != null
                ? projectionSession.getDecodedConstantBytes()
                : session.decodedConstantBytes();
    }

    long packedWeightBytes() {
        return projectionSession == null ? 0L : projectionSession.getPackedWeightBytes();
    }

    @Override
    public void close() {
        closeSessions();
    }

    private void closeSessions() {
        if (projectionSession != null) projectionSession.close();
        if (session != null) session.close();
    }
}
