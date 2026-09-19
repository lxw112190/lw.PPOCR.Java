package io.github.lxw112190.ppocr.ppocr;

import io.github.lxw112190.ppocr.image.BgrImage;
import io.github.lxw112190.ppocr.kernels.KernelBackend;
import io.github.lxw112190.ppocr.model.LwmModel;
import io.github.lxw112190.ppocr.runtime.InferenceSession;
import io.github.lxw112190.ppocr.runtime.FloatTensorView;
import io.github.lxw112190.ppocr.runtime.TensorShape;
import io.github.lxw112190.ppocr.runtime.WorkspaceDiagnostics;
import java.util.Collections;

/** Mutable fixed-shape state owned by one CLS worker. */
final class ClsSessionContext implements AutoCloseable {
    private static final int[] INPUT_DIMENSIONS = {
            1, 3, ClsPreprocess.INPUT_HEIGHT, ClsPreprocess.INPUT_WIDTH
    };

    private final InferenceSession session;

    ClsSessionContext(LwmModel model, KernelBackend backend) {
        this.session = new InferenceSession(model,
                Collections.singletonList(new TensorShape(INPUT_DIMENSIONS)), backend);
    }

    ClsClassificationResult classify(BgrImage source) {
        FloatTensorView input = session.inputView();
        ClsPreprocess.resizeNormalizeInto(source, input.array(), input.offset());
        session.runBound();
        FloatTensorView output = session.outputView();
        int resizedWidth = (int) Math.min(ClsPreprocess.INPUT_WIDTH,
                ((long) ClsPreprocess.INPUT_HEIGHT * source.width()
                        + source.height() - 1L) / source.height());
        return ClsPostprocess.decode(output.array(), output.offset(), output.length(), resizedWidth);
    }

    WorkspaceDiagnostics workspaceDiagnostics() {
        return session.workspaceDiagnostics();
    }

    long decodedConstantBytes() {
        return session.decodedConstantBytes();
    }

    @Override
    public void close() {
        session.close();
    }
}
