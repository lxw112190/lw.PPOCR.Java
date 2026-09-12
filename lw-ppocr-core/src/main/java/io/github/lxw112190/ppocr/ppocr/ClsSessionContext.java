package io.github.lxw112190.ppocr.ppocr;

import io.github.lxw112190.ppocr.image.BgrImage;
import io.github.lxw112190.ppocr.kernels.KernelBackend;
import io.github.lxw112190.ppocr.model.LwmModel;
import io.github.lxw112190.ppocr.runtime.InferenceSession;
import io.github.lxw112190.ppocr.runtime.TensorShape;
import java.util.Collections;

/** Mutable fixed-shape state owned by one CLS worker. */
final class ClsSessionContext implements AutoCloseable {
    private static final int[] INPUT_DIMENSIONS = {
            1, 3, ClsPreprocess.INPUT_HEIGHT, ClsPreprocess.INPUT_WIDTH
    };

    private final InferenceSession session;
    private final float[] probabilities = new float[2];
    private final ClsPreprocess.Workspace preprocess = new ClsPreprocess.Workspace();

    ClsSessionContext(LwmModel model, KernelBackend backend) {
        this.session = new InferenceSession(model,
                Collections.singletonList(new TensorShape(INPUT_DIMENSIONS)), backend);
    }

    ClsClassificationResult classify(BgrImage source) {
        preprocess.resizeNormalize(source);
        session.run(preprocess.getChw(), probabilities);
        return ClsPostprocess.decode(probabilities, preprocess.getResizedWidth());
    }

    @Override
    public void close() {
        session.close();
    }
}
