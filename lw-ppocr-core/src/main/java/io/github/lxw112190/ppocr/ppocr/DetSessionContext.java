package io.github.lxw112190.ppocr.ppocr;

import io.github.lxw112190.ppocr.kernels.KernelBackend;
import io.github.lxw112190.ppocr.model.LwmModel;
import io.github.lxw112190.ppocr.runtime.InferenceSession;
import io.github.lxw112190.ppocr.runtime.TensorShape;
import java.util.Collections;

/** Prepared DET graph, preprocessing workspace, and output buffer for one shape. */
final class DetSessionContext implements AutoCloseable {
    final int width;
    final int height;
    final InferenceSession session;
    final DetPreprocess.Workspace preprocess;
    final DbPostprocess.Decoder postprocessor;
    final float[] probabilityMap;

    DetSessionContext(LwmModel model, int width, int height, KernelBackend backend) {
        this.width = width;
        this.height = height;
        InferenceSession prepared = new InferenceSession(model,
                Collections.singletonList(new TensorShape(1, 3, height, width)), backend);
        try {
            TensorShape output = prepared.execution().shapes().get(
                    model.getGraphOutputs().get(0));
            int rank = output.getRank();
            if (rank < 2) throw new IllegalArgumentException("DET output rank is invalid");
            for (int axis = 0; axis < rank - 2; axis++) {
                if (output.get(axis) != 1) throw new IllegalArgumentException(
                        "DET output has unsupported leading dimensions");
            }
            int mapHeight = output.get(rank - 2);
            int mapWidth = output.get(rank - 1);
            if (mapHeight <= 0 || mapWidth <= 0 || (long) mapHeight * mapWidth > Integer.MAX_VALUE) {
                throw new IllegalArgumentException("DET output dimensions are invalid");
            }
            this.session = prepared;
            this.preprocess = new DetPreprocess.Workspace(width, height);
            this.postprocessor = DbPostprocess.createDecoder(mapWidth, mapHeight);
            this.probabilityMap = new float[mapHeight * mapWidth];
        } catch (RuntimeException e) {
            prepared.close();
            throw e;
        }
    }

    @Override
    public void close() { session.close(); }
}
