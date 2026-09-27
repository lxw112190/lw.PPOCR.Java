package io.github.lxw112190.ppocr.runtime;

import io.github.lxw112190.ppocr.model.LwmLoader;
import io.github.lxw112190.ppocr.model.LwmModel;
import io.github.lxw112190.ppocr.vector.VectorBackend;
import java.io.ByteArrayInputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.Collections;
import org.junit.Assert;
import org.junit.Test;

public final class ConvFusionRuntimeTest {
    @Test public void normalizationResidualsBranchesAndExposedOutputsRemainCorrect() {
        String prior = System.getProperty("lwppocr.disableConvFusion");
        try {
            for (int mode = 0; mode < 4; mode++) {
                try (LwmModel model = LwmLoader.load(new ByteArrayInputStream(model(mode)))) {
                    System.setProperty("lwppocr.disableConvFusion", "true");
                    InferenceSession reference = new InferenceSession(model, Collections.<TensorShape>emptyList(), new VectorBackend());
                    System.setProperty("lwppocr.disableConvFusion", "false");
                    try (InferenceSession fused = new InferenceSession(model, Collections.<TensorShape>emptyList(), new VectorBackend())) {
                        Assert.assertEquals(mode < 2 ? 1 : 0, fused.fusedConvCount());
                        Assert.assertEquals(0, reference.fusedConvCount());
                        if (mode < 2) {
                            Assert.assertTrue(fused.workspaceDiagnostics().getWorkspaceBytes()
                                    <= reference.workspaceDiagnostics().getWorkspaceBytes());
                            Assert.assertEquals(0, fused.execution().workspacePlan().getSize(2));
                            Assert.assertNotEquals(fused.inputView().offset(), fused.outputView().offset());
                            int intermediate = 2;
                            Assert.assertEquals(fused.execution().offset(intermediate), fused.outputView().offset());
                        }
                        float[] input = new float[128], expected = new float[128], actual = new float[128];
                        for (int i = 0; i < input.length; i++) input[i] = (i % 43 - 21) * 0.02f;
                        for (int repeat = 0; repeat < 3; repeat++) {
                            reference.run(input, expected); fused.run(input, actual);
                            Assert.assertArrayEquals("mode=" + mode, expected, actual, 0);
                        }
                    } finally { reference.close(); }
                }
            }
        } finally {
            if (prior == null) System.clearProperty("lwppocr.disableConvFusion");
            else System.setProperty("lwppocr.disableConvFusion", prior);
        }
    }

    // Modes: Conv-BN-ReLU; Conv-residual-ReLU; retained Conv branch; exposed Conv output.
    private static byte[] model(int mode) {
        boolean bn = mode == 0 || mode == 2;
        int tensorCount = bn ? (mode == 2 ? 10 : 9) : 5;
        int nodeCount = mode == 2 ? 4 : 3, outputCount = mode == 3 ? 2 : 1;
        int tensorOffset = 176, nodeOffset = tensorOffset + tensorCount * 80;
        int parameterOffset = nodeOffset + nodeCount * 72;
        int parameterSize = 64 + (bn ? 24 : 0), weightOffset = parameterOffset + parameterSize;
        int weightSize = 64 * 4 + (bn ? 4 * 8 * 4 : 0), fileSize = weightOffset + weightSize;
        ByteBuffer b = ByteBuffer.allocate(fileSize).order(ByteOrder.LITTLE_ENDIAN);
        b.put(0, (byte) 'L').put(1, (byte) 'W').put(2, (byte) 'M').put(3, (byte) '0');
        b.putShort(4, (short) 0).putShort(6, (short) 1).putInt(8, 160);
        b.putInt(12, 1).putInt(16, tensorCount).putInt(20, nodeCount).putInt(24, 1).putInt(28, outputCount);
        b.putLong(32, 160).putLong(40, 168).putLong(48, tensorOffset).putLong(56, nodeOffset);
        b.putLong(64, parameterOffset).putLong(72, parameterSize).putLong(80, weightOffset).putLong(88, 0);
        b.putLong(96, weightOffset).putLong(104, weightSize).putLong(112, fileSize);
        int finalOutput = tensorCount - 1;
        b.putInt(160, 0).putInt(168, finalOutput);
        if (mode == 3) b.putInt(172, 2);
        for (int i = 0; i < tensorCount; i++) {
            boolean constant = i == 1 || (bn && i >= 3 && i <= 6);
            int[] dims = i == 1 ? new int[] {8, 8, 1, 1}
                    : constant ? new int[] {8} : new int[] {1, 8, 1, 16};
            int at = tensorOffset + i * 80;
            b.putInt(at, 1).putInt(at + 4, dims.length);
            for (int d = 0; d < dims.length; d++) b.putInt(at + 8 + d * 4, dims[d]);
            int flags = constant ? 1 : i == 0 ? 2 : i == finalOutput || (mode == 3 && i == 2) ? 4 : 0;
            b.putInt(at + 40, flags).putLong(at + 64, -1);
            if (constant) {
                int dataOffset = i == 1 ? weightOffset : weightOffset + 256 + (i - 3) * 32;
                int count = i == 1 ? 64 : 8;
                b.putLong(at + 48, dataOffset).putLong(at + 56, count * 4);
                for (int j = 0; j < count; j++) b.putFloat(dataOffset + j * 4,
                        i == 6 ? 1.0f + j * 0.01f : (j % 11 - 5) * 0.03f);
            }
        }
        node(b, nodeOffset, 1, new int[] {0, 1}, 2, parameterOffset, 64);
        b.putInt(parameterOffset, 1).putInt(parameterOffset + 4, 1);
        for (int at = 8; at <= 28; at += 4) b.putInt(parameterOffset + at, 1);
        if (bn) {
            node(b, nodeOffset + 72, 7, new int[] {2, 3, 4, 5, 6}, 7, parameterOffset + 64, 24);
            b.putInt(parameterOffset + 64, 1).putFloat(parameterOffset + 68, 0.00001f);
            node(b, nodeOffset + 144, 9, new int[] {7}, 8, 0, 0);
            if (mode == 2) node(b, nodeOffset + 216, 2, new int[] {2, 8}, 9, 0, 0);
        } else {
            node(b, nodeOffset + 72, 2, new int[] {2, 0}, 3, 0, 0);
            node(b, nodeOffset + 144, 9, new int[] {3}, 4, 0, 0);
        }
        long checksum = 0xcbf29ce484222325L;
        for (int i = 0; i < fileSize; i++) { checksum ^= b.get(i) & 255; checksum *= 0x100000001b3L; }
        b.putLong(128, checksum);
        return b.array();
    }

    private static void node(ByteBuffer b, int at, int op, int[] inputs, int output, int parameters, int bytes) {
        b.putShort(at, (short) op).putShort(at + 2, (short) inputs.length).putShort(at + 4, (short) 1);
        for (int i = 0; i < inputs.length; i++) b.putInt(at + 8 + i * 4, inputs[i]);
        b.putInt(at + 40, output).putLong(at + 56, parameters).putLong(at + 64, bytes);
    }
}
