package io.github.lxw112190.ppocr.runtime;

import io.github.lxw112190.ppocr.model.LwmLoader;
import io.github.lxw112190.ppocr.model.LwmModel;
import io.github.lxw112190.ppocr.model.OcrErrorCode;
import io.github.lxw112190.ppocr.model.OcrException;
import java.io.ByteArrayInputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import org.junit.Assert;
import org.junit.Test;

public final class InferenceSessionTest {
    @Test
    public void executesScalarAddWithConstant() {
        LwmModel model = LwmLoader.load(new ByteArrayInputStream(addModel()));
        InferenceSession session = new InferenceSession(model);
        float[] output = new float[4];
        session.run(new float[] {1, 2, 3, 4}, output);
        Assert.assertArrayEquals(new float[] {2, 4, 6, 8}, output, 0.0f);
    }

    @Test
    public void runsDirectlyThroughBoundTensorViews() {
        LwmModel model = LwmLoader.load(new ByteArrayInputStream(addModel()));
        InferenceSession session = new InferenceSession(model);
        FloatTensorView input = session.inputView();
        FloatTensorView output = session.outputView();
        Assert.assertSame(input.array(), session.inputView().array());
        Assert.assertSame(output.array(), session.outputView().array());
        Assert.assertEquals("same-shape ADD should reuse the single-use input range",
                input.offset(), output.offset());
        Assert.assertEquals(input.length(), output.length());
        System.arraycopy(new float[] {1, 2, 3, 4}, 0,
                input.array(), input.offset(), input.length());
        session.runBound();
        float[] actual = new float[output.length()];
        System.arraycopy(output.array(), output.offset(), actual, 0, output.length());
        Assert.assertArrayEquals(new float[] {2, 4, 6, 8}, actual, 0.0f);
        session.close();
        model.close();
    }

    @Test
    public void reportsZeroInputNodeAsUnsupportedModel() {
        byte[] bytes = addModel();
        ByteBuffer buffer = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN);
        buffer.putShort(418, (short) 0);
        buffer.putInt(428, 0);
        buffer.putLong(128, 0);
        buffer.putLong(128, fnv1a(bytes));
        LwmModel model = LwmLoader.load(new ByteArrayInputStream(bytes));
        InferenceSession session = new InferenceSession(model);
        try {
            session.run(new float[] {1, 2, 3, 4}, new float[4]);
            Assert.fail("expected unsupported model");
        } catch (OcrException e) {
            Assert.assertEquals(OcrErrorCode.UNSUPPORTED_OPERATOR, e.getCode());
        }
    }

    @Test
    public void profilesOperatorsOnlyInsideExplicitScope() {
        LwmModel model = LwmLoader.load(new ByteArrayInputStream(addModel()));
        InferenceSession session = new InferenceSession(model);
        InferenceProfiler.Profile profile;
        try (InferenceProfiler profiler = InferenceProfiler.start()) {
            session.run(new float[] {1, 2, 3, 4}, new float[4]);
            profile = profiler.snapshot();
        }
        Assert.assertEquals(1L, profile.getInvocations(io.github.lxw112190.ppocr.model.OperatorType.ADD));
        Assert.assertTrue(profile.getElapsedNanos(io.github.lxw112190.ppocr.model.OperatorType.ADD) >= 0L);
        Assert.assertEquals(profile.getElapsedNanos(io.github.lxw112190.ppocr.model.OperatorType.ADD),
                profile.getTotalElapsedNanos());
    }

    @Test
    public void sharesDecodedConstantsAcrossDynamicSessions() {
        LwmModel model = LwmLoader.load(new ByteArrayInputStream(addModel()));
        InferenceSession first = new InferenceSession(model);
        InferenceSession second = new InferenceSession(model);
        Assert.assertSame(first.execution().constant(1), second.execution().constant(1));
        first.close();
        second.close();
        model.close();
    }

    @Test
    public void materializesConstantsOnlyWhenFirstRead() {
        LwmModel model = LwmLoader.load(new ByteArrayInputStream(addModel()));
        CompiledModel compiled = CompiledModel.acquire(model);
        Assert.assertFalse(compiled.isConstantMaterialized(1));
        float[] first = compiled.constant(1);
        Assert.assertTrue(compiled.isConstantMaterialized(1));
        Assert.assertSame(first, compiled.constant(1));
        model.close();
    }

    @Test
    public void sharesCompiledModelAcrossDynamicSessions() {
        LwmModel model = LwmLoader.load(new ByteArrayInputStream(addModel()));
        InferenceSession first = new InferenceSession(model);
        InferenceSession second = new InferenceSession(model);
        Assert.assertSame(first.execution().compiledModel(), second.execution().compiledModel());
        Assert.assertSame(first.execution().compiledModel().parameterData(0),
                second.execution().compiledModel().parameterData(0));
        Assert.assertSame(first.execution().compiledModel().nodeInputs(0),
                second.execution().compiledModel().nodeInputs(0));
        CompiledModel compiled = first.execution().compiledModel();
        Assert.assertEquals(1, compiled.tensorConsumerCount(0));
        Assert.assertEquals(0, compiled.tensorLastUse(0));
        Assert.assertEquals(1, compiled.tensorConsumerCount(1));
        Assert.assertEquals(0, compiled.tensorLastUse(1));
        Assert.assertEquals(0, compiled.tensorConsumerCount(2));
        Assert.assertEquals(1, compiled.tensorLastUse(2));
        first.close();
        second.close();
        model.close();
    }

    private static byte[] addModel() {
        final int inputOffset = 160;
        final int outputOffset = 168;
        final int tensorOffset = 176;
        final int nodeOffset = tensorOffset + 3 * 80;
        final int parameterOffset = nodeOffset + 72;
        final int fileSize = parameterOffset + 24;
        ByteBuffer b = ByteBuffer.allocate(fileSize).order(ByteOrder.LITTLE_ENDIAN);
        b.put(0, (byte) 'L').put(1, (byte) 'W').put(2, (byte) 'M').put(3, (byte) '0');
        b.putShort(4, (short) 0).putShort(6, (short) 1).putInt(8, 160);
        b.putInt(12, 1).putInt(16, 3).putInt(20, 1).putInt(24, 1).putInt(28, 1);
        b.putLong(32, inputOffset).putLong(40, outputOffset).putLong(48, tensorOffset);
        b.putLong(56, nodeOffset).putLong(64, parameterOffset).putLong(72, 0);
        b.putLong(80, parameterOffset).putLong(88, 0).putLong(96, parameterOffset);
        b.putLong(104, 24).putLong(112, fileSize).putLong(120, 0).putLong(128, 0);
        b.putInt(inputOffset, 0).putInt(outputOffset, 2);
        putTensor(b, tensorOffset, 2, 0, 0);
        putTensor(b, tensorOffset + 80, 1, parameterOffset, 16);
        putTensor(b, tensorOffset + 160, 4, 0, 0);
        b.putShort(nodeOffset, (short) 2).putShort(nodeOffset + 2, (short) 2).putShort(nodeOffset + 4, (short) 1);
        b.putInt(nodeOffset + 8, 0).putInt(nodeOffset + 12, 1).putInt(nodeOffset + 40, 2);
        b.putFloat(parameterOffset, 1).putFloat(parameterOffset + 4, 2);
        b.putFloat(parameterOffset + 8, 3).putFloat(parameterOffset + 12, 4);
        b.putLong(128, fnv1a(b.array()));
        return b.array();
    }

    private static void putTensor(ByteBuffer b, int offset, int flags, int dataOffset, int dataSize) {
        b.putInt(offset, 1).putInt(offset + 4, 1).putInt(offset + 8, 4);
        b.putInt(offset + 40, flags).putLong(offset + 48, dataOffset).putLong(offset + 56, dataSize);
        b.putLong(offset + 64, 0xffffffffffffffffL);
    }

    private static long fnv1a(byte[] bytes) {
        long value = 0xcbf29ce484222325L;
        for (int i = 0; i < bytes.length; i++) {
            int valueByte = i >= 128 && i < 136 ? 0 : bytes[i] & 0xff;
            value ^= valueByte;
            value *= 0x100000001b3L;
        }
        return value;
    }
}
