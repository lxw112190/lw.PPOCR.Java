package io.github.lxw112190.ppocr.model;

import java.io.ByteArrayInputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import org.junit.Assert;
import org.junit.Test;

/** Loader contract tests backed by a tiny in-memory LWM fixture. */
public final class LwmLoaderSelfTest {
    @Test
    public void loadsMinimalModel() {
        LwmModel model = LwmLoader.load(new ByteArrayInputStream(minimalModel()));
        Assert.assertEquals(1, model.getHeader().getFormatMinor());
        Assert.assertEquals(1, model.getTensors().size());
        Assert.assertEquals(Integer.valueOf(0), model.getGraphInputs().get(0));
        Assert.assertEquals(Integer.valueOf(0), model.getGraphOutputs().get(0));
    }

    @Test
    public void rejectsChecksumMismatch() {
        try {
            LwmLoader.load(new ByteArrayInputStream(corrupt(minimalModel())));
            Assert.fail("expected checksum mismatch");
        } catch (OcrException e) {
            Assert.assertEquals(OcrErrorCode.CHECKSUM_MISMATCH, e.getCode());
        }
    }

    @Test
    public void rejectsClosedModel() {
        LwmModel model = LwmLoader.load(new ByteArrayInputStream(minimalModel()));
        model.close();
        try {
            model.getHeader();
            Assert.fail("expected closed model failure");
        } catch (OcrException e) {
            Assert.assertEquals(OcrErrorCode.INVALID_ARGUMENT, e.getCode());
        }
    }

    @Test
    public void rejectsUnsignedHeaderCountAboveLimit() {
        byte[] bytes = minimalModel();
        ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN).putInt(16, 0x80000000);
        rewriteChecksum(bytes);
        try {
            LwmLoader.load(new ByteArrayInputStream(bytes));
            Assert.fail("expected resource limit");
        } catch (OcrException e) {
            Assert.assertEquals(OcrErrorCode.RESOURCE_LIMIT, e.getCode());
        }
    }

    @Test
    public void rejectsUnsignedTensorRank() {
        byte[] bytes = minimalModel();
        ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN).putInt(180, 0x80000000);
        rewriteChecksum(bytes);
        try {
            LwmLoader.load(new ByteArrayInputStream(bytes));
            Assert.fail("expected invalid model");
        } catch (OcrException e) {
            Assert.assertEquals(OcrErrorCode.INVALID_MODEL, e.getCode());
        }
    }

    static byte[] minimalModel() {
        final int inputOffset = 160;
        final int outputOffset = 168;
        final int tensorOffset = 176;
        final int nodeOffset = 256;
        final int fileSize = 328;
        ByteBuffer b = ByteBuffer.allocate(fileSize).order(ByteOrder.LITTLE_ENDIAN);
        b.put(0, (byte) 'L').put(1, (byte) 'W').put(2, (byte) 'M').put(3, (byte) '0');
        b.putShort(4, (short) 0).putShort(6, (short) 1).putInt(8, 160);
        b.putInt(12, 1).putInt(16, 1).putInt(20, 0).putInt(24, 1).putInt(28, 1);
        b.putLong(32, inputOffset).putLong(40, outputOffset).putLong(48, tensorOffset);
        b.putLong(56, nodeOffset).putLong(64, nodeOffset + 72).putLong(72, 0);
        b.putLong(80, nodeOffset + 72).putLong(88, 0).putLong(96, nodeOffset + 72);
        b.putLong(104, 0).putLong(112, fileSize).putLong(120, 0).putLong(128, 0);
        b.putInt(inputOffset, 0).putInt(outputOffset, 0);
        b.putInt(tensorOffset, 1).putInt(tensorOffset + 4, 1).putInt(tensorOffset + 8, 1);
        b.putInt(tensorOffset + 40, TensorInfo.INPUT | TensorInfo.OUTPUT);
        b.putLong(tensorOffset + 64, 0xffffffffffffffffL);
        long checksum = fnv1a(b.array());
        b.putLong(128, checksum);
        return b.array();
    }

    private static byte[] corrupt(byte[] source) {
        byte[] copy = source.clone();
        copy[copy.length - 1] ^= 1;
        return copy;
    }

    private static void rewriteChecksum(byte[] bytes) {
        ByteBuffer buffer = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN);
        buffer.putLong(128, 0);
        buffer.putLong(128, fnv1a(bytes));
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
