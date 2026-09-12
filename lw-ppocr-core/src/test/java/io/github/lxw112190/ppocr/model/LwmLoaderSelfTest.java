package io.github.lxw112190.ppocr.model;

import java.io.ByteArrayInputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;

/** Small dependency-free smoke test; run with assertions enabled. */
public final class LwmLoaderSelfTest {
    private LwmLoaderSelfTest() { }

    public static void main(String[] args) {
        byte[] bytes = minimalModel();
        LwmModel model = LwmLoader.load(new ByteArrayInputStream(bytes));
        assert model.getHeader().getFormatMinor() == 1;
        assert model.getTensors().size() == 1;
        assert model.getGraphInputs().get(0) == 0;
        assert model.getGraphOutputs().get(0) == 0;
        model.close();
        expect(OcrErrorCode.CHECKSUM_MISMATCH, corrupt(bytes));
        System.out.println("LwmLoaderSelfTest: OK");
    }

    private static byte[] minimalModel() {
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

    private static void expect(OcrErrorCode code, byte[] bytes) {
        try {
            LwmLoader.load(new ByteArrayInputStream(bytes));
            throw new AssertionError("expected " + code);
        } catch (OcrException e) {
            assert e.getCode() == code : e.getCode();
        }
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
