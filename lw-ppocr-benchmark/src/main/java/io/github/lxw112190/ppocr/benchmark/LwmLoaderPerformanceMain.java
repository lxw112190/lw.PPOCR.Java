package io.github.lxw112190.ppocr.benchmark;

import io.github.lxw112190.ppocr.model.LwmLoader;
import java.io.ByteArrayInputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.Locale;

/**
 * Small CI-friendly benchmark for the validated LWM loader.
 *
 * This is deliberately dependency-free. It measures model parsing rather than
 * OCR inference until the scalar graph executor exists.
 */
public final class LwmLoaderPerformanceMain {
    private static final int NODE_COUNT = 2000;
    private static final int TENSOR_COUNT = NODE_COUNT + 2;

    private LwmLoaderPerformanceMain() { }

    public static void main(String[] args) {
        byte[] fixture = fixture();
        int warmup = 5;
        int iterations = 30;
        for (int i = 0; i < warmup; i++) {
            LwmLoader.load(new ByteArrayInputStream(fixture));
        }
        long[] samples = new long[iterations];
        for (int i = 0; i < iterations; i++) {
            long start = System.nanoTime();
            LwmLoader.load(new ByteArrayInputStream(fixture));
            samples[i] = System.nanoTime() - start;
        }
        java.util.Arrays.sort(samples);
        double meanMillis = mean(samples) / 1_000_000.0;
        double medianMillis = samples[samples.length / 2] / 1_000_000.0;
        double p95Millis = samples[(int) Math.min(samples.length - 1, Math.ceil(samples.length * 0.95) - 1)]
                / 1_000_000.0;
        System.out.printf(Locale.ROOT,
                "{\"benchmark\":\"lwm-loader\",\"fixture_bytes\":%d,\"tensors\":%d,"
                        + "\"nodes\":%d,\"iterations\":%d,\"mean_ms\":%.3f,"
                        + "\"median_ms\":%.3f,\"p95_ms\":%.3f}%n",
                fixture.length, TENSOR_COUNT, NODE_COUNT, iterations,
                meanMillis, medianMillis, p95Millis);
    }

    private static double mean(long[] values) {
        long total = 0;
        for (long value : values) {
            total += value;
        }
        return ((double) total) / values.length;
    }

    private static byte[] fixture() {
        int inputOffset = 160;
        int outputOffset = inputOffset + 8;
        int tensorOffset = outputOffset + 8;
        int nodeOffset = tensorOffset + TENSOR_COUNT * 80;
        int parameterOffset = nodeOffset + NODE_COUNT * 72;
        int weightOffset = parameterOffset;
        int fileSize = weightOffset + 8;
        ByteBuffer b = ByteBuffer.allocate(fileSize).order(ByteOrder.LITTLE_ENDIAN);
        b.put(0, (byte) 'L').put(1, (byte) 'W').put(2, (byte) 'M').put(3, (byte) '0');
        b.putShort(4, (short) 0).putShort(6, (short) 1).putInt(8, 160);
        b.putInt(12, 1).putInt(16, TENSOR_COUNT).putInt(20, NODE_COUNT);
        b.putInt(24, 1).putInt(28, 1);
        b.putLong(32, inputOffset).putLong(40, outputOffset).putLong(48, tensorOffset);
        b.putLong(56, nodeOffset).putLong(64, parameterOffset).putLong(72, 0);
        b.putLong(80, parameterOffset).putLong(88, 0).putLong(96, weightOffset);
        b.putLong(104, 8).putLong(112, fileSize).putLong(120, 0).putLong(128, 0);
        b.putInt(inputOffset, 0).putInt(outputOffset, TENSOR_COUNT - 1);
        putTensor(b, tensorOffset, true, false, 0);
        putTensor(b, tensorOffset + 80, false, false, weightOffset);
        for (int i = 0; i < NODE_COUNT; i++) {
            int tensorIndex = i + 2;
            int base = tensorOffset + tensorIndex * 80;
            putTensor(b, base, false, i == NODE_COUNT - 1, 0);
            int node = nodeOffset + i * 72;
            b.putShort(node, (short) 2).putShort(node + 2, (short) 2).putShort(node + 4, (short) 1);
            b.putInt(node + 8, i == 0 ? 0 : tensorIndex - 1).putInt(node + 12, 1);
            b.putInt(node + 40, tensorIndex);
        }
        b.putFloat(weightOffset, 1.0f);
        b.putLong(128, fnv1a(b.array()));
        return b.array();
    }

    private static void putTensor(ByteBuffer b, int offset, boolean input, boolean output, int dataOffset) {
        int flags = (input ? 2 : 0) | (output ? 4 : 0);
        if (dataOffset != 0) {
            flags |= 1;
        }
        b.putInt(offset, 1).putInt(offset + 4, 1).putInt(offset + 8, 1);
        b.putInt(offset + 40, flags).putLong(offset + 48, dataOffset);
        b.putLong(offset + 56, dataOffset == 0 ? 0 : 4);
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
