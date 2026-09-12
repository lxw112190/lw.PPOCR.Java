package io.github.lxw112190.ppocr.benchmark;

import io.github.lxw112190.ppocr.kernels.KernelBackend;
import io.github.lxw112190.ppocr.kernels.ScalarBackend;
import java.util.Arrays;
import java.util.Locale;

/** Focused benchmark for a hot stride-one convolution in the Tiny DET graph. */
public final class StrideOneConvPerformanceMain {
    private static final int CHANNELS = 64;
    private static final int DEFAULT_SIZE = 80;
    private static final int OUTPUT_CHANNELS = 16;
    private static volatile float sink;

    private StrideOneConvPerformanceMain() { }

    public static void main(String[] args) throws Exception {
        String backendName = args.length > 0 ? args[0] : "scalar";
        int warmup = args.length > 1 ? positive(args[1], "warmup") : 10;
        int iterations = args.length > 2 ? positive(args[2], "iterations") : 30;
        int size = args.length > 3 ? positive(args[3], "size") : DEFAULT_SIZE;
        KernelBackend backend = createBackend(backendName);
        float[] input = fixture(CHANNELS * size * size, 101, 0.001953125f);
        float[] weights = fixture(OUTPUT_CHANNELS * CHANNELS * 9, 67, 0.0009765625f);
        float[] bias = fixture(OUTPUT_CHANNELS, 31, 0.00390625f);
        float[] output = new float[OUTPUT_CHANNELS * size * size];

        for (int i = 0; i < warmup; i++) {
            run(backend, input, weights, bias, output, size);
            consume(output, i);
        }
        long[] samples = new long[iterations];
        for (int i = 0; i < iterations; i++) {
            long start = System.nanoTime();
            run(backend, input, weights, bias, output, size);
            samples[i] = System.nanoTime() - start;
            consume(output, i);
        }
        Arrays.sort(samples);
        System.out.printf(Locale.ROOT,
                "{\"benchmark\":\"det-stride-one-conv\",\"backend\":\"%s\","
                        + "\"width\":%d,\"height\":%d,\"input_channels\":%d,"
                        + "\"output_channels\":%d,\"warmup\":%d,\"iterations\":%d,"
                        + "\"mean_ms\":%.3f,\"median_ms\":%.3f,\"p95_ms\":%.3f,"
                        + "\"checksum\":\"%s\"}%n",
                backendName, size, size, CHANNELS, OUTPUT_CHANNELS, warmup, iterations,
                mean(samples) / 1_000_000.0, samples[samples.length / 2] / 1_000_000.0,
                samples[(int) Math.min(samples.length - 1,
                        Math.ceil(samples.length * 0.95) - 1)] / 1_000_000.0,
                checksum(output));
    }

    private static void run(KernelBackend backend, float[] input, float[] weights,
                            float[] bias, float[] output, int size) {
        backend.conv(input, 0, weights, 0, bias, 0, output, 0,
                1, CHANNELS, size, size, OUTPUT_CHANNELS,
                3, 3, 1, 1, 1, 1, 1, 1, 1, 1,
                1, size, size);
    }

    private static KernelBackend createBackend(String name) throws Exception {
        if ("scalar".equals(name)) return new ScalarBackend();
        if (!"vector".equals(name)) {
            throw new IllegalArgumentException("backend must be scalar or vector");
        }
        return (KernelBackend) Class.forName("io.github.lxw112190.ppocr.vector.VectorBackend")
                .getDeclaredConstructor().newInstance();
    }

    private static int positive(String value, String name) {
        int parsed = Integer.parseInt(value);
        if (parsed <= 0) throw new IllegalArgumentException(name + " must be positive");
        return parsed;
    }

    private static float[] fixture(int length, int period, float scale) {
        float[] values = new float[length];
        int center = period / 2;
        for (int i = 0; i < length; i++) values[i] = (i % period - center) * scale;
        return values;
    }

    private static void consume(float[] values, int iteration) {
        sink += values[(iteration * 7919) % values.length];
    }

    private static double mean(long[] values) {
        long total = 0L;
        for (long value : values) total += value;
        return (double) total / values.length;
    }

    private static String checksum(float[] values) {
        long hash = -3750763034362895579L;
        for (float value : values) {
            hash ^= Float.floatToIntBits(value) & 0xffffffffL;
            hash *= 1099511628211L;
        }
        return Long.toUnsignedString(hash);
    }
}
