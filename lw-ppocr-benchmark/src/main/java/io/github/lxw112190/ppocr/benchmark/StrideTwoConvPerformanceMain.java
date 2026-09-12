package io.github.lxw112190.ppocr.benchmark;

import io.github.lxw112190.ppocr.kernels.KernelBackend;
import io.github.lxw112190.ppocr.kernels.ScalarBackend;
import java.util.Arrays;
import java.util.Locale;

/** Focused benchmark for the first stride-two convolution in the Tiny REC graph. */
public final class StrideTwoConvPerformanceMain {
    private static final int CHANNELS = 24;
    private static final int HEIGHT = 24;
    private static final int DEFAULT_WIDTH = 480;
    private static final int OUTPUT_CHANNELS = 48;
    private static volatile float sink;

    private StrideTwoConvPerformanceMain() { }

    public static void main(String[] args) throws Exception {
        String backendName = args.length > 0 ? args[0] : "scalar";
        int warmup = args.length > 1 ? positive(args[1], "warmup") : 10;
        int iterations = args.length > 2 ? positive(args[2], "iterations") : 30;
        int width = args.length > 3 ? positive(args[3], "width") : DEFAULT_WIDTH;
        if (width % 2 != 0) throw new IllegalArgumentException("width must be even");
        int outputHeight = HEIGHT / 2;
        int outputWidth = width / 2;
        KernelBackend backend = createBackend(backendName);
        float[] input = fixture(CHANNELS * HEIGHT * width, 101, 0.001953125f);
        float[] weights = fixture(OUTPUT_CHANNELS * CHANNELS * 9, 67, 0.0009765625f);
        float[] bias = fixture(OUTPUT_CHANNELS, 31, 0.00390625f);
        float[] output = new float[OUTPUT_CHANNELS * outputHeight * outputWidth];

        for (int i = 0; i < warmup; i++) {
            run(backend, input, weights, bias, output, width, outputHeight, outputWidth);
            consume(output, i);
        }
        long[] samples = new long[iterations];
        for (int i = 0; i < iterations; i++) {
            long start = System.nanoTime();
            run(backend, input, weights, bias, output, width, outputHeight, outputWidth);
            samples[i] = System.nanoTime() - start;
            consume(output, i);
        }
        Arrays.sort(samples);
        System.out.printf(Locale.ROOT,
                "{\"benchmark\":\"rec-stride-two-conv\",\"backend\":\"%s\","
                        + "\"input_width\":%d,\"input_height\":%d,\"input_channels\":%d,"
                        + "\"output_width\":%d,\"output_height\":%d,"
                        + "\"output_channels\":%d,\"warmup\":%d,\"iterations\":%d,"
                        + "\"mean_ms\":%.3f,\"median_ms\":%.3f,\"p95_ms\":%.3f,"
                        + "\"checksum\":\"%s\"}%n",
                backendName, width, HEIGHT, CHANNELS, outputWidth, outputHeight,
                OUTPUT_CHANNELS, warmup, iterations, mean(samples) / 1_000_000.0,
                samples[samples.length / 2] / 1_000_000.0,
                samples[(int) Math.min(samples.length - 1,
                        Math.ceil(samples.length * 0.95) - 1)] / 1_000_000.0,
                checksum(output));
    }

    private static void run(KernelBackend backend, float[] input, float[] weights,
                            float[] bias, float[] output, int width,
                            int outputHeight, int outputWidth) {
        backend.conv(input, 0, weights, 0, bias, 0, output, 0,
                1, CHANNELS, HEIGHT, width, OUTPUT_CHANNELS,
                3, 3, 2, 2, 1, 1, 1, 1, 1, 1,
                1, outputHeight, outputWidth);
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
