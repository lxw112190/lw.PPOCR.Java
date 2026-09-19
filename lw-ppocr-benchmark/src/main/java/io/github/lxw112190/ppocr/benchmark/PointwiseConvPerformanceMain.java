package io.github.lxw112190.ppocr.benchmark;

import io.github.lxw112190.ppocr.kernels.KernelBackend;
import io.github.lxw112190.ppocr.kernels.ScalarBackend;
import java.util.Arrays;
import java.util.Locale;

/** Focused benchmark for the 1x1 convolution workloads used by PP-OCR. */
public final class PointwiseConvPerformanceMain {
    private static volatile float sink;

    private PointwiseConvPerformanceMain() { }

    public static void main(String[] args) throws Exception {
        String backendName = args.length > 0 ? args[0] : "scalar";
        String profile = args.length > 1 ? args[1] : "rec";
        int warmup = args.length > 2 ? positive(args[2], "warmup") : 30;
        int iterations = args.length > 3 ? positive(args[3], "iterations") : 50;
        Shape shape = Shape.forProfile(profile);
        KernelBackend backend = createBackend(backendName);
        float[] input = fixture(shape.batch * shape.channels * shape.plane, 101, 0.001953125f);
        float[] weights = fixture(shape.outputChannels * shape.channels / shape.groups,
                67, 0.0009765625f);
        float[] bias = fixture(shape.outputChannels, 31, 0.00390625f);
        float[] output = new float[shape.batch * shape.outputChannels * shape.plane];

        for (int i = 0; i < warmup; i++) {
            run(backend, shape, input, weights, bias, output);
            consume(output, i);
        }
        long[] samples = new long[iterations];
        for (int i = 0; i < iterations; i++) {
            long start = System.nanoTime();
            run(backend, shape, input, weights, bias, output);
            samples[i] = System.nanoTime() - start;
            consume(output, i);
        }
        Arrays.sort(samples);
        System.out.printf(Locale.ROOT,
                "{\"benchmark\":\"pointwise-conv\",\"backend\":\"%s\","
                        + "\"profile\":\"%s\",\"batch\":%d,\"channels\":%d,"
                        + "\"output_channels\":%d,\"height\":%d,\"width\":%d,"
                        + "\"groups\":%d,\"warmup\":%d,\"iterations\":%d,"
                        + "\"mean_ms\":%.3f,\"median_ms\":%.3f,\"p95_ms\":%.3f,"
                        + "\"checksum\":\"%s\"}%n",
                backendName, profile, shape.batch, shape.channels, shape.outputChannels,
                shape.height, shape.width, shape.groups, warmup, iterations,
                mean(samples) / 1_000_000.0,
                samples[samples.length / 2] / 1_000_000.0,
                samples[(int) Math.min(samples.length - 1,
                        Math.ceil(samples.length * 0.95) - 1)] / 1_000_000.0,
                checksum(output));
    }

    private static void run(KernelBackend backend, Shape shape, float[] input,
                            float[] weights, float[] bias, float[] output) {
        backend.conv(input, 0, weights, 0, bias, 0, output, 0,
                shape.batch, shape.channels, shape.height, shape.width,
                shape.outputChannels, 1, 1, 1, 1, 1, 1,
                0, 0, 0, 0, shape.groups, shape.height, shape.width);
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

    private static final class Shape {
        final int batch;
        final int channels;
        final int height;
        final int width;
        final int outputChannels;
        final int groups;
        final int plane;

        Shape(int batch, int channels, int height, int width, int outputChannels, int groups) {
            this.batch = batch;
            this.channels = channels;
            this.height = height;
            this.width = width;
            this.outputChannels = outputChannels;
            this.groups = groups;
            this.plane = height * width;
        }

        static Shape forProfile(String profile) {
            if ("rec".equals(profile)) return new Shape(1, 320, 3, 240, 160, 1);
            if ("cls".equals(profile)) return new Shape(1, 128, 3, 80, 128, 1);
            if ("det".equals(profile)) return new Shape(1, 64, 80, 80, 32, 1);
            throw new IllegalArgumentException("profile must be rec, cls, or det");
        }
    }
}
