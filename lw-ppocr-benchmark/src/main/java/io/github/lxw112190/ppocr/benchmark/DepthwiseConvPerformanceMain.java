package io.github.lxw112190.ppocr.benchmark;

import io.github.lxw112190.ppocr.kernels.KernelBackend;
import io.github.lxw112190.ppocr.kernels.ScalarBackend;
import java.util.Arrays;
import java.util.Locale;

/** Focused benchmark for same-size 5x5 depthwise convolutions in Tiny DET, CLS, and REC. */
public final class DepthwiseConvPerformanceMain {
    private static volatile float sink;

    private DepthwiseConvPerformanceMain() { }

    public static void main(String[] args) throws Exception {
        String backendName = args.length > 0 ? args[0] : "scalar";
        String workloadName = args.length > 1 ? args[1] : "det";
        int warmup = args.length > 2 ? positive(args[2], "warmup") : 50;
        int iterations = args.length > 3 ? positive(args[3], "iterations") : 100;
        Workload workload = workload(workloadName);
        KernelBackend backend = createBackend(backendName);
        int plane = workload.height * workload.width;
        float[] input = fixture(workload.channels * plane, 101, 0.001953125f);
        float[] weights = fixture(workload.channels * 25, 67, 0.0009765625f);
        float[] bias = fixture(workload.channels, 31, 0.00390625f);
        float[] output = new float[input.length];

        for (int i = 0; i < warmup; i++) {
            run(backend, workload, input, weights, bias, output);
            consume(output, i);
        }
        long[] samples = new long[iterations];
        for (int i = 0; i < iterations; i++) {
            long start = System.nanoTime();
            run(backend, workload, input, weights, bias, output);
            samples[i] = System.nanoTime() - start;
            consume(output, i);
        }
        Arrays.sort(samples);
        System.out.printf(Locale.ROOT,
                "{\"benchmark\":\"depthwise-five-by-five-conv\",\"workload\":\"%s\","
                        + "\"backend\":\"%s\",\"width\":%d,\"height\":%d,"
                        + "\"channels\":%d,\"warmup\":%d,\"iterations\":%d,"
                        + "\"mean_ms\":%.3f,\"median_ms\":%.3f,\"p95_ms\":%.3f,"
                        + "\"checksum\":\"%s\"}%n",
                workloadName, backendName, workload.width, workload.height, workload.channels,
                warmup, iterations, mean(samples) / 1_000_000.0,
                samples[samples.length / 2] / 1_000_000.0,
                samples[(int) Math.min(samples.length - 1,
                        Math.ceil(samples.length * 0.95) - 1)] / 1_000_000.0,
                checksum(output));
    }

    private static void run(KernelBackend backend, Workload workload, float[] input,
                            float[] weights, float[] bias, float[] output) {
        backend.conv(input, 0, weights, 0, bias, 0, output, 0,
                1, workload.channels, workload.height, workload.width, workload.channels,
                5, 5, 1, 1, 1, 1, 2, 2, 2, 2,
                workload.channels, workload.height, workload.width);
    }

    private static Workload workload(String name) {
        if ("det".equals(name)) return new Workload(64, 80, 80);
        if ("cls".equals(name)) return new Workload(64, 5, 80);
        if ("rec".equals(name)) return new Workload(160, 3, 240);
        throw new IllegalArgumentException("workload must be det, cls, or rec");
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

    private static final class Workload {
        private final int channels;
        private final int height;
        private final int width;

        private Workload(int channels, int height, int width) {
            this.channels = channels;
            this.height = height;
            this.width = width;
        }
    }
}
