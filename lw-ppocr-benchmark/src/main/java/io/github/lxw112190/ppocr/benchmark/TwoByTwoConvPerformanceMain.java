package io.github.lxw112190.ppocr.benchmark;

import io.github.lxw112190.ppocr.kernels.KernelBackend;
import io.github.lxw112190.ppocr.kernels.ScalarBackend;
import java.util.Arrays;
import java.util.Locale;

/** Focused benchmark for the two same-size 2x2 convolutions in Tiny DET. */
public final class TwoByTwoConvPerformanceMain {
    private static final int SIZE = 160;
    private static volatile float sink;

    private TwoByTwoConvPerformanceMain() { }

    public static void main(String[] args) throws Exception {
        String backendName = args.length > 0 ? args[0] : "scalar";
        String workloadName = args.length > 1 ? args[1] : "expand";
        int warmup = args.length > 2 ? positive(args[2], "warmup") : 10;
        int iterations = args.length > 3 ? positive(args[3], "iterations") : 30;
        Workload workload = workload(workloadName);
        KernelBackend backend = createBackend(backendName);
        float[] input = fixture(workload.channels * SIZE * SIZE, 101, 0.001953125f);
        float[] weights = fixture(workload.outputChannels * workload.channels * 4,
                67, 0.0009765625f);
        float[] bias = fixture(workload.outputChannels, 31, 0.00390625f);
        float[] output = new float[workload.outputChannels * SIZE * SIZE];

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
                "{\"benchmark\":\"det-two-by-two-conv\",\"workload\":\"%s\","
                        + "\"backend\":\"%s\",\"width\":%d,\"height\":%d,"
                        + "\"input_channels\":%d,\"output_channels\":%d,"
                        + "\"warmup\":%d,\"iterations\":%d,\"mean_ms\":%.3f,"
                        + "\"median_ms\":%.3f,\"p95_ms\":%.3f,\"checksum\":\"%s\"}%n",
                workloadName, backendName, SIZE, SIZE, workload.channels,
                workload.outputChannels, warmup, iterations,
                mean(samples) / 1_000_000.0, samples[samples.length / 2] / 1_000_000.0,
                samples[(int) Math.min(samples.length - 1,
                        Math.ceil(samples.length * 0.95) - 1)] / 1_000_000.0,
                checksum(output));
    }

    private static void run(KernelBackend backend, Workload workload, float[] input,
                            float[] weights, float[] bias, float[] output) {
        backend.conv(input, 0, weights, 0, bias, 0, output, 0,
                1, workload.channels, SIZE, SIZE, workload.outputChannels,
                2, 2, 1, 1, 1, 1, 0, 0, 1, 1,
                1, SIZE, SIZE);
    }

    private static Workload workload(String name) {
        if ("reduce".equals(name)) return new Workload(16, 8);
        if ("expand".equals(name)) return new Workload(8, 16);
        throw new IllegalArgumentException("workload must be reduce or expand");
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
        private final int outputChannels;

        private Workload(int channels, int outputChannels) {
            this.channels = channels;
            this.outputChannels = outputChannels;
        }
    }
}
