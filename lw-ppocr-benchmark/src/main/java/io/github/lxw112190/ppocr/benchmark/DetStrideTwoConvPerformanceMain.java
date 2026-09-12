package io.github.lxw112190.ppocr.benchmark;

import io.github.lxw112190.ppocr.kernels.KernelBackend;
import io.github.lxw112190.ppocr.kernels.ScalarBackend;
import java.util.Arrays;
import java.util.Locale;

/** Focused benchmark for the two largest stride-two convolutions in Tiny DET. */
public final class DetStrideTwoConvPerformanceMain {
    private static final int OUTPUT_CHANNELS = 16;
    private static volatile float sink;

    private DetStrideTwoConvPerformanceMain() { }

    public static void main(String[] args) throws Exception {
        String backendName = args.length > 0 ? args[0] : "scalar";
        String workloadName = args.length > 1 ? args[1] : "downsample";
        int warmup = args.length > 2 ? positive(args[2], "warmup") : 10;
        int iterations = args.length > 3 ? positive(args[3], "iterations") : 30;
        Workload workload = workload(workloadName);
        KernelBackend backend = createBackend(backendName);
        int outputHeight = workload.height / 2;
        int outputWidth = workload.width / 2;
        float[] input = fixture(workload.channels * workload.height * workload.width,
                101, 0.001953125f);
        float[] weights = fixture(OUTPUT_CHANNELS * workload.channels * 9,
                67, 0.0009765625f);
        float[] bias = fixture(OUTPUT_CHANNELS, 31, 0.00390625f);
        float[] output = new float[OUTPUT_CHANNELS * outputHeight * outputWidth];

        for (int i = 0; i < warmup; i++) {
            run(backend, workload, input, weights, bias, output, outputHeight, outputWidth);
            consume(output, i);
        }
        long[] samples = new long[iterations];
        for (int i = 0; i < iterations; i++) {
            long start = System.nanoTime();
            run(backend, workload, input, weights, bias, output, outputHeight, outputWidth);
            samples[i] = System.nanoTime() - start;
            consume(output, i);
        }
        Arrays.sort(samples);
        System.out.printf(Locale.ROOT,
                "{\"benchmark\":\"det-stride-two-conv\",\"workload\":\"%s\","
                        + "\"backend\":\"%s\",\"input_width\":%d,\"input_height\":%d,"
                        + "\"input_channels\":%d,\"output_width\":%d,"
                        + "\"output_height\":%d,\"output_channels\":%d,"
                        + "\"warmup\":%d,\"iterations\":%d,\"mean_ms\":%.3f,"
                        + "\"median_ms\":%.3f,\"p95_ms\":%.3f,\"checksum\":\"%s\"}%n",
                workloadName, backendName, workload.width, workload.height, workload.channels,
                outputWidth, outputHeight, OUTPUT_CHANNELS, warmup, iterations,
                mean(samples) / 1_000_000.0, samples[samples.length / 2] / 1_000_000.0,
                samples[(int) Math.min(samples.length - 1,
                        Math.ceil(samples.length * 0.95) - 1)] / 1_000_000.0,
                checksum(output));
    }

    private static void run(KernelBackend backend, Workload workload, float[] input,
                            float[] weights, float[] bias, float[] output,
                            int outputHeight, int outputWidth) {
        backend.conv(input, 0, weights, 0, bias, 0, output, 0,
                1, workload.channels, workload.height, workload.width, OUTPUT_CHANNELS,
                3, 3, 2, 2, 1, 1, 1, 1, 1, 1,
                1, outputHeight, outputWidth);
    }

    private static Workload workload(String name) {
        if ("stem".equals(name)) return new Workload(3, 320, 320);
        if ("downsample".equals(name)) return new Workload(32, 160, 160);
        throw new IllegalArgumentException("workload must be stem or downsample");
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
