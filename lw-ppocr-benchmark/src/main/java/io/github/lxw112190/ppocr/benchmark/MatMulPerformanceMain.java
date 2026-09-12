package io.github.lxw112190.ppocr.benchmark;

import io.github.lxw112190.ppocr.kernels.KernelBackend;
import io.github.lxw112190.ppocr.kernels.ScalarBackend;
import java.util.Arrays;
import java.util.Locale;

/** Focused benchmark for the PP-OCRv6 Tiny REC projection matrix. */
public final class MatMulPerformanceMain {
    private static final int DEFAULT_ROWS = 60;
    private static final int DEFAULT_INNER = 80;
    private static final int DEFAULT_COLUMNS = 6906;
    private static volatile float sink;

    private MatMulPerformanceMain() { }

    public static void main(String[] args) throws Exception {
        String backendName = args.length > 0 ? args[0] : "scalar";
        int warmup = args.length > 1 ? positive(args[1], "warmup") : 30;
        int iterations = args.length > 2 ? positive(args[2], "iterations") : 50;
        int rows = args.length > 3 ? positive(args[3], "rows") : DEFAULT_ROWS;
        int inner = args.length > 4 ? positive(args[4], "inner") : DEFAULT_INNER;
        int columns = args.length > 5 ? positive(args[5], "columns") : DEFAULT_COLUMNS;
        KernelBackend backend = createBackend(backendName);
        float[] left = fixture(rows * inner, 101, 0.001953125f);
        float[] right = fixture(inner * columns, 67, 0.0009765625f);
        float[] output = new float[rows * columns];

        for (int i = 0; i < warmup; i++) {
            backend.matMul(left, 0, right, 0, output, 0, rows, inner, columns);
            consume(output, i);
        }
        long[] samples = new long[iterations];
        for (int i = 0; i < iterations; i++) {
            long start = System.nanoTime();
            backend.matMul(left, 0, right, 0, output, 0, rows, inner, columns);
            samples[i] = System.nanoTime() - start;
            consume(output, i);
        }
        Arrays.sort(samples);
        double meanMillis = mean(samples) / 1_000_000.0;
        double medianMillis = samples[samples.length / 2] / 1_000_000.0;
        double p95Millis = samples[(int) Math.min(samples.length - 1,
                Math.ceil(samples.length * 0.95) - 1)] / 1_000_000.0;
        System.out.printf(Locale.ROOT,
                "{\"benchmark\":\"rec-projection-matmul\",\"backend\":\"%s\","
                        + "\"rows\":%d,\"inner\":%d,\"columns\":%d,\"warmup\":%d,"
                        + "\"iterations\":%d,\"mean_ms\":%.3f,\"median_ms\":%.3f,"
                        + "\"p95_ms\":%.3f,\"checksum\":\"%s\"}%n",
                backendName, rows, inner, columns, warmup, iterations,
                meanMillis, medianMillis, p95Millis, checksum(output));
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
