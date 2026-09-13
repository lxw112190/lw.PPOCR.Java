package io.github.lxw112190.ppocr.benchmark;

import io.github.lxw112190.ppocr.kernels.KernelBackend;
import io.github.lxw112190.ppocr.kernels.ProjectionArgMaxBackend;
import io.github.lxw112190.ppocr.kernels.ScalarBackend;
import java.util.Arrays;
import java.util.Locale;

/** Compares the dense REC terminal path with projection/argmax/softmax fusion. */
public final class ProjectionArgMaxPerformanceMain {
    private static final int ROWS = 40;
    private static final int INNER = 80;
    private static final int COLUMNS = 6906;
    private static volatile long sink;

    private ProjectionArgMaxPerformanceMain() { }

    public static void main(String[] args) throws Exception {
        String backendName = args.length > 0 ? args[0] : "scalar";
        int warmup = args.length > 1 ? positive(args[1], "warmup") : 10;
        int iterations = args.length > 2 ? positive(args[2], "iterations") : 30;
        KernelBackend kernels = createBackend(backendName);
        if (!(kernels instanceof ProjectionArgMaxBackend)) {
            throw new IllegalStateException("backend does not support projection fusion");
        }
        ProjectionArgMaxBackend fused = (ProjectionArgMaxBackend) kernels;
        float[] activations = fixture(ROWS * INNER, 101, 0.001953125f);
        float[] weights = fixture(INNER * COLUMNS, 67, 0.0009765625f);
        float[] bias = fixture(COLUMNS, 43, 0.001953125f);
        float[] dense = new float[ROWS * COLUMNS];
        int[] denseIds = new int[ROWS];
        float[] denseProbabilities = new float[ROWS];
        int[] fusedIds = new int[ROWS];
        float[] fusedLogits = new float[ROWS];
        float[] fusedProbabilities = new float[ROWS];
        float[] rowScratch = new float[Math.min(ROWS, 4) * COLUMNS];

        for (int i = 0; i < warmup; i++) {
            runDense(kernels, activations, weights, bias, dense, denseIds, denseProbabilities);
            fused.projectionArgMax(activations, 0, weights, 0, bias, 0,
                    ROWS, INNER, COLUMNS, fusedIds, fusedLogits,
                    fusedProbabilities, rowScratch);
            consume(fusedIds, fusedProbabilities, i);
        }
        long[] denseSamples = new long[iterations];
        long[] fusedSamples = new long[iterations];
        for (int i = 0; i < iterations; i++) {
            long start = System.nanoTime();
            runDense(kernels, activations, weights, bias, dense, denseIds, denseProbabilities);
            denseSamples[i] = System.nanoTime() - start;
            start = System.nanoTime();
            fused.projectionArgMax(activations, 0, weights, 0, bias, 0,
                    ROWS, INNER, COLUMNS, fusedIds, fusedLogits,
                    fusedProbabilities, rowScratch);
            fusedSamples[i] = System.nanoTime() - start;
            assertEquivalent(denseIds, denseProbabilities, fusedIds, fusedProbabilities);
            consume(fusedIds, fusedProbabilities, i);
        }
        Arrays.sort(denseSamples);
        Arrays.sort(fusedSamples);
        double denseMean = mean(denseSamples) / 1_000_000.0;
        double fusedMean = mean(fusedSamples) / 1_000_000.0;
        System.out.printf(Locale.ROOT,
                "{\"benchmark\":\"rec-projection-argmax\",\"backend\":\"%s\","
                        + "\"rows\":%d,\"inner\":%d,\"columns\":%d,\"warmup\":%d,"
                        + "\"iterations\":%d,\"dense_mean_ms\":%.3f,"
                        + "\"dense_median_ms\":%.3f,\"dense_p95_ms\":%.3f,"
                        + "\"fused_mean_ms\":%.3f,\"fused_median_ms\":%.3f,"
                        + "\"fused_p95_ms\":%.3f,\"mean_speedup\":%.3f,"
                        + "\"dense_output_bytes\":%d,\"compact_output_bytes\":%d,"
                        + "\"reusable_scratch_bytes\":%d,"
                        + "\"checksum\":\"%s\"}%n",
                backendName, ROWS, INNER, COLUMNS, warmup, iterations,
                denseMean, milliseconds(percentile(denseSamples, 0.50)),
                milliseconds(percentile(denseSamples, 0.95)), fusedMean,
                milliseconds(percentile(fusedSamples, 0.50)),
                milliseconds(percentile(fusedSamples, 0.95)), denseMean / fusedMean,
                ROWS * COLUMNS * 4L, ROWS * (Integer.BYTES + 2L * Float.BYTES),
                Math.min(ROWS, 4) * COLUMNS * 4L,
                checksum(fusedIds, fusedProbabilities));
    }

    private static void runDense(KernelBackend backend, float[] activations, float[] weights,
                                 float[] bias, float[] output, int[] ids, float[] probabilities) {
        backend.matMul(activations, 0, weights, 0, output, 0, ROWS, INNER, COLUMNS);
        for (int row = 0; row < ROWS; row++) {
            int base = row * COLUMNS;
            for (int column = 0; column < COLUMNS; column++) output[base + column] += bias[column];
        }
        backend.softmax(output, 0, output, 0, ROWS, COLUMNS, 1);
        for (int row = 0; row < ROWS; row++) {
            int base = row * COLUMNS;
            int best = 0;
            for (int column = 1; column < COLUMNS; column++) {
                if (output[base + column] > output[base + best]) best = column;
            }
            ids[row] = best;
            probabilities[row] = output[base + best];
        }
    }

    private static void assertEquivalent(int[] expectedIds, float[] expectedProbabilities,
                                         int[] actualIds, float[] actualProbabilities) {
        for (int row = 0; row < ROWS; row++) {
            if (expectedIds[row] != actualIds[row] ||
                    Math.abs(expectedProbabilities[row] - actualProbabilities[row]) > 1.0e-6f) {
                throw new IllegalStateException("fused result mismatch at row " + row);
            }
        }
    }

    private static KernelBackend createBackend(String name) throws Exception {
        if ("scalar".equals(name)) return new ScalarBackend();
        if (!"vector".equals(name)) throw new IllegalArgumentException("backend must be scalar or vector");
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

    private static void consume(int[] ids, float[] probabilities, int iteration) {
        int index = iteration % ids.length;
        sink += ids[index] + Float.floatToIntBits(probabilities[index]);
    }

    private static double mean(long[] values) {
        long total = 0L;
        for (long value : values) total += value;
        return (double) total / values.length;
    }

    private static long percentile(long[] sorted, double fraction) {
        return sorted[(int) Math.min(sorted.length - 1,
                Math.ceil(sorted.length * fraction) - 1)];
    }

    private static double milliseconds(long nanos) { return nanos / 1_000_000.0; }

    private static String checksum(int[] ids, float[] probabilities) {
        long hash = -3750763034362895579L;
        for (int i = 0; i < ids.length; i++) {
            hash ^= ids[i] & 0xffffffffL;
            hash *= 1099511628211L;
            hash ^= Float.floatToIntBits(probabilities[i]) & 0xffffffffL;
            hash *= 1099511628211L;
        }
        return Long.toUnsignedString(hash);
    }
}
