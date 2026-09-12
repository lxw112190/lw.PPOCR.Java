package io.github.lxw112190.ppocr.benchmark;

import io.github.lxw112190.ppocr.ppocr.DbPostprocess;
import io.github.lxw112190.ppocr.ppocr.DetectionBox;
import java.util.List;
import java.util.Locale;

/** Dependency-free CI benchmark for DB component extraction and box fitting. */
public final class DbPostprocessPerformanceMain {
    private static final int WIDTH = 640;
    private static final int HEIGHT = 320;
    private static final int ITERATIONS = 30;

    private DbPostprocessPerformanceMain() { }

    public static void main(String[] args) {
        float[] fixture = fixture();
        int warmup = 5;
        for (int i = 0; i < warmup; i++) {
            DbPostprocess.decode(fixture, WIDTH, HEIGHT, 0.3f, 0.6f,
                    1.0f, 1.0f, 1000, 1.6f, false);
        }
        long[] samples = new long[ITERATIONS];
        List<DetectionBox> boxes = null;
        for (int i = 0; i < ITERATIONS; i++) {
            long start = System.nanoTime();
            boxes = DbPostprocess.decode(fixture, WIDTH, HEIGHT, 0.3f, 0.6f,
                    1.0f, 1.0f, 1000, 1.6f, false);
            samples[i] = System.nanoTime() - start;
        }
        java.util.Arrays.sort(samples);
        double meanMillis = mean(samples) / 1_000_000.0;
        double medianMillis = samples[samples.length / 2] / 1_000_000.0;
        double p95Millis = samples[(int) Math.min(samples.length - 1,
                Math.ceil(samples.length * 0.95) - 1)] / 1_000_000.0;
        System.out.printf(Locale.ROOT,
                "{\"benchmark\":\"db-postprocess\",\"map_width\":%d,\"map_height\":%d,"
                        + "\"foreground_pixels\":%d,\"boxes\":%d,\"iterations\":%d,"
                        + "\"mean_ms\":%.3f,\"median_ms\":%.3f,\"p95_ms\":%.3f}%n",
                WIDTH, HEIGHT, foregroundPixels(fixture), boxes == null ? 0 : boxes.size(),
                ITERATIONS, meanMillis, medianMillis, p95Millis);
    }

    private static double mean(long[] values) {
        long total = 0;
        for (long value : values) total += value;
        return (double) total / values.length;
    }

    private static float[] fixture() {
        float[] probabilities = new float[WIDTH * HEIGHT];
        for (int line = 0; line < 40; line++) {
            int column = line % 8;
            int row = line / 8;
            int baseX = 8 + column * 78;
            int baseY = 8 + row * 62;
            for (int y = 0; y < 14; y++) {
                int xStart = baseX + y / 3;
                for (int x = xStart; x < xStart + 34; x++) {
                    probabilities[(baseY + y) * WIDTH + x] = 0.9f;
                }
            }
        }
        return probabilities;
    }

    private static int foregroundPixels(float[] probabilities) {
        int count = 0;
        for (float probability : probabilities) if (probability > 0.3f) count++;
        return count;
    }
}
