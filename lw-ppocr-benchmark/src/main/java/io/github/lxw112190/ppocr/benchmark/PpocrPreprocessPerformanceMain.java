package io.github.lxw112190.ppocr.benchmark;

import io.github.lxw112190.ppocr.image.BgrImage;
import io.github.lxw112190.ppocr.ppocr.ClsPreprocess;
import io.github.lxw112190.ppocr.ppocr.DetPreprocess;
import io.github.lxw112190.ppocr.ppocr.RecPreprocess;
import java.util.Arrays;
import java.util.Locale;

/** Dependency-free CI benchmark for reusable OCR preprocessing workspaces. */
public final class PpocrPreprocessPerformanceMain {
    private static final int ITERATIONS = 30;
    private static volatile int sink;

    private PpocrPreprocessPerformanceMain() { }

    public static void main(String[] args) {
        BgrImage source = fixture();
        DetPreprocess.Workspace det = new DetPreprocess.Workspace(640, 320);
        ClsPreprocess.Workspace cls = new ClsPreprocess.Workspace();
        RecPreprocess.Workspace rec = new RecPreprocess.Workspace(320);
        for (int i = 0; i < 5; i++) {
            det.resizeNormalize(source);
            cls.resizeNormalize(source);
            rec.resizeNormalize(source);
            consume(det.getChw());
            consume(cls.getChw());
            consume(rec.getChw());
        }
        long[] detSamples = new long[ITERATIONS];
        long[] clsSamples = new long[ITERATIONS];
        long[] recSamples = new long[ITERATIONS];
        for (int i = 0; i < ITERATIONS; i++) {
            long start = System.nanoTime();
            det.resizeNormalize(source);
            consume(det.getChw());
            detSamples[i] = System.nanoTime() - start;

            start = System.nanoTime();
            cls.resizeNormalize(source);
            consume(cls.getChw());
            clsSamples[i] = System.nanoTime() - start;

            start = System.nanoTime();
            rec.resizeNormalize(source);
            consume(rec.getChw());
            recSamples[i] = System.nanoTime() - start;
        }
        Arrays.sort(detSamples);
        Arrays.sort(clsSamples);
        Arrays.sort(recSamples);
        System.out.printf(Locale.ROOT,
                "{\"benchmark\":\"ppocr-preprocess\",\"source_width\":%d,"
                        + "\"source_height\":%d,\"iterations\":%d,"
                        + "\"det_mean_ms\":%.3f,\"det_median_ms\":%.3f,\"det_p95_ms\":%.3f,"
                        + "\"cls_mean_ms\":%.3f,\"cls_median_ms\":%.3f,\"cls_p95_ms\":%.3f,"
                        + "\"rec_mean_ms\":%.3f,\"rec_median_ms\":%.3f,\"rec_p95_ms\":%.3f}%n",
                source.width(), source.height(), ITERATIONS,
                mean(detSamples) / 1_000_000.0, percentile(detSamples, 0.50) / 1_000_000.0,
                percentile(detSamples, 0.95) / 1_000_000.0,
                mean(clsSamples) / 1_000_000.0, percentile(clsSamples, 0.50) / 1_000_000.0,
                percentile(clsSamples, 0.95) / 1_000_000.0,
                mean(recSamples) / 1_000_000.0, percentile(recSamples, 0.50) / 1_000_000.0,
                percentile(recSamples, 0.95) / 1_000_000.0);
    }

    private static BgrImage fixture() {
        byte[] pixels = new byte[640 * 320 * 3];
        for (int i = 0; i < pixels.length; i++) pixels[i] = (byte) (i * 13);
        return new BgrImage(pixels, 640, 320, 640 * 3);
    }

    private static void consume(float[] values) {
        sink ^= Float.floatToIntBits(values[0]) ^ Float.floatToIntBits(values[values.length - 1]);
    }

    private static double mean(long[] values) {
        long total = 0;
        for (long value : values) total += value;
        return (double) total / values.length;
    }

    private static long percentile(long[] values, double fraction) {
        return values[(int) Math.min(values.length - 1,
                Math.ceil(values.length * fraction) - 1)];
    }
}
