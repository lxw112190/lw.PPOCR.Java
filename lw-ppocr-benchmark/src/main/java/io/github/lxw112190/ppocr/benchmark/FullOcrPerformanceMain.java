package io.github.lxw112190.ppocr.benchmark;

import io.github.lxw112190.ppocr.image.BgrImage;
import io.github.lxw112190.ppocr.imageio.ImageIoLoader;
import io.github.lxw112190.ppocr.model.LwmLoader;
import io.github.lxw112190.ppocr.model.LwmModel;
import io.github.lxw112190.ppocr.ppocr.PaddleOcr;
import io.github.lxw112190.ppocr.ppocr.PaddleOcrClassifier;
import io.github.lxw112190.ppocr.ppocr.PaddleOcrDetector;
import io.github.lxw112190.ppocr.ppocr.PaddleOcrDictionary;
import io.github.lxw112190.ppocr.ppocr.PaddleOcrRecognizer;
import java.io.IOException;
import java.io.InputStream;
import java.lang.management.ManagementFactory;
import java.lang.management.MemoryMXBean;
import java.lang.management.MemoryUsage;
import java.util.Arrays;
import java.util.Locale;

/** CI-friendly end-to-end OCR benchmark over the committed Tiny Golden assets. */
public final class FullOcrPerformanceMain {
    private static final String DET_MODEL = "/golden/det/det.lwm";
    private static final String CLS_MODEL = "/golden/cls/cls.lwm";
    private static final String REC_MODEL = "/golden/rec/rec.lwm";
    private static final String DICTIONARY = "/golden/rec/ppocr_keys.txt";
    private static final String IMAGE = "/golden/ocr/sample.jpg";
    private static final int DEFAULT_WARMUP = 1;
    private static final int DEFAULT_ITERATIONS = 5;

    private FullOcrPerformanceMain() { }

    public static void main(String[] args) throws Exception {
        int warmup = args.length > 0 ? positive(args[0], "warmup") : DEFAULT_WARMUP;
        int iterations = args.length > 1 ? positive(args[1], "iterations") : DEFAULT_ITERATIONS;
        BgrImage image = loadImage();
        MemoryMXBean memory = ManagementFactory.getMemoryMXBean();
        long heapBefore = usedHeap(memory);
        try (PaddleOcr ocr = loadOcr()) {
            for (int i = 0; i < warmup; i++) ocr.recognize(image);
            long[] samples = new long[iterations];
            long peakHeap = usedHeap(memory);
            int lineCount = -1;
            for (int i = 0; i < iterations; i++) {
                long start = System.nanoTime();
                int lines = ocr.recognize(image).getLines().size();
                samples[i] = System.nanoTime() - start;
                lineCount = lines;
                peakHeap = Math.max(peakHeap, usedHeap(memory));
            }
            Arrays.sort(samples);
            long heapAfter = usedHeap(memory);
            long peakDelta = Math.max(0L, peakHeap - heapBefore);
            System.out.printf(Locale.ROOT,
                    "{\"benchmark\":\"full-ocr\",\"image_width\":%d,"
                            + "\"image_height\":%d,\"lines\":%d,\"warmup\":%d,"
                            + "\"iterations\":%d,\"mean_ms\":%.3f,"
                            + "\"median_ms\":%.3f,\"p95_ms\":%.3f,"
                            + "\"heap_before_bytes\":%d,\"heap_after_bytes\":%d,"
                            + "\"peak_heap_bytes\":%d,\"peak_heap_delta_bytes\":%d}%n",
                    image.width(), image.height(), lineCount, warmup, iterations,
                    mean(samples) / 1_000_000.0, percentile(samples, 0.50) / 1_000_000.0,
                    percentile(samples, 0.95) / 1_000_000.0, heapBefore, heapAfter,
                    peakHeap, peakDelta);
        }
    }

    private static PaddleOcr loadOcr() throws IOException {
        LwmModel detectorModel = loadModel(DET_MODEL);
        PaddleOcrDetector detector = null;
        LwmModel classifierModel = null;
        PaddleOcrClassifier classifier = null;
        LwmModel recognizerModel = null;
        PaddleOcrDictionary dictionary = null;
        PaddleOcrRecognizer recognizer = null;
        try {
            detector = new PaddleOcrDetector(detectorModel);
            classifierModel = loadModel(CLS_MODEL);
            classifier = new PaddleOcrClassifier(classifierModel);
            recognizerModel = loadModel(REC_MODEL);
            dictionary = loadDictionary();
            recognizer = new PaddleOcrRecognizer(recognizerModel, dictionary);
            return new PaddleOcr(detector, classifier, recognizer);
        } catch (RuntimeException e) {
            if (recognizer != null) recognizer.close();
            else {
                if (dictionary != null) dictionary.close();
                if (recognizerModel != null) recognizerModel.close();
            }
            if (classifier != null) classifier.close();
            else if (classifierModel != null) classifierModel.close();
            if (detector != null) detector.close();
            else detectorModel.close();
            throw e;
        }
    }

    private static LwmModel loadModel(String resource) throws IOException {
        try (InputStream input = resource(resource)) {
            return LwmLoader.load(input);
        }
    }

    private static PaddleOcrDictionary loadDictionary() throws IOException {
        try (InputStream input = resource(DICTIONARY)) {
            return PaddleOcrDictionary.load(input);
        }
    }

    private static BgrImage loadImage() throws IOException {
        try (InputStream input = resource(IMAGE)) {
            return ImageIoLoader.load(input);
        }
    }

    private static InputStream resource(String name) throws IOException {
        InputStream input = FullOcrPerformanceMain.class.getResourceAsStream(name);
        if (input == null) throw new IOException("missing benchmark resource: " + name);
        return input;
    }

    private static long usedHeap(MemoryMXBean memory) {
        MemoryUsage usage = memory.getHeapMemoryUsage();
        return usage == null ? 0L : usage.getUsed();
    }

    private static int positive(String value, String name) {
        try {
            int parsed = Integer.parseInt(value);
            if (parsed > 0) return parsed;
        } catch (NumberFormatException ignored) { }
        throw new IllegalArgumentException(name + " must be positive");
    }

    private static double mean(long[] values) {
        long total = 0L;
        for (long value : values) total += value;
        return (double) total / values.length;
    }

    private static long percentile(long[] values, double fraction) {
        return values[(int) Math.min(values.length - 1,
                Math.ceil(values.length * fraction) - 1)];
    }
}
