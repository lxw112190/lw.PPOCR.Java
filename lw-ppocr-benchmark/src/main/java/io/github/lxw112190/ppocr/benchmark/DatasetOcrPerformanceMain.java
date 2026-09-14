package io.github.lxw112190.ppocr.benchmark;

import io.github.lxw112190.ppocr.imageio.ImageIoLoader;
import io.github.lxw112190.ppocr.kernels.KernelBackend;
import io.github.lxw112190.ppocr.kernels.ScalarBackend;
import io.github.lxw112190.ppocr.ppocr.ClsClassificationResult;
import io.github.lxw112190.ppocr.ppocr.OcrLineResult;
import io.github.lxw112190.ppocr.ppocr.OcrResult;
import io.github.lxw112190.ppocr.ppocr.PaddleOcr;
import io.github.lxw112190.ppocr.ppocr.PaddleOcrOptions;
import java.io.BufferedWriter;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.DirectoryStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;
import java.util.Locale;

/** Runs the complete Java OCR pipeline over an external metadata-backed image directory. */
public final class DatasetOcrPerformanceMain {
    private DatasetOcrPerformanceMain() { }

    public static void main(String[] args) throws Exception {
        if (args.length < 6 || args.length > 14) {
            throw new IllegalArgumentException("usage: dataset detector classifier recognizer "
                    + "dictionary output [detector-limit] [bitmap] [box] [unclip] [workers] [backend] [start] [limit]");
        }
        Path dataset = requiredDirectory(args[0]);
        Path detector = requiredFile(args[1]);
        Path classifier = requiredFile(args[2]);
        Path recognizer = requiredFile(args[3]);
        Path dictionary = requiredFile(args[4]);
        Path output = Paths.get(args[5]).toAbsolutePath().normalize();
        int detectorLimit = args.length > 6 ? positive(args[6], "detector limit") : 960;
        float bitmapThreshold = args.length > 7 ? finiteFloat(args[7], "bitmap threshold") : 0.3f;
        float boxThreshold = args.length > 8 ? finiteFloat(args[8], "box threshold") : 0.6f;
        float unclipRatio = args.length > 9 ? finiteFloat(args[9], "unclip ratio") : 1.6f;
        int workers = args.length > 10 ? positive(args[10], "workers") : 1;
        String backendName = args.length > 11 ? args[11] : "scalar";
        int start = args.length > 12 ? nonNegative(args[12], "start") : 0;
        int limit = args.length > 13 ? positive(args[13], "limit") : Integer.MAX_VALUE;
        KernelBackend backend = createBackend(backendName);
        List<Path> images = imageFiles(dataset);
        if (images.isEmpty()) throw new IllegalArgumentException("dataset has no supported images: " + dataset);
        if (start >= images.size()) throw new IllegalArgumentException("start is outside dataset: " + start);
        int end = limit > images.size() - start ? images.size() : start + limit;
        images = new ArrayList<Path>(images.subList(start, end));
        Path parent = output.getParent();
        if (parent != null) Files.createDirectories(parent);

        PaddleOcrOptions options = PaddleOcrOptions.builder()
                .setDetectionMaximumSideLength(detectorLimit)
                .setDetectionBitmapThreshold(bitmapThreshold)
                .setDetectionBoxThreshold(boxThreshold)
                .setDetectionUnclipRatio(unclipRatio)
                .setParallelism(workers)
                .build();
        long totalNanos = 0L;
        int totalLines = 0;
        long peakUsedHeap = 0L;
        try (PaddleOcr ocr = PaddleOcr.load(detector, classifier, recognizer, dictionary,
                options, backend);
             BufferedWriter writer = Files.newBufferedWriter(output, StandardCharsets.UTF_8)) {
            for (int index = 0; index < images.size(); index++) {
                Path image = images.get(index);
                long startNanos = System.nanoTime();
                OcrResult result = ocr.recognize(ImageIoLoader.load(image));
                long elapsed = System.nanoTime() - startNanos;
                totalNanos += elapsed;
                totalLines += result.getLines().size();
                peakUsedHeap = Math.max(peakUsedHeap, usedHeap());
                writer.write(toJson(datasetFileName(image), result, elapsed));
                writer.newLine();
                writer.flush();
                System.err.printf(Locale.ROOT, "[java-ocr] %d/%d %s lines=%d ms=%.3f%n",
                        index + 1, images.size(), image.getFileName(), result.getLines().size(),
                        elapsed / 1_000_000.0);
            }
        }
        System.out.printf(Locale.ROOT,
                "{\"benchmark\":\"dataset-ocr\",\"images\":%d,\"total_lines\":%d,"
                        + "\"mean_ms\":%.3f,\"peak_used_heap_bytes\":%d,\"output\":\"%s\"}%n",
                images.size(), totalLines, totalNanos / 1_000_000.0 / images.size(),
                peakUsedHeap, jsonEscape(output.toString()));
    }

    private static List<Path> imageFiles(Path dataset) throws IOException {
        List<Path> images = new ArrayList<Path>();
        try (DirectoryStream<Path> stream = Files.newDirectoryStream(dataset)) {
            for (Path path : stream) {
                if (Files.isRegularFile(path) && isSupportedImage(path)) images.add(path);
            }
        }
        Collections.sort(images, Comparator.comparing(path -> path.getFileName().toString()));
        return images;
    }

    private static boolean isSupportedImage(Path path) {
        String name = path.getFileName().toString().toLowerCase(Locale.ROOT);
        return name.endsWith(".jpg") || name.endsWith(".jpeg") || name.endsWith(".png");
    }

    private static String datasetFileName(Path path) {
        String name = path.getFileName().toString();
        int extension = name.lastIndexOf('.');
        return extension > 0 ? name.substring(0, extension) + ".jpg" : name;
    }

    private static KernelBackend createBackend(String name) {
        if ("scalar".equalsIgnoreCase(name)) return new ScalarBackend();
        if (!"vector".equalsIgnoreCase(name)) throw new IllegalArgumentException("backend must be scalar or vector");
        try {
            return (KernelBackend) Class.forName("io.github.lxw112190.ppocr.vector.VectorBackend")
                    .getDeclaredConstructor().newInstance();
        } catch (ReflectiveOperationException exception) {
            throw new IllegalStateException("Vector backend is unavailable", exception);
        }
    }

    private static String toJson(String file, OcrResult result, long elapsedNanos) {
        StringBuilder json = new StringBuilder(256);
        json.append("{\"file\":\"").append(jsonEscape(file)).append("\",\"elapsed_ms\":")
                .append(String.format(Locale.ROOT, "%.3f", elapsedNanos / 1_000_000.0))
                .append(",\"lines\":[");
        List<OcrLineResult> lines = result.getLines();
        for (int index = 0; index < lines.size(); index++) {
            if (index != 0) json.append(',');
            OcrLineResult line = lines.get(index);
            ClsClassificationResult classification = line.getClassification();
            float[] points = line.getBox().getPoints();
            json.append("{\"text\":\"").append(jsonEscape(line.getText())).append("\",\"rec\":")
                    .append(number(line.getRecognitionScore())).append(",\"det\":")
                    .append(number(line.getBox().getScore())).append(",\"cls\":")
                    .append(classification == null ? -1 : classification.getLabel()).append(",\"cls_score\":")
                    .append(classification == null ? "0" : number(classification.getScore()))
                    .append(",\"rotate\":").append(line.isRotated()).append(",\"box\":[");
            for (int point = 0; point < points.length; point++) {
                if (point != 0) json.append(',');
                json.append(number(points[point]));
            }
            json.append("]}");
        }
        return json.append("]}").toString();
    }

    private static String number(float value) {
        return Float.isFinite(value) ? String.format(Locale.ROOT, "%.9g", value) : "0";
    }

    private static String jsonEscape(String value) {
        StringBuilder escaped = new StringBuilder(value.length() + 16);
        for (int index = 0; index < value.length(); index++) {
            char character = value.charAt(index);
            if (character == '\\') escaped.append("\\\\");
            else if (character == '"') escaped.append("\\\"");
            else if (character == '\n') escaped.append("\\n");
            else if (character == '\r') escaped.append("\\r");
            else if (character == '\t') escaped.append("\\t");
            else if (character < 0x20) escaped.append(String.format(Locale.ROOT, "\\u%04x", (int) character));
            else escaped.append(character);
        }
        return escaped.toString();
    }

    private static long usedHeap() {
        Runtime runtime = Runtime.getRuntime();
        return runtime.totalMemory() - runtime.freeMemory();
    }

    private static Path requiredDirectory(String value) {
        Path path = Paths.get(value).toAbsolutePath().normalize();
        if (!Files.isDirectory(path)) throw new IllegalArgumentException("directory not found: " + path);
        return path;
    }

    private static Path requiredFile(String value) {
        Path path = Paths.get(value).toAbsolutePath().normalize();
        if (!Files.isRegularFile(path)) throw new IllegalArgumentException("file not found: " + path);
        return path;
    }

    private static int positive(String value, String label) {
        int parsed = Integer.parseInt(value);
        if (parsed <= 0) throw new IllegalArgumentException(label + " must be positive");
        return parsed;
    }

    private static int nonNegative(String value, String label) {
        int parsed = Integer.parseInt(value);
        if (parsed < 0) throw new IllegalArgumentException(label + " must be non-negative");
        return parsed;
    }

    private static float finiteFloat(String value, String label) {
        float parsed = Float.parseFloat(value);
        if (!Float.isFinite(parsed)) throw new IllegalArgumentException(label + " must be finite");
        return parsed;
    }
}
