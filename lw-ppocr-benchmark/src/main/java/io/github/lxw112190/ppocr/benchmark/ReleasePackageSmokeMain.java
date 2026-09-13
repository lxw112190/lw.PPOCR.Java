package io.github.lxw112190.ppocr.benchmark;

import io.github.lxw112190.ppocr.imageio.PaddleOcrImageIo;
import io.github.lxw112190.ppocr.kernels.KernelBackend;
import io.github.lxw112190.ppocr.ppocr.OcrResult;
import io.github.lxw112190.ppocr.ppocr.PaddleOcr;
import io.github.lxw112190.ppocr.ppocr.PaddleOcrOptions;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;

/** Runs real OCR using only the JARs and model files staged in a release bundle. */
public final class ReleasePackageSmokeMain {
    private ReleasePackageSmokeMain() { }

    public static void main(String[] args) throws Exception {
        if (args.length != 1) {
            throw new IllegalArgumentException("release package root is required");
        }
        Path root = Paths.get(args[0]).toAbsolutePath().normalize();
        Path models = root.resolve("models").resolve("ppocrv6-tiny");
        Path detector = required(models.resolve("det.lwm"));
        Path classifier = required(models.resolve("cls.lwm"));
        Path recognizer = required(models.resolve("rec.lwm"));
        Path dictionary = required(models.resolve("ppocr_keys.txt"));
        Path image = required(models.resolve("sample.jpg"));
        required(models.resolve("manifest.json"));
        required(root.resolve("LICENSE"));
        required(root.resolve("THIRD-PARTY-NOTICES.md"));
        required(root.resolve("licenses").resolve("PaddleOCR-models-APACHE-2.0.txt"));

        KernelBackend backend = (KernelBackend) Class.forName(
                "io.github.lxw112190.ppocr.vector.VectorBackend")
                .getDeclaredConstructor().newInstance();
        PaddleOcrOptions options = PaddleOcrOptions.builder()
                .setDetectionMaximumSideLength(320)
                .build();
        try (PaddleOcr ocr = PaddleOcr.load(detector, classifier, recognizer,
                dictionary, options, backend)) {
            OcrResult result = PaddleOcrImageIo.recognize(ocr, image);
            if (result.getLines().size() != 16) {
                throw new IllegalStateException("release package OCR line count: "
                        + result.getLines().size());
            }
            if (!result.getText().startsWith("纯臻营养护发素\n")
                    || !result.getText().endsWith("发足够的滋养")) {
                throw new IllegalStateException("release package OCR text mismatch");
            }
            System.out.println("{\"release_package_smoke\":\"passed\",\"lines\":16}");
        }
    }

    private static Path required(Path path) {
        if (!Files.isRegularFile(path)) {
            throw new IllegalStateException("missing release file: " + path);
        }
        return path;
    }
}
