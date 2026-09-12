package io.github.lxw112190.ppocr.vector;

import io.github.lxw112190.ppocr.golden.FullOcrGoldenFixture;
import io.github.lxw112190.ppocr.image.BgrImage;
import io.github.lxw112190.ppocr.ppocr.PaddleOcr;
import io.github.lxw112190.ppocr.ppocr.PaddleOcrOptions;
import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import org.junit.Test;

/** End-to-end Vector API parity against the same pinned C Golden as Scalar. */
public final class VectorFullOcrGoldenTest {
    @Test
    public void matchesPinnedCPipeline() throws Exception {
        VectorBackend backend = new VectorBackend();
        BgrImage image = FullOcrGoldenFixture.loadImage(VectorFullOcrGoldenTest.class);
        Path directory = Files.createTempDirectory(Paths.get("target"),
                "lw-ppocr-vector-golden-");
        Path detector = directory.resolve("det.lwm");
        Path classifier = directory.resolve("cls.lwm");
        Path recognizer = directory.resolve("rec.lwm");
        Path dictionary = directory.resolve("ppocr_keys.txt");
        copyResource(detector, FullOcrGoldenFixture.ROOT + "det/det.lwm");
        copyResource(classifier, FullOcrGoldenFixture.ROOT + "cls/cls.lwm");
        copyResource(recognizer, FullOcrGoldenFixture.ROOT + "rec/rec.lwm");
        copyResource(dictionary, FullOcrGoldenFixture.ROOT + "rec/ppocr_keys.txt");
        PaddleOcrOptions options = PaddleOcrOptions.builder()
                .setDetectionMaximumSideLength(320)
                .setRecognitionParallelism(4)
                .build();
        try (PaddleOcr ocr = PaddleOcr.load(detector, classifier, recognizer,
                dictionary, options, backend)) {
            FullOcrGoldenFixture.assertMatches(ocr.recognize(image));
        }
    }

    private static void copyResource(Path destination, String resource) throws IOException {
        try (InputStream input = VectorFullOcrGoldenTest.class.getResourceAsStream(resource)) {
            if (input == null) throw new IOException("missing Golden resource: " + resource);
            Files.copy(input, destination);
        }
    }
}
