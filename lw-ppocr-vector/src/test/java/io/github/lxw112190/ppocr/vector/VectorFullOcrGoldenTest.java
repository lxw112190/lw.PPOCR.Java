package io.github.lxw112190.ppocr.vector;

import io.github.lxw112190.ppocr.golden.FullOcrGoldenFixture;
import io.github.lxw112190.ppocr.image.BgrImage;
import io.github.lxw112190.ppocr.model.LwmModel;
import io.github.lxw112190.ppocr.ppocr.PaddleOcr;
import io.github.lxw112190.ppocr.ppocr.PaddleOcrClassifier;
import io.github.lxw112190.ppocr.ppocr.PaddleOcrDetector;
import io.github.lxw112190.ppocr.ppocr.PaddleOcrDictionary;
import io.github.lxw112190.ppocr.ppocr.PaddleOcrOptions;
import io.github.lxw112190.ppocr.ppocr.PaddleOcrRecognizer;
import org.junit.Test;

/** End-to-end Vector API parity against the same pinned C Golden as Scalar. */
public final class VectorFullOcrGoldenTest {
    @Test
    public void matchesPinnedCPipeline() throws Exception {
        VectorBackend backend = new VectorBackend();
        BgrImage image = FullOcrGoldenFixture.loadImage(VectorFullOcrGoldenTest.class);
        try (LwmModel detectorModel = FullOcrGoldenFixture.loadModel(
                    VectorFullOcrGoldenTest.class, FullOcrGoldenFixture.ROOT + "det/det.lwm");
             LwmModel classifierModel = FullOcrGoldenFixture.loadModel(
                    VectorFullOcrGoldenTest.class, FullOcrGoldenFixture.ROOT + "cls/cls.lwm");
             LwmModel recognizerModel = FullOcrGoldenFixture.loadModel(
                    VectorFullOcrGoldenTest.class, FullOcrGoldenFixture.ROOT + "rec/rec.lwm");
             PaddleOcrDetector detector = new PaddleOcrDetector(detectorModel, 320, backend);
             PaddleOcrClassifier classifier = new PaddleOcrClassifier(classifierModel, backend);
             PaddleOcrDictionary dictionary = FullOcrGoldenFixture.loadDictionary(VectorFullOcrGoldenTest.class);
             PaddleOcrRecognizer recognizer = new PaddleOcrRecognizer(recognizerModel, dictionary, backend);
             PaddleOcr ocr = new PaddleOcr(detector, classifier, recognizer,
                     PaddleOcrOptions.builder().setRecognitionParallelism(4).build())) {
            FullOcrGoldenFixture.assertMatches(ocr.recognize(image));
        }
    }
}
