package io.github.lxw112190.ppocr.ppocr;

import io.github.lxw112190.ppocr.golden.FullOcrGoldenFixture;
import io.github.lxw112190.ppocr.image.BgrImage;
import io.github.lxw112190.ppocr.model.LwmModel;
import org.junit.Test;

/** Full Tiny OCR parity against the pinned C pipeline on the committed sample image. */
public final class RealFullOcrGoldenTest {
    @Test
    public void matchesPinnedCPipeline() throws Exception {
        assertPipeline(PaddleOcrOptions.defaults());
    }

    @Test
    public void automaticParallelismPreservesPinnedPipelineOutput() throws Exception {
        assertPipeline(PaddleOcrOptions.builder()
                .setParallelismMode(ParallelismPolicy.AUTO).build());
    }

    private static void assertPipeline(PaddleOcrOptions options) throws Exception {
        BgrImage image = FullOcrGoldenFixture.loadImage(RealFullOcrGoldenTest.class);
        try (LwmModel detectorModel = FullOcrGoldenFixture.loadModel(
                    RealFullOcrGoldenTest.class, FullOcrGoldenFixture.ROOT + "det/det.lwm");
             LwmModel classifierModel = FullOcrGoldenFixture.loadModel(
                    RealFullOcrGoldenTest.class, FullOcrGoldenFixture.ROOT + "cls/cls.lwm");
             LwmModel recognizerModel = FullOcrGoldenFixture.loadModel(
                    RealFullOcrGoldenTest.class, FullOcrGoldenFixture.ROOT + "rec/rec.lwm");
             PaddleOcrDetector detector = new PaddleOcrDetector(detectorModel, 320);
             PaddleOcrClassifier classifier = new PaddleOcrClassifier(classifierModel);
             PaddleOcrDictionary dictionary = FullOcrGoldenFixture.loadDictionary(RealFullOcrGoldenTest.class);
             PaddleOcrRecognizer recognizer = new PaddleOcrRecognizer(recognizerModel, dictionary);
             PaddleOcr ocr = new PaddleOcr(detector, classifier, recognizer, options)) {
            FullOcrGoldenFixture.assertMatches(ocr.recognize(image));
        }
    }
}
