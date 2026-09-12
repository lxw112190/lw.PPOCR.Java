package io.github.lxw112190.ppocr.ppocr;

import io.github.lxw112190.ppocr.golden.GoldenTestSupport;
import io.github.lxw112190.ppocr.image.BgrImage;
import io.github.lxw112190.ppocr.model.LwmLoader;
import io.github.lxw112190.ppocr.model.LwmModel;
import java.awt.image.BufferedImage;
import java.io.InputStream;
import javax.imageio.ImageIO;
import org.junit.Assert;
import org.junit.Test;

/** Full Tiny OCR parity against the pinned C pipeline on the committed sample image. */
public final class RealFullOcrGoldenTest {
    private static final String ROOT = "/golden/";
    private static final String OCR_ROOT = ROOT + "ocr/";
    private static final String SOURCE_SHA256 = "30c417c9f758a3b62718729f5a944f7d2e10cdd2bde0e8ce6785523ddb68ffe9";
    private static final String[] TEXT = {
        "纯臻营养护发素", "产品信息/参数", "(45元/每公斤，100公斤起订)",
        "每瓶22元，1000瓶起订)", "【品牌】：代加工方式/OEM ODM", "【品名】：纯臻营养护发素",
        "【产品编号】：YM-X-3011", "ODM OEM", "【净含量】：220ml", "【适用人群】：适合所有肤质",
        "【主要成分】：鲸蜡硬脂醇、燕麦β-葡聚", "糖、椰油酰胺丙基甜菜碱、泛醌", "(成品包材)",
        "【主要功能】：可紧致头发磷层，从而达到", "即时持久改善头发光泽的效果，给干燥的头",
        "发足够的滋养"
    };
    private static final int[] LABELS = {0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0};
    private static final boolean[] ROTATED = {
        false, false, false, false, false, false, false, true,
        false, false, false, false, false, false, false, false
    };
    private static final float[][] BOXES = {
        {22.585228f, 33.522728f, 305.539795f, 33.522728f, 305.539795f, 72.727280f, 22.585228f, 72.727280f},
        {23.151596f, 77.839096f, 175.285904f, 77.839096f, 175.285904f, 104.973404f, 23.151596f, 104.973404f},
        {21.248415f, 108.748413f, 335.001587f, 108.748413f, 335.001587f, 136.564087f, 21.248415f, 136.564087f},
        {24.021086f, 141.208588f, 288.478943f, 141.208588f, 288.478943f, 165.041412f, 24.021086f, 165.041412f},
        {22.443182f, 174.005692f, 302.556824f, 174.005692f, 302.556824f, 197.869324f, 22.443182f, 197.869324f},
        {22.523321f, 205.335831f, 236.851669f, 205.335831f, 236.851669f, 229.039169f, 22.523321f, 229.039169f},
        {20.946428f, 236.571442f, 244.678574f, 236.571442f, 244.678574f, 260.303589f, 20.946428f, 260.303589f},
        {411.664246f, 231.976746f, 430.523254f, 231.976746f, 430.523254f, 302.398254f, 411.664246f, 302.398254f},
        {22.190659f, 269.065674f, 180.934357f, 269.065674f, 180.934357f, 288.746826f, 22.190659f, 288.746826f},
        {22.497845f, 299.060364f, 254.064667f, 299.060364f, 254.064667f, 322.814667f, 22.497845f, 322.814667f},
        {22.409174f, 330.221680f, 344.778320f, 330.221680f, 344.778320f, 354.153320f, 22.409174f, 354.153320f},
        {22.460228f, 361.522736f, 285.352295f, 361.522736f, 285.352295f, 385.352295f, 22.460228f, 385.352295f},
        {366.280121f, 363.482208f, 477.999115f, 365.809662f, 477.478180f, 390.814941f, 365.759186f, 388.487457f},
        {22.396803f, 392.709320f, 363.540710f, 392.709320f, 363.540710f, 416.665710f, 22.396803f, 416.665710f},
        {23.953621f, 423.953644f, 374.483887f, 423.953644f, 374.483887f, 447.921387f, 23.953621f, 447.921387f},
        {24.392857f, 455.642853f, 138.107147f, 455.642853f, 138.107147f, 478.732178f, 24.392857f, 478.732178f}
    };

    @Test
    public void matchesPinnedCPipeline() throws Exception {
        String manifest = GoldenTestSupport.readText(RealFullOcrGoldenTest.class, OCR_ROOT + "manifest.json");
        Assert.assertTrue(manifest.contains("\"source_commit\": \"9b31f1b\""));
        Assert.assertTrue(manifest.contains(SOURCE_SHA256));
        byte[] imageBytes = GoldenTestSupport.readBytes(RealFullOcrGoldenTest.class, OCR_ROOT + "sample.jpg");
        Assert.assertEquals(SOURCE_SHA256, GoldenTestSupport.sha256(imageBytes));
        BgrImage image = decode(imageBytes);
        Assert.assertEquals(500, image.width());
        Assert.assertEquals(500, image.height());

        try (LwmModel detectorModel = load(ROOT + "det/det.lwm");
             LwmModel classifierModel = load(ROOT + "cls/cls.lwm");
             LwmModel recognizerModel = load(ROOT + "rec/rec.lwm");
             PaddleOcrDetector detector = new PaddleOcrDetector(detectorModel, 320);
             PaddleOcrClassifier classifier = new PaddleOcrClassifier(classifierModel);
             PaddleOcrDictionary dictionary = PaddleOcrDictionary.load(resource(ROOT + "rec/ppocr_keys.txt"));
             PaddleOcrRecognizer recognizer = new PaddleOcrRecognizer(recognizerModel, dictionary);
             PaddleOcr ocr = new PaddleOcr(detector, classifier, recognizer)) {
            OcrResult result = ocr.recognize(image);
            Assert.assertEquals(TEXT.length, result.getLines().size());
            for (int i = 0; i < TEXT.length; i++) {
                OcrLineResult line = result.getLines().get(i);
                Assert.assertEquals(TEXT[i], line.getText());
                Assert.assertEquals(LABELS[i], line.getClassification().getLabel());
                Assert.assertEquals(ROTATED[i], line.isRotated());
                float[] actual = line.getBox().getPoints();
                Assert.assertEquals(BOXES[i].length, actual.length);
                for (int j = 0; j < actual.length; j++) {
                    Assert.assertEquals("line=" + i + " point=" + j, BOXES[i][j], actual[j], 2.0f);
                }
                Assert.assertTrue("line=" + i + " DET score", line.getBox().getScore() >= 0.70f);
                Assert.assertTrue("line=" + i + " REC score", line.getRecognitionScore() >= 0.85f);
                Assert.assertTrue("line=" + i + " CLS score", line.getClassification().getScore() >= 0.90f);
            }
        }
    }

    private static LwmModel load(String resource) throws Exception {
        try (InputStream input = resource(resource)) {
            return LwmLoader.load(input);
        }
    }

    private static InputStream resource(String name) {
        InputStream input = RealFullOcrGoldenTest.class.getResourceAsStream(name);
        if (input == null) throw new AssertionError("missing Golden resource: " + name);
        return input;
    }

    private static BgrImage decode(byte[] bytes) throws Exception {
        BufferedImage source = ImageIO.read(new java.io.ByteArrayInputStream(bytes));
        if (source == null) throw new AssertionError("unable to decode Golden JPEG");
        int width = source.getWidth();
        int height = source.getHeight();
        byte[] pixels = new byte[width * height * 3];
        int[] row = new int[width];
        for (int y = 0; y < height; y++) {
            source.getRGB(0, y, width, 1, row, 0, width);
            for (int x = 0; x < width; x++) {
                int value = row[x];
                int offset = (y * width + x) * 3;
                pixels[offset] = (byte) value;
                pixels[offset + 1] = (byte) (value >>> 8);
                pixels[offset + 2] = (byte) (value >>> 16);
            }
        }
        return new BgrImage(pixels, width, height, width * 3);
    }
}
