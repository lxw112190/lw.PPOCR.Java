package io.github.lxw112190.ppocr.ppocr;

import io.github.lxw112190.ppocr.golden.GoldenTestSupport;
import io.github.lxw112190.ppocr.image.BgrImage;
import io.github.lxw112190.ppocr.model.LwmLoader;
import io.github.lxw112190.ppocr.model.LwmModel;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import org.junit.Assert;
import org.junit.Test;

/** Real Tiny CLS API parity against the pinned C classifier result. */
public final class RealClsModelGoldenTest {
    private static final String ROOT = "/golden/cls/";
    private static final String MODEL_SHA256 = "d426c23f4758c9f21c5cbd0550ea9b90f1e7c0dd0289e1541e9754dee2e6ed61";

    @Test
    public void matchesPinnedCClassifier() throws Exception {
        String manifest = GoldenTestSupport.readText(RealClsModelGoldenTest.class, ROOT + "manifest.json");
        Assert.assertTrue(manifest.contains("\"source_commit\": \"9b31f1b\""));
        Assert.assertTrue(manifest.contains(MODEL_SHA256));
        Assert.assertEquals(MODEL_SHA256, GoldenTestSupport.sha256(
                GoldenTestSupport.readBytes(RealClsModelGoldenTest.class, ROOT + "cls.lwm")));
        BgrImage source = readPpmAsBgr(ROOT + "sample-crop.ppm");
        try (LwmModel model = LwmLoader.load(GoldenTestSupport.resource(
                     RealClsModelGoldenTest.class, ROOT + "cls.lwm"));
             PaddleOcrClassifier classifier = new PaddleOcrClassifier(model)) {
            ClsClassificationResult result = classifier.classify(source);
            Assert.assertEquals(0, result.getLabel());
            Assert.assertEquals(0, result.getOrientationDegrees());
            Assert.assertEquals(160, result.getResizedWidth());
            Assert.assertEquals(0.99998593f, result.getScore(), 2.0e-4f);
        }
    }

    private static BgrImage readPpmAsBgr(String name) throws IOException {
        byte[] bytes = GoldenTestSupport.readBytes(RealClsModelGoldenTest.class, name);
        Cursor cursor = new Cursor(bytes);
        Assert.assertEquals("P6", cursor.token());
        int width = Integer.parseInt(cursor.token());
        int height = Integer.parseInt(cursor.token());
        Assert.assertEquals("255", cursor.token());
        cursor.skipWhitespace();
        byte[] bgr = new byte[width * height * 3];
        int sourceOffset = cursor.position;
        Assert.assertEquals(bgr.length, bytes.length - sourceOffset);
        for (int i = 0; i < bgr.length; i += 3) {
            bgr[i] = bytes[sourceOffset + i + 2];
            bgr[i + 1] = bytes[sourceOffset + i + 1];
            bgr[i + 2] = bytes[sourceOffset + i];
        }
        return new BgrImage(bgr, width, height, width * 3);
    }

    private static final class Cursor {
        private final byte[] bytes;
        private int position;

        private Cursor(byte[] bytes) { this.bytes = bytes; }

        private void skipWhitespace() {
            while (position < bytes.length && (bytes[position] & 0xff) <= 32) position++;
        }

        private String token() {
            skipWhitespace();
            while (position < bytes.length && bytes[position] == '#') {
                while (position < bytes.length && bytes[position] != '\n') position++;
                skipWhitespace();
            }
            int start = position;
            while (position < bytes.length && (bytes[position] & 0xff) > 32) position++;
            return new String(bytes, start, position - start, StandardCharsets.US_ASCII);
        }
    }

}
