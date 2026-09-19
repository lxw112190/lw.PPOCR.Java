package io.github.lxw112190.ppocr.ppocr;

import io.github.lxw112190.ppocr.model.OcrErrorCode;
import io.github.lxw112190.ppocr.model.OcrException;
import java.io.ByteArrayInputStream;
import java.nio.charset.StandardCharsets;
import org.junit.Assert;
import org.junit.Test;

public final class CtcDecoderTest {
    @Test
    public void loadsDictionaryAndCollapsesGreedyPath() {
        PaddleOcrDictionary dictionary = PaddleOcrDictionary.load(new ByteArrayInputStream(
                "\uFEFF你\r\n好\r\n".getBytes(StandardCharsets.UTF_8)));
        Assert.assertEquals(4, dictionary.classCount());
        float[] probabilities = new float[] {
                0.9f, 0.1f, 0.0f, 0.0f,
                0.1f, 0.8f, 0.1f, 0.0f,
                0.1f, 0.8f, 0.1f, 0.0f,
                0.9f, 0.1f, 0.0f, 0.0f,
                0.1f, 0.0f, 0.8f, 0.1f,
                0.1f, 0.0f, 0.1f, 0.8f
        };
        CtcDecodeResult result = CtcDecoder.decodeGreedy(probabilities, 6, 4, dictionary);
        Assert.assertEquals("你好 ", result.getText());
        Assert.assertEquals(3, result.getEmittedCount());
        Assert.assertEquals((0.8f + 0.8f + 0.8f) / 3.0f, result.getScore(), 0.00001f);
    }

    @Test
    public void rejectsMalformedUtf8() {
        try {
            PaddleOcrDictionary.load(new ByteArrayInputStream(new byte[] {(byte) 0xc3, (byte) 0x28}));
            Assert.fail("expected invalid UTF-8");
        } catch (OcrException e) {
            Assert.assertEquals(OcrErrorCode.INVALID_MODEL, e.getCode());
        }
    }

    @Test
    public void reusesCallerOwnedTextBuilderWithoutChangingOutput() throws Exception {
        PaddleOcrDictionary dictionary = PaddleOcrDictionary.load(new ByteArrayInputStream(
                "你\r\n好\r\n".getBytes(StandardCharsets.UTF_8)));
        try {
            float[] probabilities = new float[] {
                    0.9f, 0.1f, 0.0f, 0.0f,
                    0.1f, 0.8f, 0.1f, 0.0f,
                    0.1f, 0.8f, 0.1f, 0.0f,
                    0.9f, 0.1f, 0.0f, 0.0f
            };
            float[] scores = new float[1];
            int[] emitted = new int[1];
            StringBuilder builder = new StringBuilder("stale");
            String text = CtcDecoder.decodeGreedyInto(probabilities, 0, 4, 4,
                    dictionary, scores, 0, emitted, 0, builder);
            Assert.assertEquals("你", text);
            Assert.assertEquals(1, emitted[0]);
            Assert.assertEquals(0.8f, scores[0], 0.00001f);
            Assert.assertEquals("你", builder.toString());

            float[] second = new float[] {
                    0.9f, 0.1f, 0.0f, 0.0f,
                    0.1f, 0.0f, 0.8f, 0.1f,
                    0.1f, 0.0f, 0.8f, 0.1f,
                    0.9f, 0.1f, 0.0f, 0.0f
            };
            String secondText = CtcDecoder.decodeGreedyInto(second, 0, 4, 4,
                    dictionary, scores, 0, emitted, 0, builder);
            Assert.assertEquals("好", secondText);
            Assert.assertEquals("你", text);
            Assert.assertEquals("好", builder.toString());
        } finally {
            dictionary.close();
        }
    }
}
