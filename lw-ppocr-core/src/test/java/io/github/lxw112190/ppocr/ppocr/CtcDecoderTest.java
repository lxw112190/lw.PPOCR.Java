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
}
