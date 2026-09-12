package io.github.lxw112190.ppocr.ppocr;

import io.github.lxw112190.ppocr.image.BgrImage;
import org.junit.Assert;
import org.junit.Test;

public final class RecPreprocessTest {
    @Test
    public void preservesAspectRatioAndUsesChwNormalization() {
        byte[] pixels = new byte[2 * 16 * 3];
        for (int i = 0; i < pixels.length; i += 3) {
            pixels[i] = (byte) 255;
            pixels[i + 1] = 0;
            pixels[i + 2] = (byte) 128;
        }
        RecPreprocessResult result = RecPreprocess.resizeNormalize(new BgrImage(pixels, 2, 16, 6), 8);
        Assert.assertEquals(6, result.getResizedWidth());
        Assert.assertEquals(3 * 48 * 8, result.getChw().length);
        float[] output = result.getChw();
        Assert.assertEquals(1.0f, output[0], 0.00001f);
        Assert.assertEquals(-1.0f, output[48 * 8], 0.00001f);
        Assert.assertEquals(128.0f * 2.0f / 255.0f - 1.0f, output[2 * 48 * 8], 0.00001f);
        Assert.assertEquals(128.0f * 2.0f / 255.0f - 1.0f, output[6], 0.00001f);
    }
}
