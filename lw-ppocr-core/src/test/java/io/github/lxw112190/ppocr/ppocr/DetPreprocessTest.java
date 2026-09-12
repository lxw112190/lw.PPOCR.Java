package io.github.lxw112190.ppocr.ppocr;

import io.github.lxw112190.ppocr.image.BgrImage;
import org.junit.Assert;
import org.junit.Test;

public final class DetPreprocessTest {
    @Test
    public void roundsToStrideAndNormalizesBgrWithPaddedStride() {
        byte[] pixels = new byte[4 * 8];
        for (int row = 0; row < 4; row++) {
            for (int column = 0; column < 2; column++) {
                int index = row * 8 + column * 3;
                pixels[index] = (byte) 255;
                pixels[index + 1] = 0;
                pixels[index + 2] = (byte) 128;
            }
        }
        DetPreprocessResult result = DetPreprocess.resizeNormalize(new BgrImage(pixels, 2, 4, 8), 32);
        Assert.assertEquals(32, result.getResizedWidth());
        Assert.assertEquals(32, result.getResizedHeight());
        Assert.assertEquals(3 * 32 * 32, result.getChw().length);
        Assert.assertEquals((1.0f - 0.485f) / 0.229f, result.getChw()[0], 0.00001f);
        Assert.assertEquals((0.0f - 0.456f) / 0.224f, result.getChw()[32 * 32], 0.00001f);
        Assert.assertEquals((128.0f / 255.0f - 0.406f) / 0.225f,
                result.getChw()[2 * 32 * 32], 0.00001f);
    }

    @Test
    public void limitsLongSideAndKeepsThirtyTwoPixelAlignment() {
        byte[] pixels = new byte[100 * 50 * 3];
        DetPreprocessResult result = DetPreprocess.resizeNormalize(new BgrImage(pixels, 100, 50, 300), 64);
        Assert.assertEquals(64, result.getResizedWidth());
        Assert.assertEquals(32, result.getResizedHeight());
        Assert.assertEquals(0.64f, result.getWidthRatio(), 0.00001f);
        Assert.assertEquals(0.64f, result.getHeightRatio(), 0.00001f);
    }
}
