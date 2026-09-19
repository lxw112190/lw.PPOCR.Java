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

    @Test
    public void reusableWorkspaceMatchesOneShotPreprocessAndClearsPadding() {
        BgrImage source = new BgrImage(new byte[2 * 16 * 3], 2, 16, 6);
        RecPreprocessResult oneShot = RecPreprocess.resizeNormalize(source, 32);
        RecPreprocess.Workspace workspace = new RecPreprocess.Workspace(32);
        workspace.resizeNormalize(source);
        Assert.assertEquals(oneShot.getResizedWidth(), workspace.getResizedWidth());
        Assert.assertArrayEquals(oneShot.getChw(), workspace.getChw(), 0.0f);

        byte[] bright = new byte[16 * 16 * 3];
        java.util.Arrays.fill(bright, (byte) 255);
        workspace.resizeNormalize(new BgrImage(bright, 16, 16, 48));
        Assert.assertEquals(32, workspace.getResizedWidth());
        Assert.assertEquals(1.0f, workspace.getChw()[32 * 48 + 8], 0.00001f);
    }

    @Test
    public void intoPathMatchesOneShotWhenDestinationContainsStaleValues() {
        BgrImage source = new BgrImage(new byte[7 * 19 * 3], 7, 19, 21);
        RecPreprocessResult expected = RecPreprocess.resizeNormalize(source, 32);
        float[] actual = new float[expected.getChw().length + 11];
        java.util.Arrays.fill(actual, 0.25f);
        RecPreprocess.resizeNormalizeInto(source, 32, actual, 5);
        Assert.assertArrayEquals(expected.getChw(),
                java.util.Arrays.copyOfRange(actual, 5, 5 + expected.getChw().length), 0.0f);
    }
}
