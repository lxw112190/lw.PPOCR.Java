package io.github.lxw112190.ppocr.ppocr;

import io.github.lxw112190.ppocr.image.BgrImage;
import org.junit.Assert;
import org.junit.Test;

public final class ClsPreprocessTest {
    @Test
    public void preservesAspectRatioAndUsesBlackPadding() {
        byte[] pixels = new byte[2 * 16 * 3];
        for (int i = 0; i < pixels.length; i += 3) {
            pixels[i] = (byte) 255;
            pixels[i + 1] = 0;
            pixels[i + 2] = (byte) 128;
        }
        ClsPreprocessResult result = ClsPreprocess.resizeNormalize(new BgrImage(pixels, 2, 16, 6));
        Assert.assertEquals(10, result.getResizedWidth());
        Assert.assertEquals(3 * 80 * 160, result.getChw().length);
        float[] output = result.getChw();
        Assert.assertEquals(1.0f, output[0], 0.00001f);
        Assert.assertEquals(-1.0f, output[80 * 160], 0.00001f);
        Assert.assertEquals(128.0f * 2.0f / 255.0f - 1.0f,
                output[2 * 80 * 160], 0.00001f);
        Assert.assertEquals(-1.0f, output[10], 0.00001f);
    }

    @Test
    public void reusableWorkspaceMatchesOneShotPreprocessAndClearsPadding() {
        BgrImage source = new BgrImage(new byte[2 * 16 * 3], 2, 16, 6);
        ClsPreprocessResult oneShot = ClsPreprocess.resizeNormalize(source);
        ClsPreprocess.Workspace workspace = new ClsPreprocess.Workspace();
        workspace.resizeNormalize(source);
        Assert.assertEquals(oneShot.getResizedWidth(), workspace.getResizedWidth());
        Assert.assertArrayEquals(oneShot.getChw(), workspace.getChw(), 0.0f);

        byte[] bright = new byte[80 * 80 * 3];
        java.util.Arrays.fill(bright, (byte) 255);
        workspace.resizeNormalize(new BgrImage(bright, 80, 80, 240));
        Assert.assertEquals(80, workspace.getResizedWidth());
        Assert.assertEquals(1.0f, workspace.getChw()[80 * 160 + 20], 0.00001f);
    }
}
