package io.github.lxw112190.ppocr.ppocr;

import io.github.lxw112190.ppocr.image.BgrImage;
import org.junit.Assert;
import org.junit.Test;

public final class PerspectiveCropTest {
    @Test
    public void cropsAxisAlignedQuadWithoutUsingPadding() {
        byte[] pixels = new byte[4 * 14];
        for (int y = 0; y < 4; y++) {
            for (int x = 0; x < 4; x++) {
                int index = y * 14 + x * 3;
                pixels[index] = (byte) (10 + x + y * 4);
                pixels[index + 1] = (byte) 20;
                pixels[index + 2] = (byte) 30;
            }
        }
        BgrImage crop = PerspectiveCrop.crop(new BgrImage(pixels, 4, 4, 14),
                new DetectionBox(new float[] {1, 1, 3, 1, 3, 3, 1, 3}, 0.9f));
        Assert.assertEquals(2, crop.width());
        Assert.assertEquals(2, crop.height());
        Assert.assertArrayEquals(new byte[] {15, 20, 30, 16, 20, 30, 19, 20, 30, 20, 20, 30}, crop.pixels());
    }

    @Test
    public void rotatesVeryTallQuadIntoHorizontalCrop() {
        byte[] pixels = new byte[5 * 3];
        for (int y = 0; y < 5; y++) pixels[y * 3] = (byte) (10 + y);
        BgrImage crop = PerspectiveCrop.crop(new BgrImage(pixels, 1, 5, 3),
                new DetectionBox(new float[] {0, 0, 1, 0, 1, 5, 0, 5}, 0.9f));
        Assert.assertEquals(5, crop.width());
        Assert.assertEquals(1, crop.height());
        Assert.assertEquals(14, crop.pixels()[0] & 0xff);
        Assert.assertEquals(10, crop.pixels()[12] & 0xff);
    }

    @Test
    public void reusableWorkspaceMatchesPublicCrop() {
        BgrImage source = new BgrImage(new byte[4 * 12], 4, 4, 12);
        DetectionBox box = new DetectionBox(new float[] {0, 0, 3, 0, 3, 3, 0, 3}, 0.9f);
        BgrImage expected = PerspectiveCrop.crop(source, box);
        PerspectiveCrop.Workspace workspace = new PerspectiveCrop.Workspace();
        BgrImage actual = workspace.crop(source, box);
        Assert.assertEquals(expected.width(), actual.width());
        Assert.assertEquals(expected.height(), actual.height());
        Assert.assertArrayEquals(expected.pixels(), actual.pixels());
    }
}
