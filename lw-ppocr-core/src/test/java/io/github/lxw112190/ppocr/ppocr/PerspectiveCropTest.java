package io.github.lxw112190.ppocr.ppocr;

import io.github.lxw112190.ppocr.image.BgrImage;
import java.util.Arrays;
import java.util.ArrayList;
import java.util.List;
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

    @Test
    public void reusableWorkspaceRetainsPixelBuffersBySlot() {
        byte[] pixels = new byte[6 * 6 * 3];
        for (int i = 0; i < pixels.length; i++) pixels[i] = (byte) i;
        BgrImage source = new BgrImage(pixels, 6, 6, 18);
        PerspectiveCrop.Workspace workspace = new PerspectiveCrop.Workspace();
        DetectionBox large = new DetectionBox(
                new float[] {0, 0, 5, 0, 5, 5, 0, 5}, 0.9f);
        DetectionBox small = new DetectionBox(
                new float[] {1, 1, 4, 1, 4, 4, 1, 4}, 0.9f);

        BgrImage first = workspace.crop(source, large, 0);
        byte[] retained = first.pixels();
        BgrImage reused = workspace.crop(source, small, 0);
        BgrImage otherSlot = workspace.crop(source, small, 1);

        Assert.assertSame(retained, reused.pixels());
        Assert.assertNotSame(retained, otherSlot.pixels());
        Assert.assertEquals(3, reused.width());
        Assert.assertEquals(3, reused.height());
        BgrImage expected = PerspectiveCrop.crop(source, small);
        Assert.assertArrayEquals(expected.pixels(),
                Arrays.copyOf(reused.pixels(), expected.pixels().length));
    }

    @Test
    public void batchWorkspaceKeepsMultipleCropsInOneArena() {
        byte[] pixels = new byte[6 * 6 * 3];
        for (int i = 0; i < pixels.length; i++) pixels[i] = (byte) i;
        BgrImage source = new BgrImage(pixels, 6, 6, 18);
        List<DetectionBox> boxes = Arrays.asList(
                new DetectionBox(new float[] {0, 0, 5, 0, 5, 5, 0, 5}, 0.9f),
                new DetectionBox(new float[] {1, 1, 4, 1, 4, 4, 1, 4}, 0.8f));
        PerspectiveCrop.Workspace workspace = new PerspectiveCrop.Workspace();
        List<BgrImage> crops = new ArrayList<BgrImage>();
        workspace.cropAll(source, boxes, crops);
        Assert.assertEquals(2, crops.size());
        Assert.assertSame(crops.get(0).pixels(), crops.get(1).pixels());
        Assert.assertNotEquals(crops.get(0).offset(), crops.get(1).offset());
        BgrImage expected = PerspectiveCrop.crop(source, boxes.get(1));
        byte[] actual = crops.get(1).pixels();
        for (int i = 0; i < expected.width() * expected.height() * 3; i++) {
            Assert.assertEquals(expected.pixels()[i] & 0xff,
                    actual[crops.get(1).offset() + i] & 0xff);
        }
    }
}
