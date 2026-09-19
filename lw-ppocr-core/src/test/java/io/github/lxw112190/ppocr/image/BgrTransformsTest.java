package io.github.lxw112190.ppocr.image;

import org.junit.Assert;
import org.junit.Test;

public final class BgrTransformsTest {
    @Test
    public void rotatesPixelsWithoutReadingRowPaddingOrMutatingSource() {
        byte[] pixels = new byte[] {
                1, 2, 3, 4, 5, 6, 99, 99,
                7, 8, 9, 10, 11, 12, 88, 88
        };
        BgrImage source = new BgrImage(pixels, 2, 2, 8);
        BgrImage rotated = BgrTransforms.rotate180(source);
        Assert.assertArrayEquals(new byte[] {
                10, 11, 12, 7, 8, 9,
                4, 5, 6, 1, 2, 3
        }, rotated.pixels());
        Assert.assertArrayEquals(new byte[] {
                1, 2, 3, 4, 5, 6, 99, 99,
                7, 8, 9, 10, 11, 12, 88, 88
        }, source.pixels());
    }

    @Test
    public void rotatesOffsetImageView() {
        byte[] pixels = new byte[] {
                77, 77,
                1, 2, 3, 4, 5, 6, 99, 99,
                7, 8, 9, 10, 11, 12, 88, 88,
                66, 66
        };
        BgrImage source = new BgrImage(pixels, 2, 2, 2, 8);
        BgrImage rotated = BgrTransforms.rotate180(source);
        Assert.assertArrayEquals(new byte[] {
                10, 11, 12, 7, 8, 9,
                4, 5, 6, 1, 2, 3
        }, rotated.pixels());
    }
}
