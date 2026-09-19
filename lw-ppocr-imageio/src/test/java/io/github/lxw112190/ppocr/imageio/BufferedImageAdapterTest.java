package io.github.lxw112190.ppocr.imageio;

import io.github.lxw112190.ppocr.image.BgrImage;
import java.awt.image.BufferedImage;
import java.awt.image.DataBufferByte;
import org.junit.Assert;
import org.junit.Test;

public class BufferedImageAdapterTest {
    @Test
    public void convertsArgbToPackedBgr() {
        BufferedImage source = new BufferedImage(2, 1, BufferedImage.TYPE_INT_ARGB);
        source.setRGB(0, 0, 0xFF112233);
        source.setRGB(1, 0, 0x80445566);

        BgrImage result = BufferedImageAdapter.toBgr(source);

        Assert.assertEquals(2, result.width());
        Assert.assertEquals(1, result.height());
        Assert.assertEquals(6, result.stride());
        Assert.assertArrayEquals(new byte[] {0x33, 0x22, 0x11, 0x66, 0x55, 0x44}, result.pixels());
    }

    @Test
    public void exposesStandardThreeByteBgrWithoutCopying() {
        BufferedImage source = new BufferedImage(2, 1, BufferedImage.TYPE_3BYTE_BGR);
        source.setRGB(0, 0, 0xFF112233);
        source.setRGB(1, 0, 0xFF445566);

        BgrImage result = BufferedImageAdapter.toBgr(source);

        Assert.assertSame(((DataBufferByte) source.getRaster().getDataBuffer()).getData(),
                result.pixels());
        int base = result.offset();
        Assert.assertEquals(0x33, result.pixels()[base] & 0xff);
        Assert.assertEquals(0x22, result.pixels()[base + 1] & 0xff);
        Assert.assertEquals(0x11, result.pixels()[base + 2] & 0xff);
        Assert.assertEquals(0x66, result.pixels()[base + 3] & 0xff);
    }
}
