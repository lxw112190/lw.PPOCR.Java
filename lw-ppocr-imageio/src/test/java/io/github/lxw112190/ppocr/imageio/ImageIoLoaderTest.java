package io.github.lxw112190.ppocr.imageio;

import io.github.lxw112190.ppocr.image.BgrImage;
import java.awt.image.BufferedImage;
import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import javax.imageio.ImageIO;
import org.junit.Assert;
import org.junit.Test;

public class ImageIoLoaderTest {
    @Test
    public void decodesPngFromCallerOwnedStream() throws Exception {
        BufferedImage source = new BufferedImage(1, 1, BufferedImage.TYPE_INT_RGB);
        source.setRGB(0, 0, 0xFFABCDEF);
        ByteArrayOutputStream encoded = new ByteArrayOutputStream();
        Assert.assertTrue(ImageIO.write(source, "png", encoded));

        BgrImage result = ImageIoLoader.load(new ByteArrayInputStream(encoded.toByteArray()));

        Assert.assertArrayEquals(new byte[] {(byte) 0xEF, (byte) 0xCD, (byte) 0xAB}, result.pixels());
    }

    @Test(expected = io.github.lxw112190.ppocr.model.OcrException.class)
    public void rejectsMissingPipelineBeforeDecoding() {
        PaddleOcrImageIo.recognize(null, new BufferedImage(1, 1, BufferedImage.TYPE_INT_RGB));
    }
}
