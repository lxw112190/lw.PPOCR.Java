package io.github.lxw112190.ppocr.imageio;

import io.github.lxw112190.ppocr.image.BgrImage;
import io.github.lxw112190.ppocr.model.OcrErrorCode;
import io.github.lxw112190.ppocr.model.OcrException;
import java.awt.image.BufferedImage;

/** Converts any Java BufferedImage color model into a packed BGR view. */
public final class BufferedImageAdapter {
    private BufferedImageAdapter() { }

    public static BgrImage toBgr(BufferedImage image) {
        if (image == null || image.getWidth() <= 0 || image.getHeight() <= 0) {
            throw new OcrException(OcrErrorCode.INVALID_ARGUMENT, "BufferedImage is invalid");
        }
        int width = image.getWidth();
        int height = image.getHeight();
        long byteCount = (long) width * height * 3L;
        if (byteCount > Integer.MAX_VALUE) {
            throw new OcrException(OcrErrorCode.RESOURCE_LIMIT, "image is too large");
        }
        byte[] pixels = new byte[(int) byteCount];
        int[] argb = new int[width];
        for (int y = 0; y < height; y++) {
            image.getRGB(0, y, width, 1, argb, 0, width);
            int destination = y * width * 3;
            for (int x = 0; x < width; x++) {
                int value = argb[x];
                pixels[destination++] = (byte) value;
                pixels[destination++] = (byte) (value >>> 8);
                pixels[destination++] = (byte) (value >>> 16);
            }
        }
        return new BgrImage(pixels, width, height, width * 3);
    }
}
