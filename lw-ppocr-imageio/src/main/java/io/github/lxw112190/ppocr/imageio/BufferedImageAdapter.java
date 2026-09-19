package io.github.lxw112190.ppocr.imageio;

import io.github.lxw112190.ppocr.image.BgrImage;
import io.github.lxw112190.ppocr.model.OcrErrorCode;
import io.github.lxw112190.ppocr.model.OcrException;
import java.awt.image.BufferedImage;
import java.awt.image.ComponentSampleModel;
import java.awt.image.DataBufferByte;

/** Converts any Java BufferedImage color model into a packed BGR view. */
public final class BufferedImageAdapter {
    private BufferedImageAdapter() { }

    public static BgrImage toBgr(BufferedImage image) {
        if (image == null || image.getWidth() <= 0 || image.getHeight() <= 0) {
            throw new OcrException(OcrErrorCode.INVALID_ARGUMENT, "BufferedImage is invalid");
        }
        BgrImage direct = tryDirectBgr(image);
        if (direct != null) return direct;
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

    private static BgrImage tryDirectBgr(BufferedImage image) {
        if (image.getType() != BufferedImage.TYPE_3BYTE_BGR) return null;
        if (!(image.getRaster().getDataBuffer() instanceof DataBufferByte) ||
                !(image.getRaster().getSampleModel() instanceof ComponentSampleModel)) return null;
        if (image.getRaster().getMinX() != 0 || image.getRaster().getMinY() != 0 ||
                image.getRaster().getSampleModelTranslateX() != 0 ||
                image.getRaster().getSampleModelTranslateY() != 0) return null;
        ComponentSampleModel sampleModel =
                (ComponentSampleModel) image.getRaster().getSampleModel();
        int[] bands = sampleModel.getBandOffsets();
        if (sampleModel.getPixelStride() != 3 || bands.length != 3 ||
                bands[0] != 2 || bands[1] != 1 || bands[2] != 0) return null;
        DataBufferByte dataBuffer = (DataBufferByte) image.getRaster().getDataBuffer();
        if (dataBuffer.getNumBanks() != 1) return null;
        return new BgrImage(dataBuffer.getData(), dataBuffer.getOffset(), image.getWidth(),
                image.getHeight(), sampleModel.getScanlineStride());
    }
}
