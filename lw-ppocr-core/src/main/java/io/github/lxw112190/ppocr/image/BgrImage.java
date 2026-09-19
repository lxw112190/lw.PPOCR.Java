package io.github.lxw112190.ppocr.image;

import io.github.lxw112190.ppocr.model.OcrErrorCode;
import io.github.lxw112190.ppocr.model.OcrException;

/** Non-owning BGR uint8 image view. The backing array must remain unchanged while used. */
public final class BgrImage {
    private final byte[] pixels;
    private final int offset;
    private final int width;
    private final int height;
    private final int stride;

    public BgrImage(byte[] pixels, int width, int height, int stride) {
        this(pixels, 0, width, height, stride);
    }

    /** Creates a non-owning view into a BGR byte array. */
    public BgrImage(byte[] pixels, int offset, int width, int height, int stride) {
        if (pixels == null || offset < 0 || width <= 0 || height <= 0 ||
                stride < width * 3L || offset > pixels.length ||
                (long) offset + (long) (height - 1) * stride + (long) width * 3 > pixels.length) {
            throw new OcrException(OcrErrorCode.INVALID_ARGUMENT, "invalid BGR image buffer or dimensions");
        }
        this.pixels = pixels;
        this.offset = offset;
        this.width = width;
        this.height = height;
        this.stride = stride;
    }

    public byte[] pixels() { return pixels; }
    public int offset() { return offset; }
    public int width() { return width; }
    public int height() { return height; }
    public int stride() { return stride; }
}
