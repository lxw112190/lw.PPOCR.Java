package io.github.lxw112190.ppocr.imageio;

import io.github.lxw112190.ppocr.image.BgrImage;
import io.github.lxw112190.ppocr.model.OcrErrorCode;
import io.github.lxw112190.ppocr.model.OcrException;
import java.awt.image.BufferedImage;
import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import javax.imageio.ImageIO;

/** Standard-library image decoder kept outside the inference core. */
public final class ImageIoLoader {
    private ImageIoLoader() { }

    public static BgrImage load(Path path) {
        if (path == null) throw new OcrException(OcrErrorCode.INVALID_ARGUMENT, "image path is required");
        try (InputStream input = Files.newInputStream(path)) {
            return load(input);
        } catch (IOException e) {
            throw new OcrException(OcrErrorCode.IO_ERROR, "unable to read image: " + path, e);
        }
    }

    /** Reads the stream without closing it; ownership remains with the caller. */
    public static BgrImage load(InputStream input) {
        if (input == null) throw new OcrException(OcrErrorCode.INVALID_ARGUMENT, "image stream is required");
        try {
            BufferedImage image = ImageIO.read(input);
            if (image == null) {
                throw new OcrException(OcrErrorCode.IO_ERROR, "image format is unsupported");
            }
            return BufferedImageAdapter.toBgr(image);
        } catch (IOException e) {
            throw new OcrException(OcrErrorCode.IO_ERROR, "unable to decode image", e);
        }
    }
}
