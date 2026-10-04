package io.github.lxw112190.ppocr.model;

import java.io.IOException;
import java.io.InputStream;
import java.io.PushbackInputStream;
import java.nio.file.Files;
import java.nio.file.Path;

/** Content-based LWM / ONNX loading; file suffixes are not trusted. */
public final class ModelLoader {
    private ModelLoader() { }
    public static LwmModel load(Path path) { return load(path, RuntimeLimits.defaults()); }
    public static LwmModel load(Path path, RuntimeLimits limits) {
        if (path == null || limits == null) throw argument();
        try (InputStream in = Files.newInputStream(path)) {
            byte[] prefix = new byte[4]; int count = prefix(in, prefix);
            return isLwm(prefix, count) ? LwmLoader.load(path, limits) : OnnxLoader.load(path, limits);
        } catch (IOException e) { throw new OcrException(OcrErrorCode.IO_ERROR, "unable to read model: " + path, e); }
    }
    public static LwmModel load(InputStream input) { return load(input, RuntimeLimits.defaults()); }
    /** Leaves the caller-owned stream open. */
    public static LwmModel load(InputStream input, RuntimeLimits limits) {
        if (input == null || limits == null) throw argument();
        PushbackInputStream in = new PushbackInputStream(input, 4);
        try {
            byte[] prefix = new byte[4]; int count = prefix(in, prefix); in.unread(prefix, 0, count);
            return isLwm(prefix, count) ? LwmLoader.load(in, limits) : OnnxLoader.load(in, limits);
        } catch (IOException e) { throw new OcrException(OcrErrorCode.IO_ERROR, "unable to read model stream", e); }
    }
    private static int prefix(InputStream in, byte[] prefix) throws IOException {
        int count = 0;
        while (count < prefix.length) { int b = in.read(); if (b < 0) break; prefix[count++] = (byte) b; }
        return count;
    }
    private static boolean isLwm(byte[] p, int n) { return n == 4 && p[0] == 'L' && p[1] == 'W' && p[2] == 'M' && p[3] == '0'; }
    private static OcrException argument() { return new OcrException(OcrErrorCode.INVALID_ARGUMENT, "model source and limits are required"); }
}
