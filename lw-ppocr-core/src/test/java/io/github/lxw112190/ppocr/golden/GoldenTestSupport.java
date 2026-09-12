package io.github.lxw112190.ppocr.golden;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import org.junit.Assert;

/** Shared resource, little-endian tensor, hash, and error checks for Golden tests. */
public final class GoldenTestSupport {
    private GoldenTestSupport() { }

    public static InputStream resource(Class<?> owner, String name) {
        InputStream input = owner.getResourceAsStream(name);
        if (input == null) throw new AssertionError("missing Golden resource: " + name);
        return input;
    }

    public static byte[] readBytes(Class<?> owner, String name) throws IOException {
        try (InputStream input = resource(owner, name); ByteArrayOutputStream output = new ByteArrayOutputStream()) {
            byte[] buffer = new byte[8192];
            int count;
            while ((count = input.read(buffer)) >= 0) if (count != 0) output.write(buffer, 0, count);
            return output.toByteArray();
        }
    }

    public static String readText(Class<?> owner, String name) throws IOException {
        return new String(readBytes(owner, name), StandardCharsets.UTF_8);
    }

    public static float[] readFloat32LittleEndian(Class<?> owner, String name) throws IOException {
        byte[] bytes = readBytes(owner, name);
        Assert.assertEquals("Golden f32 file must be aligned", 0, bytes.length % 4);
        float[] values = new float[bytes.length / 4];
        ByteBuffer buffer = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN);
        for (int i = 0; i < values.length; i++) values[i] = buffer.getFloat();
        return values;
    }

    public static void assertTensorClose(String label, float[] expected, float[] actual,
                                         float relativeTolerance, float absoluteTolerance,
                                         double meanTolerance, float maxTolerance) {
        Assert.assertEquals(label + " element count", expected.length, actual.length);
        float maxAbs = 0.0f;
        double sumAbs = 0.0;
        for (int i = 0; i < actual.length; i++) {
            float difference = Math.abs(actual[i] - expected[i]);
            maxAbs = Math.max(maxAbs, difference);
            sumAbs += difference;
            Assert.assertEquals(label + " index=" + i, expected[i], actual[i],
                    relativeTolerance * Math.max(1.0f, Math.abs(expected[i])) + absoluteTolerance);
        }
        double meanAbs = actual.length == 0 ? 0.0 : sumAbs / actual.length;
        Assert.assertTrue(label + " mean absolute error=" + meanAbs, meanAbs <= meanTolerance);
        Assert.assertTrue(label + " max absolute error=" + maxAbs, maxAbs <= maxTolerance);
    }

    public static long checksum(float[] values) {
        long hash = 0xcbf29ce484222325L;
        for (float value : values) {
            hash ^= Float.floatToIntBits(value) & 0xffffffffL;
            hash *= 0x100000001b3L;
        }
        return hash;
    }

    public static String sha256(byte[] bytes) throws NoSuchAlgorithmException {
        byte[] digest = MessageDigest.getInstance("SHA-256").digest(bytes);
        StringBuilder result = new StringBuilder(digest.length * 2);
        for (byte value : digest) result.append(String.format("%02x", value & 0xff));
        return result.toString();
    }
}
