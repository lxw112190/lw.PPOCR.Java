package io.github.lxw112190.ppocr.ppocr;

import io.github.lxw112190.ppocr.model.LwmLoader;
import io.github.lxw112190.ppocr.model.LwmModel;
import io.github.lxw112190.ppocr.runtime.InferenceSession;
import io.github.lxw112190.ppocr.runtime.TensorShape;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.Collections;
import org.junit.Assert;
import org.junit.Test;

/** Real Tiny DET graph parity against the pinned C runtime outputs. */
public final class RealDetModelGoldenTest {
    private static final String RESOURCE_ROOT = "/golden/det/";
    private static final String MODEL_SHA256 = "ba9164d371ac7003f90710c3106a344aeb906df2b0f1e7617fcf4608fa8cd66c";

    @Test
    public void matchesPinnedCGraphAtDynamicShapes() throws Exception {
        String manifest = readText(RESOURCE_ROOT + "manifest.json");
        Assert.assertTrue(manifest.contains("\"source_commit\": \"9b31f1b\""));
        Assert.assertTrue(manifest.contains(MODEL_SHA256));
        Assert.assertEquals(MODEL_SHA256, sha256(readBytes(RESOURCE_ROOT + "det.lwm")));
        try (LwmModel model = LwmLoader.load(resource(RESOURCE_ROOT + "det.lwm"))) {
            assertGraphCase(model, 32, 32, "shape-32x32.c-output.f32");
            assertGraphCase(model, 32, 64, "shape-32x64.c-output.f32");
        }
    }

    private static void assertGraphCase(LwmModel model, int height, int width, String outputFile)
            throws IOException {
        float[] input = new float[3 * height * width];
        for (int i = 0; i < input.length; i++) input[i] = ((i * 23) % 269 - 134) / 134.0f;
        float[] expected = readFloats(RESOURCE_ROOT + outputFile);
        Assert.assertEquals(height * width, expected.length);
        try (InferenceSession session = new InferenceSession(model,
                Collections.singletonList(new TensorShape(1, 3, height, width)))) {
            TensorShape outputShape = session.execution().shapes().get(model.getGraphOutputs().get(0));
            Assert.assertEquals(4, outputShape.getRank());
            Assert.assertEquals(height, outputShape.get(2));
            Assert.assertEquals(width, outputShape.get(3));
            float[] actual = new float[expected.length];
            session.run(input, actual);
            double sumAbs = 0.0;
            for (int i = 0; i < actual.length; i++) {
                Assert.assertTrue("non-finite DET output", Float.isFinite(actual[i]));
                Assert.assertTrue("DET probability below zero", actual[i] >= 0.0f);
                Assert.assertTrue("DET probability above one", actual[i] <= 1.0f);
                float difference = Math.abs(actual[i] - expected[i]);
                sumAbs += difference;
                Assert.assertEquals("shape=" + height + "x" + width + " index=" + i,
                        expected[i], actual[i], 3.0e-3f * Math.max(1.0f, Math.abs(expected[i])) + 3.0e-5f);
            }
            Assert.assertTrue("mean DET error is too large", sumAbs / actual.length <= 3.0e-4);
        }
    }

    private static InputStream resource(String name) {
        InputStream input = RealDetModelGoldenTest.class.getResourceAsStream(name);
        if (input == null) throw new AssertionError("missing Golden resource: " + name);
        return input;
    }

    private static byte[] readBytes(String name) throws IOException {
        try (InputStream input = resource(name); ByteArrayOutputStream output = new ByteArrayOutputStream()) {
            byte[] buffer = new byte[8192];
            int count;
            while ((count = input.read(buffer)) >= 0) if (count != 0) output.write(buffer, 0, count);
            return output.toByteArray();
        }
    }

    private static String readText(String name) throws IOException {
        return new String(readBytes(name), StandardCharsets.UTF_8);
    }

    private static float[] readFloats(String name) throws IOException {
        byte[] bytes = readBytes(name);
        Assert.assertEquals(0, bytes.length % 4);
        float[] values = new float[bytes.length / 4];
        ByteBuffer buffer = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN);
        for (int i = 0; i < values.length; i++) values[i] = buffer.getFloat();
        return values;
    }

    private static String sha256(byte[] bytes) throws NoSuchAlgorithmException {
        byte[] digest = MessageDigest.getInstance("SHA-256").digest(bytes);
        StringBuilder result = new StringBuilder(digest.length * 2);
        for (byte value : digest) result.append(String.format("%02x", value & 0xff));
        return result.toString();
    }
}
