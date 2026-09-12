package io.github.lxw112190.ppocr.ppocr;

import io.github.lxw112190.ppocr.model.LwmLoader;
import io.github.lxw112190.ppocr.model.LwmModel;
import io.github.lxw112190.ppocr.runtime.InferenceSession;
import io.github.lxw112190.ppocr.runtime.TensorShape;
import java.io.ByteArrayOutputStream;
import java.io.DataInputStream;
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

/** Real Tiny REC graph parity against the pinned C runtime outputs. */
public final class RealRecModelGoldenTest {
    private static final String RESOURCE_ROOT = "/golden/rec/";
    private static final String MODEL_SHA256 = "59440146ae64068b70441f9c16b5878ccba21e75b41cb27220c8d9cc2d61e0ae";
    private static final String DICTIONARY_SHA256 = "46e1b34ef45684cb46d75ac76d355341fe7f0a2c38d6ee02e63ae6b3878019fc";
    private static final int CLASS_COUNT = 6906;

    @Test
    public void matchesPinnedCGraphAtDynamicWidths() throws Exception {
        String manifest = readText(RESOURCE_ROOT + "manifest.json");
        Assert.assertTrue("manifest must pin the C source commit", manifest.contains("\"source_commit\": \"9b31f1b\""));
        Assert.assertTrue("manifest must pin the model hash", manifest.contains(MODEL_SHA256));
        Assert.assertTrue("manifest must pin the dictionary hash", manifest.contains(DICTIONARY_SHA256));
        Assert.assertEquals(MODEL_SHA256, sha256(readBytes(RESOURCE_ROOT + "rec.lwm")));
        Assert.assertEquals(DICTIONARY_SHA256, sha256(readBytes(RESOURCE_ROOT + "ppocr_keys.txt")));

        try (LwmModel model = LwmLoader.load(resource(RESOURCE_ROOT + "rec.lwm"))) {
            assertGraphCase(model, 7, 1, "width-7.c-output.f32");
            assertGraphCase(model, 17, 2, "width-17.c-output.f32");
        }
    }

    @Test
    public void matchesPinnedCPipelineForBgrCrop() throws Exception {
        String expected = readText(RESOURCE_ROOT + "crop-7x5.expected.json");
        Assert.assertTrue(expected.contains("\"text\": \"2\""));
        Assert.assertTrue(expected.contains("\"resized_width\": 68"));
        byte[] pixels = readBytes(RESOURCE_ROOT + "crop-7x5.bgr");
        Assert.assertEquals(7 * 5 * 3, pixels.length);
        try (LwmModel model = LwmLoader.load(resource(RESOURCE_ROOT + "rec.lwm"));
             PaddleOcrDictionary dictionary = PaddleOcrDictionary.load(resource(RESOURCE_ROOT + "ppocr_keys.txt"));
             PaddleOcrRecognizer recognizer = new PaddleOcrRecognizer(model, dictionary)) {
            RecRecognitionResult result = recognizer.recognize(new io.github.lxw112190.ppocr.image.BgrImage(
                    pixels, 7, 5, 21));
            Assert.assertEquals("2", result.getText());
            Assert.assertEquals(1, result.getEmittedCount());
            Assert.assertEquals(68, result.getResizedWidth());
            Assert.assertEquals(0.217422441f, result.getScore(), 1.0e-4f);
        }
    }

    private static void assertGraphCase(LwmModel model, int width, int timeSteps, String outputFile)
            throws IOException {
        int inputLength = 3 * 48 * width;
        float[] input = new float[inputLength];
        for (int i = 0; i < input.length; i++) {
            input[i] = ((i * 17) % 257 - 128) / 127.0f;
        }
        float[] expected = readFloats(RESOURCE_ROOT + outputFile);
        Assert.assertEquals(timeSteps * CLASS_COUNT, expected.length);
        try (InferenceSession session = new InferenceSession(model,
                Collections.singletonList(new TensorShape(1, 3, 48, width)))) {
            Assert.assertEquals(timeSteps, session.execution().shapes()
                    .get(model.getGraphOutputs().get(0)).get(session.execution().shapes()
                            .get(model.getGraphOutputs().get(0)).getRank() - 2));
            float[] actual = new float[expected.length];
            session.run(input, actual);
            float maxAbs = 0.0f;
            double sumAbs = 0.0;
            for (int i = 0; i < actual.length; i++) {
                float difference = Math.abs(actual[i] - expected[i]);
                maxAbs = Math.max(maxAbs, difference);
                sumAbs += difference;
                Assert.assertEquals("width=" + width + " index=" + i,
                        expected[i], actual[i], 3.0e-3f * Math.max(1.0f, Math.abs(expected[i])) + 3.0e-5f);
            }
            Assert.assertTrue("mean error too large for width=" + width,
                    sumAbs / actual.length <= 3.0e-4);
            Assert.assertTrue("max error too large for width=" + width, maxAbs <= 3.0e-2f);
        }
    }

    private static InputStream resource(String name) {
        InputStream input = RealRecModelGoldenTest.class.getResourceAsStream(name);
        if (input == null) throw new AssertionError("missing Golden resource: " + name);
        return input;
    }

    private static byte[] readBytes(String name) throws IOException {
        try (InputStream input = resource(name); ByteArrayOutputStream output = new ByteArrayOutputStream()) {
            byte[] buffer = new byte[8192];
            int count;
            while ((count = input.read(buffer)) >= 0) {
                if (count != 0) output.write(buffer, 0, count);
            }
            return output.toByteArray();
        }
    }

    private static String readText(String name) throws IOException {
        return new String(readBytes(name), StandardCharsets.UTF_8);
    }

    private static float[] readFloats(String name) throws IOException {
        byte[] bytes = readBytes(name);
        Assert.assertEquals("Golden f32 file must be aligned", 0, bytes.length % 4);
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
