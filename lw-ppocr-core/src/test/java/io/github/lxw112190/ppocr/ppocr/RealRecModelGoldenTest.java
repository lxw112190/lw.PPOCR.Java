package io.github.lxw112190.ppocr.ppocr;

import io.github.lxw112190.ppocr.golden.GoldenTestSupport;
import io.github.lxw112190.ppocr.model.LwmLoader;
import io.github.lxw112190.ppocr.model.LwmModel;
import io.github.lxw112190.ppocr.runtime.InferenceSession;
import io.github.lxw112190.ppocr.runtime.TensorShape;
import java.io.IOException;
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
        String manifest = GoldenTestSupport.readText(RealRecModelGoldenTest.class, RESOURCE_ROOT + "manifest.json");
        Assert.assertTrue("manifest must pin the C source commit", manifest.contains("\"source_commit\": \"9b31f1b\""));
        Assert.assertTrue("manifest must pin the model hash", manifest.contains(MODEL_SHA256));
        Assert.assertTrue("manifest must pin the dictionary hash", manifest.contains(DICTIONARY_SHA256));
        Assert.assertEquals(MODEL_SHA256, GoldenTestSupport.sha256(
                GoldenTestSupport.readBytes(RealRecModelGoldenTest.class, RESOURCE_ROOT + "rec.lwm")));
        Assert.assertEquals(DICTIONARY_SHA256, GoldenTestSupport.sha256(
                GoldenTestSupport.readBytes(RealRecModelGoldenTest.class, RESOURCE_ROOT + "ppocr_keys.txt")));

        try (LwmModel model = LwmLoader.load(GoldenTestSupport.resource(
                RealRecModelGoldenTest.class, RESOURCE_ROOT + "rec.lwm"))) {
            assertGraphCase(model, 7, 1, "width-7.c-output.f32");
            assertGraphCase(model, 17, 2, "width-17.c-output.f32");
        }
    }

    @Test
    public void matchesPinnedCPipelineForBgrCrop() throws Exception {
        String expected = GoldenTestSupport.readText(RealRecModelGoldenTest.class,
                RESOURCE_ROOT + "crop-7x5.expected.json");
        Assert.assertTrue(expected.contains("\"text\": \"2\""));
        Assert.assertTrue(expected.contains("\"resized_width\": 68"));
        byte[] pixels = GoldenTestSupport.readBytes(RealRecModelGoldenTest.class, RESOURCE_ROOT + "crop-7x5.bgr");
        Assert.assertEquals(7 * 5 * 3, pixels.length);
        try (LwmModel model = LwmLoader.load(GoldenTestSupport.resource(
                     RealRecModelGoldenTest.class, RESOURCE_ROOT + "rec.lwm"));
             PaddleOcrDictionary dictionary = PaddleOcrDictionary.load(GoldenTestSupport.resource(
                     RealRecModelGoldenTest.class, RESOURCE_ROOT + "ppocr_keys.txt"));
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
        float[] expected = GoldenTestSupport.readFloat32LittleEndian(RealRecModelGoldenTest.class,
                RESOURCE_ROOT + outputFile);
        Assert.assertEquals(timeSteps * CLASS_COUNT, expected.length);
        try (InferenceSession session = new InferenceSession(model,
                Collections.singletonList(new TensorShape(1, 3, 48, width)))) {
            Assert.assertEquals(timeSteps, session.execution().shapes()
                    .get(model.getGraphOutputs().get(0)).get(session.execution().shapes()
                            .get(model.getGraphOutputs().get(0)).getRank() - 2));
            float[] actual = new float[expected.length];
            session.run(input, actual);
            GoldenTestSupport.assertTensorClose("width=" + width, expected, actual,
                    3.0e-3f, 3.0e-5f, 3.0e-4, 3.0e-2f);
        }
    }
}
