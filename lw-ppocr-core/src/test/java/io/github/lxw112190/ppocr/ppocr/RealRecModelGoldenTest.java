package io.github.lxw112190.ppocr.ppocr;

import io.github.lxw112190.ppocr.golden.GoldenTestSupport;
import io.github.lxw112190.ppocr.model.LwmLoader;
import io.github.lxw112190.ppocr.model.LwmModel;
import io.github.lxw112190.ppocr.kernels.ScalarBackend;
import io.github.lxw112190.ppocr.runtime.CtcProjectionSession;
import io.github.lxw112190.ppocr.runtime.InferenceSession;
import io.github.lxw112190.ppocr.runtime.TensorShape;
import java.io.IOException;
import java.util.Arrays;
import java.util.Collections;
import org.junit.Assert;
import org.junit.Test;

/** Real Tiny REC graph parity against the pinned C runtime outputs. */
public final class RealRecModelGoldenTest {
    private static final String RESOURCE_ROOT = "/golden/rec/";
    private static final String MODEL_SHA256 = "59440146ae64068b70441f9c16b5878ccba21e75b41cb27220c8d9cc2d61e0ae";
    private static final String DICTIONARY_SHA256 = "5911341b8d8bdef3924e1ab7c85094f09cf05107d9983476a10c0ee57dac1d82";
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
            io.github.lxw112190.ppocr.image.BgrImage source =
                    new io.github.lxw112190.ppocr.image.BgrImage(pixels, 7, 5, 21);
            RecRecognitionResult result = recognizer.recognize(source);
            Assert.assertEquals("2", result.getText());
            Assert.assertEquals(1, result.getEmittedCount());
            Assert.assertEquals(68, result.getResizedWidth());
            Assert.assertEquals(0.217422441f, result.getScore(), 1.0e-4f);

            java.util.List<io.github.lxw112190.ppocr.image.BgrImage> batch =
                    Arrays.asList(source, source, source);
            RecRecognitionResult[] reusable = new RecRecognitionResult[batch.size()];
            recognizer.recognizeAllInto(batch, 3, reusable);
            RecRecognitionResult firstResult = reusable[0];
            recognizer.recognizeAllInto(batch, 2, reusable);
            for (RecRecognitionResult item : reusable) {
                Assert.assertEquals(firstResult.getText(), item.getText());
                Assert.assertEquals(firstResult.getScore(), item.getScore(), 0.0f);
                Assert.assertEquals(firstResult.getResizedWidth(), item.getResizedWidth());
            }

            String[] texts = new String[batch.size()];
            float[] scores = new float[batch.size()];
            int[] emittedCounts = new int[batch.size()];
            int[] resizedWidths = new int[batch.size()];
            recognizer.recognizeAllInto(batch, 2, texts, scores, emittedCounts, resizedWidths);
            for (int i = 0; i < batch.size(); i++) {
                Assert.assertEquals(firstResult.getText(), texts[i]);
                Assert.assertEquals(firstResult.getScore(), scores[i], 0.0f);
                Assert.assertEquals(firstResult.getEmittedCount(), emittedCounts[i]);
                Assert.assertEquals(firstResult.getResizedWidth(), resizedWidths[i]);
            }
        }
    }

    @Test
    public void terminalFusionMatchesDenseOutputAcrossWidthPolicy() throws Exception {
        int[] widths = {192, 320, 480, 640, 960};
        try (LwmModel model = LwmLoader.load(GoldenTestSupport.resource(
                RealRecModelGoldenTest.class, RESOURCE_ROOT + "rec.lwm"))) {
            for (int width : widths) {
                TensorShape inputShape = new TensorShape(1, 3, 48, width);
                java.util.List<TensorShape> shapes = Collections.singletonList(inputShape);
                try (InferenceSession dense = new InferenceSession(model, shapes, new ScalarBackend());
                     CtcProjectionSession compact = CtcProjectionSession.tryCreate(
                             model, shapes, new ScalarBackend(), CLASS_COUNT)) {
                    Assert.assertNotNull("fusion must support width=" + width, compact);
                    Assert.assertEquals(80L * CLASS_COUNT * Float.BYTES, compact.getPackedWeightBytes());
                    int timeSteps = compact.getTimeSteps();
                    float[] input = new float[3 * 48 * width];
                    for (int i = 0; i < input.length; i++) {
                        input[i] = ((i * 29 + width) % 509 - 254) / 253.0f;
                    }
                    float[] probabilities = new float[timeSteps * CLASS_COUNT];
                    dense.run(input, probabilities);
                    int[] classIds = new int[timeSteps];
                    float[] logits = new float[timeSteps];
                    float[] bestProbabilities = new float[timeSteps];
                    compact.run(input, classIds, logits, bestProbabilities);
                    for (int step = 0; step < timeSteps; step++) {
                        int row = step * CLASS_COUNT;
                        int expected = 0;
                        for (int column = 1; column < CLASS_COUNT; column++) {
                            if (probabilities[row + column] > probabilities[row + expected]) {
                                expected = column;
                            }
                        }
                        Assert.assertEquals("class id at width=" + width + ", step=" + step,
                                expected, classIds[step]);
                        Assert.assertEquals("score at width=" + width + ", step=" + step,
                                probabilities[row + expected], bestProbabilities[step], 1.0e-7f);
                    }
                    Assert.assertTrue("compact workspace must be smaller at width=" + width,
                            compact.getWorkspaceBytes() < dense.execution().workspacePlan().getTotalBytes());
                    Assert.assertEquals((long) timeSteps * CLASS_COUNT * 4L,
                            compact.getDenseOutputBytesAvoided());
                }
            }
        }
    }

    @Test
    public void terminalFusionCanBeDisabledForFallback() throws Exception {
        String previous = System.getProperty(CtcProjectionSession.DISABLE_PROPERTY);
        System.setProperty(CtcProjectionSession.DISABLE_PROPERTY, "true");
        try (LwmModel model = LwmLoader.load(GoldenTestSupport.resource(
                RealRecModelGoldenTest.class, RESOURCE_ROOT + "rec.lwm"))) {
            CtcProjectionSession compact = CtcProjectionSession.tryCreate(model,
                    Collections.singletonList(new TensorShape(1, 3, 48, 320)),
                    new ScalarBackend(), CLASS_COUNT);
            Assert.assertNull(compact);
        } finally {
            if (previous == null) System.clearProperty(CtcProjectionSession.DISABLE_PROPERTY);
            else System.setProperty(CtcProjectionSession.DISABLE_PROPERTY, previous);
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
