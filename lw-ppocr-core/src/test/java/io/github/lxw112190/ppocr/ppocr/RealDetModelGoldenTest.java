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

/** Real Tiny DET graph parity against the pinned C runtime outputs. */
public final class RealDetModelGoldenTest {
    private static final String RESOURCE_ROOT = "/golden/det/";
    private static final String MODEL_SHA256 = "ba9164d371ac7003f90710c3106a344aeb906df2b0f1e7617fcf4608fa8cd66c";

    @Test
    public void matchesPinnedCGraphAtDynamicShapes() throws Exception {
        String manifest = GoldenTestSupport.readText(RealDetModelGoldenTest.class, RESOURCE_ROOT + "manifest.json");
        Assert.assertTrue(manifest.contains("\"source_commit\": \"9b31f1b\""));
        Assert.assertTrue(manifest.contains(MODEL_SHA256));
        Assert.assertEquals(MODEL_SHA256, GoldenTestSupport.sha256(
                GoldenTestSupport.readBytes(RealDetModelGoldenTest.class, RESOURCE_ROOT + "det.lwm")));
        try (LwmModel model = LwmLoader.load(GoldenTestSupport.resource(
                RealDetModelGoldenTest.class, RESOURCE_ROOT + "det.lwm"))) {
            assertGraphCase(model, 32, 32, "shape-32x32.c-output.f32");
            assertGraphCase(model, 32, 64, "shape-32x64.c-output.f32");
        }
    }

    @Test
    public void acceptsDynamicBatchModelFacade() {
        try (LwmModel model = LwmLoader.load(GoldenTestSupport.resource(
                RealDetModelGoldenTest.class, RESOURCE_ROOT + "det.lwm"));
             PaddleOcrDetector detector = new PaddleOcrDetector(model)) {
            Assert.assertNotNull(detector);
        }
    }

    private static void assertGraphCase(LwmModel model, int height, int width, String outputFile)
            throws IOException {
        float[] input = new float[3 * height * width];
        for (int i = 0; i < input.length; i++) input[i] = ((i * 23) % 269 - 134) / 134.0f;
        float[] expected = GoldenTestSupport.readFloat32LittleEndian(RealDetModelGoldenTest.class,
                RESOURCE_ROOT + outputFile);
        Assert.assertEquals(height * width, expected.length);
        try (InferenceSession session = new InferenceSession(model,
                Collections.singletonList(new TensorShape(1, 3, height, width)))) {
            TensorShape outputShape = session.execution().shapes().get(model.getGraphOutputs().get(0));
            Assert.assertEquals(4, outputShape.getRank());
            Assert.assertEquals(height, outputShape.get(2));
            Assert.assertEquals(width, outputShape.get(3));
            float[] actual = new float[expected.length];
            session.run(input, actual);
            for (int i = 0; i < actual.length; i++) {
                Assert.assertTrue("non-finite DET output", Float.isFinite(actual[i]));
                Assert.assertTrue("DET probability below zero", actual[i] >= 0.0f);
                Assert.assertTrue("DET probability above one", actual[i] <= 1.0f);
            }
            GoldenTestSupport.assertTensorClose("shape=" + height + "x" + width, expected, actual,
                    3.0e-3f, 3.0e-5f, 3.0e-4, 3.0e-2f);
        }
    }
}
