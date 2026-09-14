package io.github.lxw112190.ppocr.ppocr;

import io.github.lxw112190.ppocr.golden.GoldenTestSupport;
import io.github.lxw112190.ppocr.kernels.ScalarBackend;
import io.github.lxw112190.ppocr.model.LwmLoader;
import io.github.lxw112190.ppocr.model.LwmModel;
import io.github.lxw112190.ppocr.runtime.TensorShape;
import java.util.Collections;
import org.junit.Assert;
import org.junit.Test;

/** Regression coverage for the height-first DET cache key and width-first context API. */
public final class DetSessionCacheTest {
    @Test
    public void preservesWidthAndHeightWhenCreatingDynamicSession() {
        try (LwmModel model = LwmLoader.load(GoldenTestSupport.resource(
                DetSessionCacheTest.class, "/golden/det/det.lwm"));
             DetSessionCache cache = new DetSessionCache(3)) {
            DetSessionContext context = cache.getOrCreate(new DetShapeKey(288, 960), model,
                    new ScalarBackend());
            Assert.assertEquals(960, context.width);
            Assert.assertEquals(288, context.height);
            Assert.assertEquals(960, context.preprocess.getResizedWidth());
            Assert.assertEquals(288, context.preprocess.getResizedHeight());
            Assert.assertEquals(new TensorShape(1, 3, 288, 960),
                    context.session.execution().shapes().get(model.getGraphInputs().get(0)));
        }
    }
}
