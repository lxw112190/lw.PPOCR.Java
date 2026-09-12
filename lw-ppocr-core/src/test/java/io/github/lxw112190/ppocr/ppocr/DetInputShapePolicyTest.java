package io.github.lxw112190.ppocr.ppocr;

import io.github.lxw112190.ppocr.image.BgrImage;
import org.junit.Assert;
import org.junit.Test;

public final class DetInputShapePolicyTest {
    @Test
    public void preservesAspectRatioWhileAligningToThirtyTwo() {
        DetInputShape shape = DetInputShapePolicy.choose(image(100, 50), 64);
        Assert.assertEquals(64, shape.getInputWidth());
        Assert.assertEquals(32, shape.getInputHeight());
        Assert.assertEquals(0.64f, shape.getScaleX(), 0.00001f);
        Assert.assertEquals(0.64f, shape.getScaleY(), 0.00001f);
    }

    @Test
    public void clampsTheLongSideAndUsesMinimumDimension() {
        DetInputShape wide = DetInputShapePolicy.choose(image(2000, 1000), 960);
        Assert.assertEquals(960, wide.getInputWidth());
        Assert.assertEquals(480, wide.getInputHeight());

        DetInputShape tiny = DetInputShapePolicy.choose(image(3, 7), 960);
        Assert.assertEquals(32, tiny.getInputWidth());
        Assert.assertEquals(32, tiny.getInputHeight());
    }

    private static BgrImage image(int width, int height) {
        return new BgrImage(new byte[width * height * 3], width, height, width * 3);
    }
}
