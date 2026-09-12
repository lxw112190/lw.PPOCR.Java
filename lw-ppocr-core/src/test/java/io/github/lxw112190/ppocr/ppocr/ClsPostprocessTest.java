package io.github.lxw112190.ppocr.ppocr;

import io.github.lxw112190.ppocr.model.OcrErrorCode;
import io.github.lxw112190.ppocr.model.OcrException;
import org.junit.Assert;
import org.junit.Test;

public final class ClsPostprocessTest {
    @Test
    public void derivesOrientationAndAppliesStrictRotationThreshold() {
        ClsClassificationResult result = ClsPostprocess.decode(new float[] {0.1f, 0.9f}, 37);
        Assert.assertEquals(1, result.getLabel());
        Assert.assertEquals(0.9f, result.getScore(), 0.0f);
        Assert.assertEquals(180, result.getOrientationDegrees());
        Assert.assertEquals(37, result.getResizedWidth());
        Assert.assertTrue(result.requiresRotation(0.89f));
        Assert.assertFalse(result.requiresRotation(0.9f));
    }

    @Test
    public void tiesPreferZeroOrientationAndRejectNonFiniteOutput() {
        ClsClassificationResult result = ClsPostprocess.decode(new float[] {0.5f, 0.5f}, 20);
        Assert.assertEquals(0, result.getLabel());
        Assert.assertEquals(0, result.getOrientationDegrees());
        Assert.assertFalse(result.requiresRotation(-1.0f));
        try {
            ClsPostprocess.decode(new float[] {Float.NaN, 1.0f}, 20);
            Assert.fail("expected invalid CLS output");
        } catch (OcrException e) {
            Assert.assertEquals(OcrErrorCode.INVALID_ARGUMENT, e.getCode());
        }
    }
}
