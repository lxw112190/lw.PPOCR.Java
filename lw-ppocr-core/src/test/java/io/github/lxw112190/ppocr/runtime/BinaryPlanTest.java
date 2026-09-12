package io.github.lxw112190.ppocr.runtime;

import io.github.lxw112190.ppocr.kernels.BinaryOp;
import io.github.lxw112190.ppocr.kernels.ScalarBackend;
import org.junit.Assert;
import org.junit.Test;

public final class BinaryPlanTest {
    private final ScalarBackend backend = new ScalarBackend();

    @Test
    public void mapsChannelBroadcastAcrossSpatialAxes() {
        BinaryPlan plan = new BinaryPlan(
                new TensorShape(1, 2, 2, 3),
                new TensorShape(1, 2, 1, 1),
                new TensorShape(1, 2, 2, 3));

        Assert.assertEquals(BinaryVariant.GENERAL, plan.getVariant());
        Assert.assertEquals(12, plan.getOutputLength());
        Assert.assertArrayEquals(new int[] {0, 6, 3, 1}, plan.getLeftStrides());
        Assert.assertArrayEquals(new int[] {0, 1, 0, 0}, plan.getRightStrides());

        float[] output = new float[12];
        backend.binary(BinaryOp.ADD,
                new float[] {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}, 0,
                new float[] {10, 20}, 0, output, 0, plan);
        Assert.assertArrayEquals(new float[] {
                11, 12, 13, 14, 15, 16, 27, 28, 29, 30, 31, 32
        }, output, 0.0f);
    }

    @Test
    public void recognizesEitherSideScalar() {
        BinaryPlan right = new BinaryPlan(new TensorShape(2, 2), new TensorShape(1), new TensorShape(2, 2));
        Assert.assertEquals(BinaryVariant.RIGHT_SCALAR, right.getVariant());
        BinaryPlan left = new BinaryPlan(new TensorShape(1), new TensorShape(2, 2), new TensorShape(2, 2));
        Assert.assertEquals(BinaryVariant.LEFT_SCALAR, left.getVariant());
    }

    @Test
    public void rejectsIncompatibleOutputShape() {
        try {
            new BinaryPlan(new TensorShape(2, 3), new TensorShape(2, 2), new TensorShape(2, 3));
            Assert.fail("expected incompatible broadcast to fail");
        } catch (IllegalArgumentException expected) {
            Assert.assertTrue(expected.getMessage().contains("broadcast"));
        }
    }
}
