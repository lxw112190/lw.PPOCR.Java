package io.github.lxw112190.ppocr.runtime;

import io.github.lxw112190.ppocr.model.OcrException;
import io.github.lxw112190.ppocr.model.OperatorType;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import org.junit.Assert;
import org.junit.Test;

public final class OnnxRuntimePlansTest {
    @Test public void matrixBatchOffsetsAreBroadcastNotFlattened() {
        MatMulPlan p = new MatMulPlan(new TensorShape(2,1,3,4),new TensorShape(1,5,4,6));
        Assert.assertEquals(10,p.batches);
        Assert.assertArrayEquals(new int[] {0,0,0,0,0,12,12,12,12,12},p.leftOffsets);
        Assert.assertArrayEquals(new int[] {0,24,48,72,96,0,24,48,72,96},p.rightOffsets);
        Assert.assertEquals(new TensorShape(2,5,3,6),MatMulPlan.outputShape(new TensorShape(2,1,3,4),new TensorShape(1,5,4,6)));
    }
    @Test public void sliceHonorsEndsNegativeBoundsStepsAndSentinels() {
        ByteBuffer p = ByteBuffer.allocate(136).order(ByteOrder.LITTLE_ENDIAN);
        p.putShort(2,(short)1); p.putInt(4,-5);p.putInt(36,-1);p.putInt(68,-1);p.putInt(100,2);
        SlicePlan plan = new SlicePlan(new TensorShape(2,6),new TensorShape(2,2),p);
        float[] input={0,1,2,3,4,5,6,7,8,9,10,11},output={-99,-99,-99,-99,-99,-99};
        plan.run(input,0,output,1);Assert.assertArrayEquals(new float[] {-99,1,3,7,9,-99},output,0);
    }
    @Test public void unsqueezeUsesOutputAxesSimultaneously() {
        ByteBuffer p=ByteBuffer.allocate(40).order(ByteOrder.LITTLE_ENDIAN);
        p.putShort(2,(short)2);p.putInt(4,-1);p.putInt(8,0);
        Assert.assertEquals(new TensorShape(1,2,3,1),ShapeResolver.resolveNode(OperatorType.UNSQUEEZE,new TensorShape[] {new TensorShape(2,3)},new int[0],p));
    }
    @Test public void rejectsIncompatibleMatrixBatches() {
        try { MatMulPlan.outputShape(new TensorShape(2,3,4),new TensorShape(5,4,6));Assert.fail(); }
        catch(OcrException expected) { Assert.assertTrue(expected.getMessage().contains("broadcast")); }
    }
}
