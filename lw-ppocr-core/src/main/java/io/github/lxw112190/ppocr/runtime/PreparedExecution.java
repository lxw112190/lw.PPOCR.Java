package io.github.lxw112190.ppocr.runtime;

import io.github.lxw112190.ppocr.model.DataType;
import io.github.lxw112190.ppocr.model.LwmModel;
import io.github.lxw112190.ppocr.model.TensorInfo;
import java.nio.ByteBuffer;
import java.util.List;

/** Immutable graph metadata prepared once before a session enters its run loop. */
public final class PreparedExecution {
    private final LwmModel model;
    private final List<TensorShape> shapes;
    private final WorkspacePlan workspacePlan;
    private final float[][] constants;
    private final int[] offsets;
    private final int[] lengths;

    public PreparedExecution(LwmModel model, List<TensorShape> inputShapes) {
        this.model = model;
        this.shapes = ShapeResolver.resolve(model, inputShapes);
        this.workspacePlan = MemoryPlanner.plan(model.getTensors(), model.getNodes(),
                model.getGraphInputs(), model.getGraphOutputs(), shapes);
        this.constants = new float[model.getTensors().size()][];
        this.offsets = new int[model.getTensors().size()];
        this.lengths = new int[model.getTensors().size()];
        for (int i = 0; i < model.getTensors().size(); i++) {
            TensorInfo tensor = model.getTensors().get(i);
            long elements = shapes.get(i).getElementCount();
            if (elements > Integer.MAX_VALUE) {
                throw new IllegalArgumentException("tensor exceeds Java array capacity: " + i);
            }
            lengths[i] = (int) elements;
            if (tensor.isConstant()) {
                if (tensor.getDataType() != DataType.F32) {
                    throw new IllegalArgumentException("only F32 execution is supported: tensor " + i);
                }
                constants[i] = readF32(model.constantData(i), lengths[i]);
            } else {
                long byteOffset = workspacePlan.getOffset(i);
                if (byteOffset < 0 || byteOffset % 4 != 0 || byteOffset / 4 > Integer.MAX_VALUE) {
                    throw new IllegalArgumentException("tensor has no usable workspace allocation: " + i);
                }
                offsets[i] = (int) (byteOffset / 4);
            }
        }
    }

    public LwmModel model() { return model; }
    public List<TensorShape> shapes() { return shapes; }
    public WorkspacePlan workspacePlan() { return workspacePlan; }
    public float[] constant(int tensorIndex) { return constants[tensorIndex]; }
    public int offset(int tensorIndex) { return offsets[tensorIndex]; }
    public int length(int tensorIndex) { return lengths[tensorIndex]; }

    private static float[] readF32(ByteBuffer bytes, int length) {
        float[] values = new float[length];
        for (int i = 0; i < length; i++) {
            values[i] = bytes.getFloat(i * 4);
        }
        return values;
    }
}
