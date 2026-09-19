package io.github.lxw112190.ppocr.runtime;

import io.github.lxw112190.ppocr.model.LwmModel;
import io.github.lxw112190.ppocr.model.TensorInfo;
import java.util.Collections;
import java.util.List;

/** Immutable graph metadata prepared once before a session enters its run loop. */
public final class PreparedExecution {
    private final LwmModel model;
    private final CompiledModel compiledModel;
    private final List<TensorShape> shapes;
    private final WorkspacePlan workspacePlan;
    private final float[][] constants;
    private final int[] offsets;
    private final int[] lengths;

    public PreparedExecution(LwmModel model, List<TensorShape> inputShapes) {
        this(model, inputShapes, model.getNodes(), model.getGraphOutputs(), false, false);
    }

    PreparedExecution(LwmModel model, List<TensorShape> inputShapes, boolean fuseGelu) {
        this(model, inputShapes, model.getNodes(), model.getGraphOutputs(), false, fuseGelu);
    }

    PreparedExecution(LwmModel model, List<TensorShape> inputShapes,
                      List<io.github.lxw112190.ppocr.model.NodeInfo> nodes,
                      int outputIndex) {
        this(model, inputShapes, nodes, Collections.singletonList(outputIndex), true, false);
    }

    PreparedExecution(LwmModel model, List<TensorShape> inputShapes,
                      List<io.github.lxw112190.ppocr.model.NodeInfo> nodes,
                      int outputIndex, boolean fuseGelu) {
        this(model, inputShapes, nodes, Collections.singletonList(outputIndex), true, fuseGelu);
    }

    private PreparedExecution(LwmModel model, List<TensorShape> inputShapes,
                              List<io.github.lxw112190.ppocr.model.NodeInfo> nodes,
                              List<Integer> outputs, boolean partial, boolean fuseGelu) {
        this.compiledModel = CompiledModel.acquire(model);
        this.model = compiledModel.model();
        this.shapes = ShapeResolver.resolve(model, inputShapes);
        this.workspacePlan = partial
                ? MemoryPlanner.planPartial(model.getTensors(), nodes, model.getGraphInputs(),
                        outputs, shapes, fuseGelu)
                : MemoryPlanner.plan(model.getTensors(), nodes, model.getGraphInputs(),
                        outputs, shapes, fuseGelu);
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
                continue;
            } else if (workspacePlan.isAllocated(i)) {
                long byteOffset = workspacePlan.getOffset(i);
                if (byteOffset < 0 || byteOffset % 4 != 0 || byteOffset / 4 > Integer.MAX_VALUE) {
                    throw new IllegalArgumentException("tensor has no usable workspace allocation: " + i);
                }
                offsets[i] = (int) (byteOffset / 4);
            } else {
                offsets[i] = -1;
            }
        }
        this.constants = compiledModel.constants();
    }

    public LwmModel model() { return model; }
    public List<TensorShape> shapes() { return shapes; }
    public WorkspacePlan workspacePlan() { return workspacePlan; }
    public float[] constant(int tensorIndex) { return constants[tensorIndex]; }
    public int offset(int tensorIndex) { return offsets[tensorIndex]; }
    public int length(int tensorIndex) { return lengths[tensorIndex]; }

    CompiledModel compiledModel() { return compiledModel; }

}
