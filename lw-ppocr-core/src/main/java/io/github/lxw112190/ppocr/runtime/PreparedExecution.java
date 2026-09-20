package io.github.lxw112190.ppocr.runtime;

import io.github.lxw112190.ppocr.model.LwmModel;
import io.github.lxw112190.ppocr.model.NodeInfo;
import io.github.lxw112190.ppocr.model.OperatorType;
import io.github.lxw112190.ppocr.model.TensorInfo;
import java.nio.ByteBuffer;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

/** Immutable graph metadata prepared once before a session enters its run loop. */
public final class PreparedExecution {
    private final LwmModel model;
    private final CompiledModel compiledModel;
    private final List<TensorShape> shapes;
    private final WorkspacePlan workspacePlan;
    private final int[] offsets;
    private final int[] lengths;

    public PreparedExecution(LwmModel model, List<TensorShape> inputShapes) {
        this(model, inputShapes, model.getNodes(), model.getGraphOutputs(), false, false, false);
    }

    PreparedExecution(LwmModel model, List<TensorShape> inputShapes, boolean fuseGelu) {
        this(model, inputShapes, model.getNodes(), model.getGraphOutputs(), false, fuseGelu, false);
    }

    PreparedExecution(LwmModel model, List<TensorShape> inputShapes, boolean fuseGelu,
                      boolean aliasElementwise) {
        this(model, inputShapes, model.getNodes(), model.getGraphOutputs(), false,
                fuseGelu, aliasElementwise);
    }

    PreparedExecution(LwmModel model, List<TensorShape> inputShapes,
                      List<io.github.lxw112190.ppocr.model.NodeInfo> nodes,
                      int outputIndex) {
        this(model, inputShapes, nodes, Collections.singletonList(outputIndex), true, false, false);
    }

    PreparedExecution(LwmModel model, List<TensorShape> inputShapes,
                      List<io.github.lxw112190.ppocr.model.NodeInfo> nodes,
                      int outputIndex, boolean fuseGelu) {
        this(model, inputShapes, nodes, Collections.singletonList(outputIndex), true, fuseGelu, false);
    }

    PreparedExecution(LwmModel model, List<TensorShape> inputShapes,
                      List<io.github.lxw112190.ppocr.model.NodeInfo> nodes,
                      int outputIndex, boolean fuseGelu, boolean aliasElementwise) {
        this(model, inputShapes, nodes, Collections.singletonList(outputIndex), true,
                fuseGelu, aliasElementwise);
    }

    private PreparedExecution(LwmModel model, List<TensorShape> inputShapes,
                              List<NodeInfo> nodes,
                              List<Integer> outputs, boolean partial, boolean fuseGelu,
                              boolean aliasElementwise) {
        this.compiledModel = CompiledModel.acquire(model);
        this.model = compiledModel.model();
        this.shapes = ShapeResolver.resolve(model, inputShapes);
        int[] concatAxes = concatAxes(nodes, compiledModel);
        this.workspacePlan = partial
                ? MemoryPlanner.planPartial(model.getTensors(), nodes, model.getGraphInputs(),
                        outputs, shapes, fuseGelu, aliasElementwise, concatAxes)
                : MemoryPlanner.plan(model.getTensors(), nodes, model.getGraphInputs(),
                        outputs, shapes, fuseGelu, aliasElementwise, concatAxes);
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
    }

    public LwmModel model() { return model; }
    public List<TensorShape> shapes() { return shapes; }
    public WorkspacePlan workspacePlan() { return workspacePlan; }
    public float[] constant(int tensorIndex) { return compiledModel.constant(tensorIndex); }
    ByteBuffer constantRaw(int tensorIndex) { return compiledModel.constantRaw(tensorIndex); }
    PreparedMatMulWeights preparedMatMulWeights(int tensorIndex, int inner, int columns) {
        return compiledModel.preparedMatMulWeights(tensorIndex, inner, columns);
    }
    long preparedMatMulWeightBytes() { return compiledModel.preparedMatMulWeightBytes(); }
    /** Returns the currently materialized canonical constant storage in bytes. */
    public long decodedConstantBytes() { return compiledModel.decodedConstantBytes(); }
    public int offset(int tensorIndex) { return offsets[tensorIndex]; }
    public int length(int tensorIndex) { return lengths[tensorIndex]; }

    CompiledModel compiledModel() { return compiledModel; }

    private static int[] concatAxes(List<NodeInfo> nodes, CompiledModel compiledModel) {
        int[] axes = new int[nodes.size()];
        Arrays.fill(axes, Integer.MIN_VALUE);
        for (int i = 0; i < nodes.size(); i++) {
            if (nodes.get(i).getOperator() != OperatorType.CONCAT) continue;
            ByteBuffer parameters = compiledModel.parameterData(i);
            if (parameters.limit() >= 8) axes[i] = parameters.getInt(4);
        }
        return axes;
    }

}
