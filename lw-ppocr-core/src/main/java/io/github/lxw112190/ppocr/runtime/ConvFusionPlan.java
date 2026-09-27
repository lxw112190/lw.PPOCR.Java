package io.github.lxw112190.ppocr.runtime;

import io.github.lxw112190.ppocr.kernels.ConvEpilogue;
import io.github.lxw112190.ppocr.model.NodeInfo;
import io.github.lxw112190.ppocr.model.DataType;
import io.github.lxw112190.ppocr.model.OperatorType;
import java.nio.ByteBuffer;
import java.util.List;

/** Shared semantic-to-physical compiler used by both allocator and executor. */
final class ConvFusionPlan {
    final int first, last, output, residual;
    final ConvEpilogue epilogue;

    private ConvFusionPlan(int first, int last, int output, int residual, ConvEpilogue epilogue) {
        this.first = first; this.last = last; this.output = output;
        this.residual = residual; this.epilogue = epilogue;
    }

    static ConvFusionPlan[] compile(CompiledModel model, List<NodeInfo> nodes,
                                    List<TensorShape> shapes, List<Integer> outputs) {
        ConvFusionPlan[] result = new ConvFusionPlan[nodes.size()];
        int[] uses = new int[shapes.size()];
        for (int n = 0; n < nodes.size(); n++) for (int input : model.nodeInputs(n)) uses[input]++;
        for (int output : outputs) uses[output]++;
        for (int start = 0; start < nodes.size(); start++) {
            if (nodes.get(start).getOperator() != OperatorType.CONV
                    || model.nodeOutputs(start).length != 1) continue;
            int current = model.nodeOutputs(start)[0];
            TensorShape shape = shapes.get(current);
            if (shape.getRank() != 4) continue;
            int channels = shape.get(1), cursor = start + 1, residual = -1;
            float[] mean = null, factor = null, normBias = null, postBias = null;
            boolean scalarBias = false;
            int activation = ConvEpilogue.NONE;
            float divisor = 1, addend = 0, multiplier = 1, alpha = 0, beta = 0;
            if (cursor < nodes.size() && uses[current] == 1
                    && nodes.get(cursor).getOperator() == OperatorType.BATCH_NORMALIZATION) {
                int[] in = model.nodeInputs(cursor), out = model.nodeOutputs(cursor);
                if (in.length == 5 && out.length == 1 && in[0] == current
                        && shape.equals(shapes.get(out[0]))
                        && constantLength(model, shapes, in[1], channels)
                        && constantLength(model, shapes, in[2], channels)
                        && constantLength(model, shapes, in[3], channels)
                        && constantLength(model, shapes, in[4], channels)) {
                    float[] scale = model.constant(in[1]), variance = model.constant(in[4]);
                    mean = model.constant(in[3]); normBias = model.constant(in[2]);
                    factor = new float[channels];
                    float epsilon = model.parameterData(cursor).getFloat(4);
                    for (int c = 0; c < channels; c++) {
                        factor[c] = scale[c] / (float) Math.sqrt(variance[c] + epsilon);
                    }
                    current = out[0]; cursor++;
                }
            }
            if (cursor < nodes.size() && uses[current] == 1
                    && nodes.get(cursor).getOperator() == OperatorType.ADD) {
                int[] in = model.nodeInputs(cursor), out = model.nodeOutputs(cursor);
                if (in.length == 2 && out.length == 1 && shape.equals(shapes.get(out[0]))) {
                    int other = in[0] == current ? in[1] : in[1] == current ? in[0] : -1;
                    if (other >= 0) {
                        if (!model.model().getTensors().get(other).isConstant() && shape.equals(shapes.get(other))) {
                            residual = other; current = out[0]; cursor++;
                        } else if (model.model().getTensors().get(other).isConstant()) {
                            TensorShape biasShape = shapes.get(other);
                            boolean scalar = biasShape.getElementCount() == 1;
                            boolean channelBias = (biasShape.getRank() == 4 && biasShape.get(0) == 1
                                    && biasShape.get(1) == channels && biasShape.get(2) == 1 && biasShape.get(3) == 1)
                                    || (biasShape.getRank() == 3 && biasShape.get(0) == channels
                                    && biasShape.get(1) == 1 && biasShape.get(2) == 1);
                            if (scalar || channelBias) {
                                postBias = model.constant(other); scalarBias = scalar;
                                current = out[0]; cursor++;
                            }
                        }
                    }
                }
            }
            if (cursor < nodes.size() && uses[current] == 1
                    && nodes.get(cursor).getOperator() == OperatorType.RELU
                    && unary(model, shapes, cursor, current, shape)) {
                activation = ConvEpilogue.RELU; current = model.nodeOutputs(cursor)[0]; cursor++;
            } else if (cursor + 1 < nodes.size() && uses[current] == 2
                    && nodes.get(cursor).getOperator() == OperatorType.HARD_SIGMOID
                    && nodes.get(cursor + 1).getOperator() == OperatorType.MUL
                    && unary(model, shapes, cursor, current, shape)) {
                int gate = model.nodeOutputs(cursor)[0];
                int[] out = model.nodeOutputs(cursor + 1);
                if (uses[gate] == 1 && pair(model.nodeInputs(cursor + 1), current, gate)
                        && out.length == 1 && shape.equals(shapes.get(out[0]))) {
                    ByteBuffer params = model.parameterData(cursor);
                    alpha = params.getFloat(4); beta = params.getFloat(8);
                    activation = ConvEpilogue.HARD_SWISH; current = out[0]; cursor += 2;
                }
            } else if (cursor < nodes.size() && uses[current] == 1
                    && nodes.get(cursor).getOperator() == OperatorType.HARD_SIGMOID
                    && unary(model, shapes, cursor, current, shape)) {
                ByteBuffer params = model.parameterData(cursor);
                alpha = params.getFloat(4); beta = params.getFloat(8);
                activation = ConvEpilogue.HARD_SIGMOID; current = model.nodeOutputs(cursor)[0]; cursor++;
            } else if (uses[current] == 2 && gelu(model, nodes, shapes, uses, cursor, current, shape)) {
                divisor = model.constant(model.nodeInputs(cursor)[1])[0];
                addend = model.constant(model.nodeInputs(cursor + 2)[1])[0];
                multiplier = model.constant(model.nodeInputs(cursor + 4)[1])[0];
                activation = ConvEpilogue.GELU; current = model.nodeOutputs(cursor + 4)[0]; cursor += 5;
            }
            if (cursor > start + 1) {
                result[start] = new ConvFusionPlan(start, cursor - 1, current, residual,
                        new ConvEpilogue(mean, factor, normBias, postBias, scalarBias, activation,
                                divisor, addend, multiplier, alpha, beta));
                start = cursor - 1;
            }
        }
        return result;
    }

    private static boolean gelu(CompiledModel model, List<NodeInfo> nodes, List<TensorShape> shapes,
                                int[] uses, int start, int input, TensorShape shape) {
        OperatorType[] ops = {OperatorType.DIV, OperatorType.ERF, OperatorType.ADD, OperatorType.MUL, OperatorType.MUL};
        if (start + 4 >= nodes.size()) return false;
        for (int i = 0; i < 5; i++) {
            int[] in = model.nodeInputs(start + i), out = model.nodeOutputs(start + i);
            if (nodes.get(start + i).getOperator() != ops[i] || out.length != 1
                    || !shape.equals(shapes.get(out[0])) || in.length != (i == 1 ? 1 : 2)) return false;
            if (i < 4 && uses[out[0]] != 1) return false;
        }
        return model.nodeInputs(start)[0] == input
                && constantLength(model, shapes, model.nodeInputs(start)[1], 1)
                && model.nodeInputs(start + 1)[0] == model.nodeOutputs(start)[0]
                && model.nodeInputs(start + 2)[0] == model.nodeOutputs(start + 1)[0]
                && constantLength(model, shapes, model.nodeInputs(start + 2)[1], 1)
                && pair(model.nodeInputs(start + 3), input, model.nodeOutputs(start + 2)[0])
                && model.nodeInputs(start + 4)[0] == model.nodeOutputs(start + 3)[0]
                && constantLength(model, shapes, model.nodeInputs(start + 4)[1], 1);
    }

    private static boolean constantLength(CompiledModel model, List<TensorShape> shapes, int tensor, int length) {
        return model.model().getTensors().get(tensor).isConstant()
                && model.model().getTensors().get(tensor).getDataType() == DataType.F32
                && shapes.get(tensor).getElementCount() == length;
    }
    private static boolean unary(CompiledModel model, List<TensorShape> shapes, int node, int input, TensorShape shape) {
        int[] in = model.nodeInputs(node), out = model.nodeOutputs(node);
        return in.length == 1 && out.length == 1 && in[0] == input && shape.equals(shapes.get(out[0]));
    }
    private static boolean pair(int[] inputs, int a, int b) {
        return inputs.length == 2 && ((inputs[0] == a && inputs[1] == b) || (inputs[0] == b && inputs[1] == a));
    }
}
