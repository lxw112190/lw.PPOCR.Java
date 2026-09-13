package io.github.lxw112190.ppocr.runtime;

import io.github.lxw112190.ppocr.model.DataType;
import io.github.lxw112190.ppocr.model.LwmModel;
import io.github.lxw112190.ppocr.model.NodeInfo;
import io.github.lxw112190.ppocr.model.TensorInfo;
import java.lang.ref.WeakReference;
import java.nio.ByteBuffer;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.WeakHashMap;

/** Immutable model-wide metadata shared by shape-specialized sessions. */
final class CompiledModel {
    private static final Map<LwmModel, WeakReference<CompiledModel>> CACHE =
            new WeakHashMap<LwmModel, WeakReference<CompiledModel>>();

    private final LwmModel model;
    private final NodeInfo[] nodes;
    private final int[][] nodeInputs;
    private final int[][] nodeOutputs;
    private final ByteBuffer[] parameters;
    private final float[][] constants;
    private final int[] tensorConsumerCounts;
    private final int[] tensorLastUses;

    private CompiledModel(LwmModel model) {
        this.model = model;
        List<NodeInfo> sourceNodes = model.getNodes();
        this.nodes = sourceNodes.toArray(new NodeInfo[sourceNodes.size()]);
        this.nodeInputs = new int[nodes.length][];
        this.nodeOutputs = new int[nodes.length][];
        this.parameters = new ByteBuffer[nodes.length];
        for (int i = 0; i < nodes.length; i++) {
            nodeInputs[i] = nodes[i].getInputs();
            nodeOutputs[i] = nodes[i].getOutputs();
            parameters[i] = model.parameterData(i);
        }
        this.constants = decodeConstants(model);
        this.tensorConsumerCounts = new int[model.getTensors().size()];
        this.tensorLastUses = new int[tensorConsumerCounts.length];
        Arrays.fill(tensorLastUses, -1);
        for (int nodeIndex = 0; nodeIndex < nodes.length; nodeIndex++) {
            for (int input : nodeInputs[nodeIndex]) {
                tensorConsumerCounts[input]++;
                tensorLastUses[input] = nodeIndex;
            }
        }
        for (int output : model.getGraphOutputs()) {
            tensorLastUses[output] = nodes.length;
        }
    }

    static CompiledModel acquire(LwmModel model) {
        if (model == null) throw new IllegalArgumentException("model is required");
        model.getHeader(); // Validate that the source model is still open.
        synchronized (CACHE) {
            WeakReference<CompiledModel> reference = CACHE.get(model);
            CompiledModel existing = reference == null ? null : reference.get();
            if (existing != null) return existing;
            CompiledModel compiled = new CompiledModel(model);
            CACHE.put(model, new WeakReference<CompiledModel>(compiled));
            return compiled;
        }
    }

    LwmModel model() { return model; }
    int nodeCount() { return nodes.length; }
    NodeInfo node(int index) { return nodes[index]; }
    int[] nodeInputs(int index) { return nodeInputs[index]; }
    int[] nodeOutputs(int index) { return nodeOutputs[index]; }
    ByteBuffer parameterData(int index) { return parameters[index]; }
    float[] constant(int tensorIndex) { return constants[tensorIndex]; }
    float[][] constants() { return constants; }
    int tensorConsumerCount(int tensorIndex) { return tensorConsumerCounts[tensorIndex]; }
    int tensorLastUse(int tensorIndex) { return tensorLastUses[tensorIndex]; }

    private static float[][] decodeConstants(LwmModel model) {
        List<TensorInfo> tensors = model.getTensors();
        float[][] decoded = new float[tensors.size()][];
        for (int i = 0; i < tensors.size(); i++) {
            TensorInfo tensor = tensors.get(i);
            if (!tensor.isConstant()) continue;
            if (tensor.getDataType() != DataType.F32) {
                throw new IllegalArgumentException("only F32 execution is supported: tensor " + i);
            }
            int length = elementCount(tensor, i);
            ByteBuffer bytes = model.constantData(i);
            float[] values = new float[length];
            for (int element = 0; element < length; element++) {
                values[element] = bytes.getFloat(element * 4);
            }
            decoded[i] = values;
        }
        return decoded;
    }

    private static int elementCount(TensorInfo tensor, int tensorIndex) {
        long count = 1;
        for (int dimension : tensor.getDimensions()) {
            if (dimension <= 0 || count > Integer.MAX_VALUE / dimension) {
                throw new IllegalArgumentException("constant tensor shape is invalid: " + tensorIndex);
            }
            count *= dimension;
        }
        return (int) count;
    }
}
