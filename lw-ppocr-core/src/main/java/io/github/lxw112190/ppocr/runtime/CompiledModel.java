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
    private final ConstantTensor[] constants;
    private final PreparedMatMulWeights[] preparedMatMulWeights;
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
        this.constants = prepareConstants(model);
        this.preparedMatMulWeights = new PreparedMatMulWeights[constants.length];
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
    float[] constant(int tensorIndex) {
        ConstantTensor constant = constants[tensorIndex];
        return constant == null ? null : constant.canonical();
    }
    ByteBuffer constantRaw(int tensorIndex) {
        ConstantTensor constant = constants[tensorIndex];
        return constant == null ? null : constant.raw();
    }

    PreparedMatMulWeights preparedMatMulWeights(int tensorIndex, int inner, int columns) {
        if (tensorIndex < 0 || tensorIndex >= constants.length || inner <= 0 || columns <= 0) {
            throw new IllegalArgumentException("prepared MatMul weight request is invalid");
        }
        PreparedMatMulWeights value = preparedMatMulWeights[tensorIndex];
        if (value != null) {
            verifyPreparedMatMulShape(value, inner, columns, tensorIndex);
            return value;
        }
        synchronized (this) {
            value = preparedMatMulWeights[tensorIndex];
            if (value == null) {
                ConstantTensor constant = constants[tensorIndex];
                if (constant == null) {
                    throw new IllegalArgumentException(
                            "MatMul weight tensor is not constant: " + tensorIndex);
                }
                value = PreparedMatMulWeights.fromLittleEndian(constant.raw(), inner, columns);
                preparedMatMulWeights[tensorIndex] = value;
            }
            verifyPreparedMatMulShape(value, inner, columns, tensorIndex);
            return value;
        }
    }

    long preparedMatMulWeightBytes() {
        long total = 0L;
        for (PreparedMatMulWeights value : preparedMatMulWeights) {
            if (value != null) total += value.packedBytes();
        }
        return total;
    }

    private static void verifyPreparedMatMulShape(PreparedMatMulWeights value, int inner,
                                                   int columns, int tensorIndex) {
        if (value.getInner() != inner || value.getColumns() != columns) {
            throw new IllegalStateException(
                    "prepared MatMul weight shape changed for tensor " + tensorIndex);
        }
    }
    boolean isConstantMaterialized(int tensorIndex) {
        ConstantTensor constant = constants[tensorIndex];
        return constant != null && constant.isMaterialized();
    }

    long decodedConstantBytes() {
        long total = 0L;
        for (ConstantTensor constant : constants) {
            if (constant != null) total += constant.decodedBytes();
        }
        return total;
    }

    int tensorConsumerCount(int tensorIndex) { return tensorConsumerCounts[tensorIndex]; }
    int tensorLastUse(int tensorIndex) { return tensorLastUses[tensorIndex]; }

    private static ConstantTensor[] prepareConstants(LwmModel model) {
        List<TensorInfo> tensors = model.getTensors();
        ConstantTensor[] prepared = new ConstantTensor[tensors.size()];
        for (int i = 0; i < tensors.size(); i++) {
            TensorInfo tensor = tensors.get(i);
            if (!tensor.isConstant()) continue;
            if (tensor.getDataType() != DataType.F32) {
                throw new IllegalArgumentException("only F32 execution is supported: tensor " + i);
            }
            int length = elementCount(tensor, i);
            prepared[i] = new ConstantTensor(model.constantData(i), length);
        }
        return prepared;
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
