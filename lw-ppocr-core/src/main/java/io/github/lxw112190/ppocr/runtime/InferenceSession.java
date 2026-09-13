package io.github.lxw112190.ppocr.runtime;

import io.github.lxw112190.ppocr.kernels.KernelBackend;
import io.github.lxw112190.ppocr.kernels.BinaryOp;
import io.github.lxw112190.ppocr.kernels.FusedGeluBackend;
import io.github.lxw112190.ppocr.kernels.ScalarBackend;
import io.github.lxw112190.ppocr.model.LwmModel;
import io.github.lxw112190.ppocr.model.OcrErrorCode;
import io.github.lxw112190.ppocr.model.OcrException;
import io.github.lxw112190.ppocr.model.OperatorType;
import io.github.lxw112190.ppocr.model.NodeInfo;
import java.nio.ByteBuffer;
import java.util.Collections;
import java.util.List;

/** Prepared, reusable scalar execution session for the initial operator subset. */
public final class InferenceSession implements AutoCloseable {
    private final PreparedExecution execution;
    private final Workspace workspace;
    private final KernelBackend backend;
    private final FusedGeluBackend fusedGeluBackend;
    private final PreparedNode[] nodes;
    private final int[][] transposePermutations;
    private final int[][] transposeInputStrides;
    private final int[][] reduceAxes;
    private final int[][] sliceStarts;
    private final int[][] sliceAxes;
    private final int[][] sliceSteps;
    private final ConcatPlan[] concatPlans;
    private final int[][] layoutAxes;
    private final BinaryPlan[] binaryPlans;
    private final InferenceProfiler.NodeDescriptor[] profileDescriptors;
    private final GeluPlan[] geluPlans;
    private final int inputIndex;
    private final int outputIndex;
    private boolean closed;

    public InferenceSession(LwmModel model) {
        this(model, Collections.<TensorShape>emptyList(), new ScalarBackend());
    }

    public InferenceSession(LwmModel model, List<TensorShape> inputShapes) {
        this(model, inputShapes, new ScalarBackend());
    }

    public InferenceSession(LwmModel model, List<TensorShape> inputShapes, KernelBackend backend) {
        this(model, inputShapes, backend, model == null ? null : model.getNodes(),
                model == null || model.getGraphOutputs().isEmpty() ? -1 : model.getGraphOutputs().get(0), false);
    }

    InferenceSession(LwmModel model, List<TensorShape> inputShapes, KernelBackend backend,
                     int nodeLimit, int outputIndex) {
        this(model, inputShapes, backend,
                model == null ? null : model.getNodes().subList(0, nodeLimit), outputIndex, true);
    }

    private InferenceSession(LwmModel model, List<TensorShape> inputShapes, KernelBackend backend,
                             List<NodeInfo> sessionNodes, int outputIndex, boolean partial) {
        if (model == null || inputShapes == null || backend == null) {
            throw new OcrException(OcrErrorCode.INVALID_ARGUMENT, "model, input shapes, and backend are required");
        }
        if (model.getGraphInputs().size() != 1 || outputIndex < 0 ||
                sessionNodes == null || sessionNodes.size() > model.getNodes().size()) {
            throw new OcrException(OcrErrorCode.INVALID_MODEL,
                    "session requires one input and a valid execution output");
        }
        this.inputIndex = model.getGraphInputs().get(0);
        this.outputIndex = outputIndex;
        this.execution = partial
                ? new PreparedExecution(model, inputShapes, sessionNodes, outputIndex)
                : new PreparedExecution(model, inputShapes);
        this.workspace = new Workspace(execution.workspacePlan());
        this.backend = backend;
        this.fusedGeluBackend = backend instanceof FusedGeluBackend
                ? (FusedGeluBackend) backend : null;
        CompiledModel compiledModel = execution.compiledModel();
        this.nodes = new PreparedNode[sessionNodes.size()];
        this.transposePermutations = new int[nodes.length][];
        this.transposeInputStrides = new int[nodes.length][];
        this.reduceAxes = new int[nodes.length][];
        this.sliceStarts = new int[nodes.length][];
        this.sliceAxes = new int[nodes.length][];
        this.sliceSteps = new int[nodes.length][];
        this.concatPlans = new ConcatPlan[nodes.length];
        this.layoutAxes = new int[nodes.length][];
        this.binaryPlans = new BinaryPlan[nodes.length];
        this.profileDescriptors = new InferenceProfiler.NodeDescriptor[nodes.length];
        this.geluPlans = new GeluPlan[nodes.length];
        for (int i = 0; i < nodes.length; i++) {
            NodeInfo node = compiledModel.node(i);
            PreparedNode prepared = new PreparedNode(i, node.getOperator(),
                    compiledModel.nodeInputs(i), compiledModel.nodeOutputs(i),
                    compiledModel.parameterData(i));
            this.nodes[i] = prepared;
            prepareBinary(prepared);
            prepareTranspose(prepared);
            prepareReduceMean(prepared);
            prepareSlice(prepared);
            prepareConcat(prepared);
            prepareLayout(prepared);
        }
        if (fusedGeluBackend != null) prepareGeluPlans();
    }

    public void run(float[] input, float[] output) {
        ensureOpen();
        if (execution.model().getGraphInputs().size() != 1) {
            throw new OcrException(OcrErrorCode.INVALID_MODEL, "session currently requires one input and one output");
        }
        requireLength(input, execution.length(inputIndex), "input");
        requireLength(output, execution.length(outputIndex), "output");
        float[] storage = workspace.fp32();
        System.arraycopy(input, 0, storage, execution.offset(inputIndex), input.length);
        for (int i = 0; i < nodes.length; i++) {
            GeluPlan gelu = geluPlans[i];
            if (gelu == null) {
                executeNode(i, nodes[i], storage);
            } else {
                executeGelu(i, gelu, storage);
                i += 4;
            }
        }
        System.arraycopy(storage, execution.offset(outputIndex), output, 0, output.length);
    }

    public PreparedExecution execution() { ensureOpen(); return execution; }

    @Override
    public void close() { closed = true; }

    private void executeNode(int nodeIndex, PreparedNode node, float[] storage) {
        InferenceProfiler profiler = InferenceProfiler.current();
        if (profiler == null) {
            executeNodeGuarded(node, storage);
            return;
        }
        long start = System.nanoTime();
        try {
            executeNodeGuarded(node, storage);
        } finally {
            long elapsedNanos = System.nanoTime() - start;
            profiler.record(profileDescriptor(nodeIndex, node), elapsedNanos);
        }
    }

    private InferenceProfiler.NodeDescriptor profileDescriptor(int nodeIndex, PreparedNode node) {
        InferenceProfiler.NodeDescriptor descriptor = profileDescriptors[nodeIndex];
        if (descriptor != null) return descriptor;
        int[] inputs = node.inputs;
        int[] outputs = node.outputs;
        StringBuilder description = new StringBuilder("inputs=");
        appendShapes(description, inputs);
        description.append(",outputs=");
        appendShapes(description, outputs);
        if (node.getOperator() == OperatorType.CONV) {
            ByteBuffer params = node.parameters;
            description.append(",groups=").append(params.getInt(4))
                    .append(",kernel=").append(params.getInt(8)).append('x')
                    .append(params.getInt(12)).append(",stride=")
                    .append(params.getInt(16)).append('x').append(params.getInt(20))
                    .append(",dilation=").append(params.getInt(24)).append('x')
                    .append(params.getInt(28)).append(",pads=")
                    .append(params.getInt(32)).append(',').append(params.getInt(36))
                    .append(',').append(params.getInt(40)).append(',')
                    .append(params.getInt(44));
        } else if (node.getOperator() == OperatorType.MAX_POOL
                || node.getOperator() == OperatorType.AVERAGE_POOL) {
            ByteBuffer params = node.parameters;
            description.append(",kernel=").append(params.getInt(8)).append('x')
                    .append(params.getInt(12)).append(",stride=")
                    .append(params.getInt(16)).append('x').append(params.getInt(20))
                    .append(",pads=").append(params.getInt(24)).append(',')
                    .append(params.getInt(28)).append(',').append(params.getInt(32))
                    .append(',').append(params.getInt(36));
        }
        descriptor = new InferenceProfiler.NodeDescriptor(execution.model(), nodeIndex,
                node.getOperator(), description.toString());
        profileDescriptors[nodeIndex] = descriptor;
        return descriptor;
    }

    private void appendShapes(StringBuilder result, int[] tensorIndexes) {
        result.append('[');
        for (int i = 0; i < tensorIndexes.length; i++) {
            if (i != 0) result.append(',');
            result.append(execution.shapes().get(tensorIndexes[i]));
        }
        result.append(']');
    }

    private void executeGelu(int nodeIndex, GeluPlan plan, float[] storage) {
        InferenceProfiler profiler = InferenceProfiler.current();
        if (profiler == null) {
            fusedGeluBackend.gelu(data(plan.input, storage), offset(plan.input),
                    storage, offset(plan.output), execution.length(plan.output),
                    plan.divisor, plan.addend, plan.multiplier);
            return;
        }
        long start = System.nanoTime();
        try {
            fusedGeluBackend.gelu(data(plan.input, storage), offset(plan.input),
                    storage, offset(plan.output), execution.length(plan.output),
                    plan.divisor, plan.addend, plan.multiplier);
        } finally {
            long elapsedNanos = System.nanoTime() - start;
            profiler.record(geluProfileDescriptor(nodeIndex, plan), elapsedNanos);
        }
    }

    private InferenceProfiler.NodeDescriptor geluProfileDescriptor(int nodeIndex, GeluPlan plan) {
        InferenceProfiler.NodeDescriptor descriptor = profileDescriptors[nodeIndex];
        if (descriptor != null) return descriptor;
        String description = "fused_gelu,input=" + execution.shapes().get(plan.input)
                + ",output=" + execution.shapes().get(plan.output)
                + ",divisor=" + plan.divisor + ",addend=" + plan.addend
                + ",multiplier=" + plan.multiplier;
        descriptor = new InferenceProfiler.NodeDescriptor(execution.model(), nodeIndex,
                OperatorType.ERF, description);
        profileDescriptors[nodeIndex] = descriptor;
        return descriptor;
    }

    private void prepareGeluPlans() {
        int[] uses = tensorUseCounts();
        for (int start = 0; start + 4 < nodes.length; start++) {
            PreparedNode divide = nodes[start];
            PreparedNode erf = nodes[start + 1];
            PreparedNode add = nodes[start + 2];
            PreparedNode multiply = nodes[start + 3];
            PreparedNode scale = nodes[start + 4];
            if (divide.getOperator() != OperatorType.DIV ||
                    erf.getOperator() != OperatorType.ERF ||
                    add.getOperator() != OperatorType.ADD ||
                    multiply.getOperator() != OperatorType.MUL ||
                    scale.getOperator() != OperatorType.MUL) continue;
            int[] divideInputs = divide.inputs;
            int[] divideOutputs = divide.outputs;
            int[] erfInputs = erf.inputs;
            int[] erfOutputs = erf.outputs;
            int[] addInputs = add.inputs;
            int[] addOutputs = add.outputs;
            int[] multiplyInputs = multiply.inputs;
            int[] multiplyOutputs = multiply.outputs;
            int[] scaleInputs = scale.inputs;
            int[] scaleOutputs = scale.outputs;
            if (divideInputs.length != 2 || divideOutputs.length != 1 ||
                    erfInputs.length != 1 || erfOutputs.length != 1 ||
                    addInputs.length != 2 || addOutputs.length != 1 ||
                    multiplyInputs.length != 2 || multiplyOutputs.length != 1 ||
                    scaleInputs.length != 2 || scaleOutputs.length != 1) continue;
            int input = divideInputs[0];
            int divideOutput = divideOutputs[0];
            int erfOutput = erfOutputs[0];
            int addOutput = addOutputs[0];
            int multiplyOutput = multiplyOutputs[0];
            int output = scaleOutputs[0];
            if (erfInputs[0] != divideOutput || addInputs[0] != erfOutput ||
                    !samePair(multiplyInputs, input, addOutput) ||
                    scaleInputs[0] != multiplyOutput || uses[divideOutput] != 1 ||
                    uses[erfOutput] != 1 || uses[addOutput] != 1 ||
                    uses[multiplyOutput] != 1 || !sameShape(input, divideOutput, erfOutput,
                            addOutput, multiplyOutput, output)) {
                continue;
            }
            Float divisor = scalarConstant(divideInputs[1]);
            Float addend = scalarConstant(addInputs[1]);
            Float multiplier = scalarConstant(scaleInputs[1]);
            if (divisor == null || addend == null || multiplier == null) continue;
            geluPlans[start] = new GeluPlan(input, output, divisor, addend, multiplier);
            start += 4;
        }
    }

    private int[] tensorUseCounts() {
        int[] uses = new int[execution.model().getTensors().size()];
        for (PreparedNode node : nodes) {
            for (int input : node.inputs) uses[input]++;
        }
        uses[outputIndex]++;
        return uses;
    }

    private Float scalarConstant(int tensorIndex) {
        float[] constant = execution.constant(tensorIndex);
        return constant != null && execution.length(tensorIndex) == 1 ? constant[0] : null;
    }

    private static boolean samePair(int[] values, int first, int second) {
        return values.length == 2 && ((values[0] == first && values[1] == second) ||
                (values[0] == second && values[1] == first));
    }

    private boolean sameShape(int first, int... remaining) {
        TensorShape shape = execution.shapes().get(first);
        for (int tensor : remaining) {
            if (!shape.equals(execution.shapes().get(tensor))) return false;
        }
        return true;
    }

    private void executeNodeGuarded(PreparedNode node, float[] storage) {
        try {
            executeNodeUnchecked(node, storage);
        } catch (OcrException e) {
            throw e;
        } catch (RuntimeException e) {
            throw unsupported(node, "invalid operator parameters or tensor layout", e);
        }
    }

    private void executeNodeUnchecked(PreparedNode node, float[] storage) {
        int[] inputs = node.inputs;
        int[] outputs = node.outputs;
        if (inputs.length == 0) {
            throw unsupported(node, "node has no inputs");
        }
        if (outputs.length != 1) {
            throw unsupported(node, "multiple outputs");
        }
        int output = outputs[0];
        float[] left = data(inputs[0], storage);
        int leftOffset = offset(inputs[0]);
        float[] right;
        int rightOffset;
        switch (node.getOperator()) {
            case ADD: case MUL: case DIV: case SUB:
                executeBinary(node, storage, output, binaryOperation(node.getOperator()));
                break;
            case RELU:
                if (inputs.length != 1 || execution.length(inputs[0]) != execution.length(output)) throw unsupported(node, "shape mismatch");
                backend.relu(left, leftOffset, storage, offset(output), execution.length(output));
                break;
            case SIGMOID:
                if (inputs.length != 1 || execution.length(inputs[0]) != execution.length(output)) throw unsupported(node, "shape mismatch");
                backend.sigmoid(left, leftOffset, storage, offset(output), execution.length(output));
                break;
            case ERF:
                if (inputs.length != 1 || execution.length(inputs[0]) != execution.length(output)) throw unsupported(node, "shape mismatch");
                backend.erf(left, leftOffset, storage, offset(output), execution.length(output));
                break;
            case HARD_SIGMOID:
                if (inputs.length != 1 || execution.length(inputs[0]) != execution.length(output)) throw unsupported(node, "shape mismatch");
                ByteBuffer hardSigmoid = node.parameters;
                backend.hardSigmoid(left, leftOffset, storage, offset(output), execution.length(output),
                        hardSigmoid.getFloat(4), hardSigmoid.getFloat(8));
                break;
            case BATCH_NORMALIZATION:
                executeBatchNormalization(node, storage, output);
                break;
            case SQRT:
                if (inputs.length != 1 || execution.length(inputs[0]) != execution.length(output)) throw unsupported(node, "shape mismatch");
                backend.sqrt(left, leftOffset, storage, offset(output), execution.length(output));
                break;
            case POW:
                executeBinary(node, storage, output, BinaryOp.POW);
                break;
            case REDUCE_MEAN:
                executeReduceMean(node, storage, output);
                break;
            case CONCAT:
                executeConcat(node, storage, output);
                break;
            case SLICE:
                executeSlice(node, storage, output);
                break;
            case CONV:
                executeConv(node, storage, output);
                break;
            case CONV_TRANSPOSE:
                executeConvTranspose(node, storage, output);
                break;
            case AVERAGE_POOL:
            case MAX_POOL:
                executePool(node, storage, output);
                break;
            case RESIZE:
                executeResize(node, storage, output);
                break;
            case TRANSPOSE:
                executeTranspose(node, storage, output);
                break;
            case SQUEEZE:
            case UNSQUEEZE:
                executeSqueezeOrUnsqueeze(node, storage, output);
                break;
            case RESHAPE:
                if (inputs.length != 1 || execution.length(inputs[0]) != execution.length(output)) throw unsupported(node, "reshape element count mismatch");
                System.arraycopy(left, leftOffset, storage, offset(output), execution.length(output));
                break;
            case SOFTMAX:
                executeSoftmax(node, storage, output);
                break;
            case MAT_MUL:
                if (inputs.length != 2 || execution.shapes().get(inputs[1]).getRank() != 2) {
                    throw unsupported(node, "MatMul requires two inputs and a rank-2 right input");
                }
                right = data(inputs[1], storage);
                rightOffset = offset(inputs[1]);
                TensorShape a = execution.shapes().get(inputs[0]);
                TensorShape b = execution.shapes().get(inputs[1]);
                TensorShape c = execution.shapes().get(output);
                if (a.getRank() == 2 && c.getRank() == 2) {
                    if (a.get(1) != b.get(0) || c.get(0) != a.get(0) || c.get(1) != b.get(1)) throw unsupported(node, "MatMul shape mismatch");
                    backend.matMul(left, leftOffset, right, rightOffset, storage, offset(output), a.get(0), a.get(1), b.get(1));
                } else if (a.getRank() == 3 && c.getRank() == 3 && a.get(2) == b.get(0) &&
                        c.get(0) == a.get(0) && c.get(1) == a.get(1) && c.get(2) == b.get(1)) {
                    int batch = a.get(0);
                    int rows = a.get(1);
                    int inner = a.get(2);
                    int columns = b.get(1);
                    int leftBatch = rows * inner;
                    int outputBatch = rows * columns;
                    for (int i = 0; i < batch; i++) {
                        backend.matMul(left, leftOffset + i * leftBatch, right, rightOffset,
                                storage, offset(output) + i * outputBatch, rows, inner, columns);
                    }
                } else {
                    throw unsupported(node, "MatMul shape mismatch");
                }
                break;
            default:
                throw unsupported(node, "operator not implemented by scalar executor");
        }
    }

    private void executeBinary(PreparedNode node, float[] storage, int output, BinaryOp operation) {
        int[] inputs = node.inputs;
        if (inputs.length != 2) throw unsupported(node, "binary operator requires two inputs");
        BinaryPlan plan = binaryPlans[node.index];
        if (plan == null) {
            if (execution.length(inputs[0]) != execution.length(inputs[1]) ||
                    execution.length(inputs[0]) != execution.length(output)) {
                throw unsupported(node, "binary tensor shape mismatch");
            }
            plan = new BinaryPlan(execution.shapes().get(inputs[0]), execution.shapes().get(inputs[1]),
                    execution.shapes().get(output));
        }
        backend.binary(operation, data(inputs[0], storage), offset(inputs[0]),
                data(inputs[1], storage), offset(inputs[1]), storage, offset(output), plan);
    }

    private void prepareBinary(PreparedNode node) {
        OperatorType operator = node.getOperator();
        if (operator != OperatorType.ADD && operator != OperatorType.MUL && operator != OperatorType.DIV &&
                operator != OperatorType.SUB && operator != OperatorType.POW) return;
        int[] inputs = node.inputs;
        int[] outputs = node.outputs;
        if (inputs.length != 2 || outputs.length != 1) return;
        binaryPlans[node.index] = new BinaryPlan(execution.shapes().get(inputs[0]),
                execution.shapes().get(inputs[1]), execution.shapes().get(outputs[0]));
    }

    private static BinaryOp binaryOperation(OperatorType operator) {
        switch (operator) {
            case ADD: return BinaryOp.ADD;
            case MUL: return BinaryOp.MUL;
            case DIV: return BinaryOp.DIV;
            case SUB: return BinaryOp.SUB;
            default: throw new IllegalArgumentException("not a binary operator: " + operator);
        }
    }

    private void executeConv(PreparedNode node, float[] storage, int output) {
        int[] inputs = node.inputs;
        if (inputs.length < 2 || inputs.length > 3 || execution.shapes().get(inputs[0]).getRank() != 4 ||
                execution.shapes().get(inputs[1]).getRank() != 4 || execution.shapes().get(output).getRank() != 4) {
            throw unsupported(node, "Conv requires rank-4 input, weights, and output");
        }
        TensorShape inputShape = execution.shapes().get(inputs[0]);
        TensorShape weightShape = execution.shapes().get(inputs[1]);
        TensorShape outputShape = execution.shapes().get(output);
        ByteBuffer params = node.parameters;
        int groups = params.getInt(4);
        int kernelHeight = params.getInt(8);
        int kernelWidth = params.getInt(12);
        int strideHeight = params.getInt(16);
        int strideWidth = params.getInt(20);
        int dilationHeight = params.getInt(24);
        int dilationWidth = params.getInt(28);
        int padTop = params.getInt(32);
        int padLeft = params.getInt(36);
        int padBottom = params.getInt(40);
        int padRight = params.getInt(44);
        if (groups <= 0 || inputShape.get(1) % groups != 0 || weightShape.get(0) % groups != 0 ||
                weightShape.get(1) != inputShape.get(1) / groups || weightShape.get(2) != kernelHeight ||
                weightShape.get(3) != kernelWidth || outputShape.get(0) != inputShape.get(0) ||
                outputShape.get(1) != weightShape.get(0)) {
            throw unsupported(node, "Conv shape or parameter mismatch");
        }
        float[] weights = data(inputs[1], storage);
        float[] bias = inputs.length == 3 ? data(inputs[2], storage) : null;
        backend.conv(data(inputs[0], storage), offset(inputs[0]), weights, offset(inputs[1]), bias,
                inputs.length == 3 ? offset(inputs[2]) : 0, storage, offset(output),
                inputShape.get(0), inputShape.get(1), inputShape.get(2), inputShape.get(3),
                weightShape.get(0), kernelHeight, kernelWidth, strideHeight, strideWidth,
                dilationHeight, dilationWidth, padTop, padLeft, padBottom, padRight, groups,
                outputShape.get(2), outputShape.get(3));
    }

    private void executeSoftmax(PreparedNode node, float[] storage, int output) {
        int[] inputs = node.inputs;
        if (inputs.length != 1 || execution.length(inputs[0]) != execution.length(output)) throw unsupported(node, "shape mismatch");
        ByteBuffer params = node.parameters;
        int axis = params.getInt(4);
        TensorShape shape = execution.shapes().get(inputs[0]);
        if (axis < 0) axis += shape.getRank();
        if (axis < 0 || axis >= shape.getRank()) throw unsupported(node, "Softmax axis is invalid");
        int outer = 1;
        for (int i = 0; i < axis; i++) outer *= shape.get(i);
        int inner = 1;
        for (int i = axis + 1; i < shape.getRank(); i++) inner *= shape.get(i);
        backend.softmax(data(inputs[0], storage), offset(inputs[0]), storage, offset(output),
                outer, shape.get(axis), inner);
    }

    private void executePool(PreparedNode node, float[] storage, int output) {
        int[] inputs = node.inputs;
        if (inputs.length != 1 || execution.shapes().get(inputs[0]).getRank() != 4 || execution.shapes().get(output).getRank() != 4) {
            throw unsupported(node, "pool requires rank-4 input and output");
        }
        ByteBuffer params = node.parameters;
        TensorShape inputShape = execution.shapes().get(inputs[0]);
        TensorShape outputShape = execution.shapes().get(output);
        backend.pool(data(inputs[0], storage), offset(inputs[0]), storage, offset(output),
                inputShape.get(0), inputShape.get(1), inputShape.get(2), inputShape.get(3),
                params.getInt(8), params.getInt(12), params.getInt(16), params.getInt(20),
                params.getInt(24), params.getInt(28), params.getInt(32), params.getInt(36),
                outputShape.get(2), outputShape.get(3),
                node.getOperator() == OperatorType.MAX_POOL, params.getInt(44) != 0);
    }

    private void executeResize(PreparedNode node, float[] storage, int output) {
        int[] inputs = node.inputs;
        if (inputs.length != 1 || execution.shapes().get(inputs[0]).getRank() != 4 || execution.shapes().get(output).getRank() != 4) {
            throw unsupported(node, "Resize requires rank-4 input and output");
        }
        ByteBuffer params = node.parameters;
        TensorShape inputShape = execution.shapes().get(inputs[0]);
        TensorShape outputShape = execution.shapes().get(output);
        backend.resizeNearest(data(inputs[0], storage), offset(inputs[0]), storage, offset(output),
                inputShape.get(0), inputShape.get(1), inputShape.get(2), inputShape.get(3),
                outputShape.get(2), outputShape.get(3), params.getFloat(12), params.getFloat(16));
    }

    private void executeConvTranspose(PreparedNode node, float[] storage, int output) {
        int[] inputs = node.inputs;
        if (inputs.length < 2 || inputs.length > 3 || execution.shapes().get(inputs[0]).getRank() != 4 ||
                execution.shapes().get(inputs[1]).getRank() != 4 || execution.shapes().get(output).getRank() != 4) {
            throw unsupported(node, "ConvTranspose requires rank-4 input, weights, and output");
        }
        TensorShape inputShape = execution.shapes().get(inputs[0]);
        TensorShape weightShape = execution.shapes().get(inputs[1]);
        TensorShape outputShape = execution.shapes().get(output);
        ByteBuffer params = node.parameters;
        int groups = params.getInt(4);
        int kernelHeight = params.getInt(8);
        int kernelWidth = params.getInt(12);
        int strideHeight = params.getInt(16);
        int strideWidth = params.getInt(20);
        int dilationHeight = params.getInt(24);
        int dilationWidth = params.getInt(28);
        int padTop = params.getInt(32);
        int padLeft = params.getInt(36);
        if (groups <= 0 || inputShape.get(1) % groups != 0 || weightShape.get(0) != inputShape.get(1) ||
                weightShape.get(2) != kernelHeight || weightShape.get(3) != kernelWidth ||
                outputShape.get(0) != inputShape.get(0) || outputShape.get(1) != weightShape.get(1) * groups) {
            throw unsupported(node, "ConvTranspose shape or parameter mismatch");
        }
        float[] bias = inputs.length == 3 ? data(inputs[2], storage) : null;
        backend.convTranspose(data(inputs[0], storage), offset(inputs[0]), data(inputs[1], storage), offset(inputs[1]),
                bias, inputs.length == 3 ? offset(inputs[2]) : 0, storage, offset(output), inputShape.get(0),
                inputShape.get(1), inputShape.get(2), inputShape.get(3), outputShape.get(1), kernelHeight,
                kernelWidth, strideHeight, strideWidth, dilationHeight, dilationWidth, padTop, padLeft, groups,
                outputShape.get(2), outputShape.get(3));
    }

    private void executeReduceMean(PreparedNode node, float[] storage, int output) {
        int[] inputs = node.inputs;
        if (inputs.length != 1) throw unsupported(node, "ReduceMean requires one input");
        ByteBuffer params = node.parameters;
        int[] axes = reduceAxes[node.index];
        if (axes == null) {
            int count = params.getShort(2) & 0xffff;
            axes = new int[count];
            for (int i = 0; i < count; i++) axes[i] = params.getInt(12 + i * 4);
        }
        backend.reduceMean(data(inputs[0], storage), offset(inputs[0]), storage, offset(output),
                execution.shapes().get(inputs[0]).dimensionsUnsafe(), axes, params.getInt(4) != 0);
    }

    private void executeBatchNormalization(PreparedNode node, float[] storage, int output) {
        int[] inputs = node.inputs;
        if (inputs.length != 5 || execution.shapes().get(inputs[0]).getRank() < 2 ||
                execution.length(inputs[0]) != execution.length(output)) {
            throw unsupported(node, "BatchNormalization requires five inputs and matching output");
        }
        TensorShape inputShape = execution.shapes().get(inputs[0]);
        int channels = inputShape.get(1);
        for (int i = 1; i < inputs.length; i++) {
            TensorShape parameterShape = execution.shapes().get(inputs[i]);
            if (parameterShape.getRank() != 1 || parameterShape.get(0) != channels) {
                throw unsupported(node, "BatchNormalization parameter shape mismatch");
            }
        }
        ByteBuffer params = node.parameters;
        float epsilon = params.getFloat(4);
        if (!Float.isFinite(epsilon) || epsilon <= 0.0f) {
            throw unsupported(node, "BatchNormalization epsilon is invalid");
        }
        backend.batchNormalization(data(inputs[0], storage), offset(inputs[0]),
                data(inputs[1], storage), offset(inputs[1]), data(inputs[2], storage), offset(inputs[2]),
                data(inputs[3], storage), offset(inputs[3]), data(inputs[4], storage), offset(inputs[4]),
                epsilon, storage, offset(output), inputShape.dimensionsUnsafe());
    }

    private void executeSqueezeOrUnsqueeze(PreparedNode node, float[] storage, int output) {
        int[] inputs = node.inputs;
        if (inputs.length != 1 || execution.length(inputs[0]) != execution.length(output)) {
            throw unsupported(node, "layout operator shape mismatch");
        }
        TensorShape inputShape = execution.shapes().get(inputs[0]);
        TensorShape outputShape = execution.shapes().get(output);
        int[] axes = layoutAxes[node.index];
        if (axes == null) {
            ByteBuffer params = node.parameters;
            int count = params.getShort(2) & 0xffff;
            axes = new int[count];
            for (int i = 0; i < count; i++) axes[i] = params.getInt(4 + i * 4);
            if (node.getOperator() == OperatorType.SQUEEZE) {
                validateSqueeze(inputShape, outputShape, axes);
            } else {
                validateUnsqueeze(inputShape, outputShape, axes);
            }
        }
        System.arraycopy(data(inputs[0], storage), offset(inputs[0]), storage, offset(output), execution.length(output));
    }

    private void validateSqueeze(TensorShape input, TensorShape output, int[] axes) {
        boolean[] squeezed = new boolean[input.getRank()];
        if (axes.length == 0) {
            for (int axis = 0; axis < input.getRank(); axis++) squeezed[axis] = input.get(axis) == 1;
        } else {
            for (int axisValue : axes) {
                int axis = normalizeAxis(axisValue, input.getRank());
                if (axis < 0 || axis >= input.getRank() || squeezed[axis] || input.get(axis) != 1) {
                    throw new IllegalArgumentException("invalid squeeze axis");
                }
                squeezed[axis] = true;
            }
        }
        int outputAxis = 0;
        for (int axis = 0; axis < input.getRank(); axis++) {
            if (!squeezed[axis] && (outputAxis >= output.getRank() || output.get(outputAxis++) != input.get(axis))) {
                throw new IllegalArgumentException("squeeze output shape mismatch");
            }
        }
        if (outputAxis != output.getRank()) throw new IllegalArgumentException("squeeze output rank mismatch");
    }

    private void validateUnsqueeze(TensorShape input, TensorShape output, int[] axes) {
        if (input.getRank() + axes.length != output.getRank()) throw new IllegalArgumentException("unsqueeze rank mismatch");
        boolean[] inserted = new boolean[output.getRank()];
        for (int axisValue : axes) {
            int axis = normalizeAxis(axisValue, output.getRank());
            if (axis < 0 || axis >= output.getRank() || inserted[axis]) {
                throw new IllegalArgumentException("invalid unsqueeze axis");
            }
            inserted[axis] = true;
        }
        int inputAxis = 0;
        for (int axis = 0; axis < output.getRank(); axis++) {
            int expected = inserted[axis] ? 1 : input.get(inputAxis++);
            if (output.get(axis) != expected) throw new IllegalArgumentException("unsqueeze output shape mismatch");
        }
    }

    private int normalizeAxis(int axis, int rank) {
        return axis < 0 ? axis + rank : axis;
    }

    private void executeConcat(PreparedNode node, float[] storage, int output) {
        int[] inputs = node.inputs;
        if (inputs.length == 0) throw unsupported(node, "Concat requires inputs");
        ConcatPlan plan = concatPlans[node.index];
        TensorShape shape = execution.shapes().get(inputs[0]);
        if (plan != null) {
            for (int i = 0; i < inputs.length; i++) plan.values[i] = data(inputs[i], storage);
            backend.concat(plan.values, plan.offsets, storage, offset(output),
                    shape.dimensionsUnsafe(), plan.axis, plan.axisSizes);
            return;
        }
        ByteBuffer params = node.parameters;
        int axis = params.getInt(4);
        if (axis < 0) axis += shape.getRank();
        if (axis < 0 || axis >= shape.getRank()) throw unsupported(node, "Concat axis is invalid");
        float[][] values = new float[inputs.length][];
        int[] offsets = new int[inputs.length];
        int[] axisSizes = new int[inputs.length];
        for (int i = 0; i < inputs.length; i++) {
            TensorShape inputShape = execution.shapes().get(inputs[i]);
            if (inputShape.getRank() != shape.getRank()) throw unsupported(node, "Concat rank mismatch");
            for (int dimension = 0; dimension < shape.getRank(); dimension++) {
                if (dimension != axis && inputShape.get(dimension) != shape.get(dimension)) throw unsupported(node, "Concat shape mismatch");
            }
            values[i] = data(inputs[i], storage);
            offsets[i] = offset(inputs[i]);
            axisSizes[i] = inputShape.get(axis);
        }
        backend.concat(values, offsets, storage, offset(output), shape.dimensionsUnsafe(), axis, axisSizes);
    }

    private void executeSlice(PreparedNode node, float[] storage, int output) {
        int[] inputs = node.inputs;
        if (inputs.length != 1) throw unsupported(node, "Slice requires one input");
        int[] starts = sliceStarts[node.index];
        int[] axes = sliceAxes[node.index];
        int[] steps = sliceSteps[node.index];
        if (starts == null) {
            ByteBuffer params = node.parameters;
            int count = params.getShort(2) & 0xffff;
            starts = new int[count];
            axes = new int[count];
            steps = new int[count];
            for (int i = 0; i < count; i++) {
                starts[i] = params.getInt(4 + i * 4);
                axes[i] = params.getInt(68 + i * 4);
                steps[i] = params.getInt(100 + i * 4);
            }
        }
        backend.slice(data(inputs[0], storage), offset(inputs[0]), storage, offset(output),
                execution.shapes().get(inputs[0]).dimensionsUnsafe(), starts, axes, steps);
    }

    private void executeTranspose(PreparedNode node, float[] storage, int output) {
        int[] inputs = node.inputs;
        if (inputs.length != 1 || execution.length(inputs[0]) != execution.length(output)) throw unsupported(node, "shape mismatch");
        TensorShape inputShape = execution.shapes().get(inputs[0]);
        TensorShape outputShape = execution.shapes().get(output);
        int[] permutation = transposePermutations[node.index];
        int[] inputStrides = transposeInputStrides[node.index];
        int rank;
        if (permutation == null) {
            ByteBuffer params = node.parameters;
            rank = params.getShort(2) & 0xffff;
            if (rank != inputShape.getRank() || rank != outputShape.getRank()) {
                throw unsupported(node, "Transpose rank mismatch");
            }
            permutation = new int[rank];
            for (int i = 0; i < rank; i++) permutation[i] = params.getInt(4 + i * 4);
            inputStrides = strides(inputShape);
        } else {
            rank = permutation.length;
        }
        float[] input = data(inputs[0], storage);
        for (int linear = 0; linear < execution.length(output); linear++) {
            int remainder = linear;
            int source = 0;
            for (int axis = rank - 1; axis >= 0; axis--) {
                int coordinate = remainder % outputShape.get(axis);
                remainder /= outputShape.get(axis);
                source += coordinate * inputStrides[permutation[axis]];
            }
            storage[offset(output) + linear] = input[offset(inputs[0]) + source];
        }
    }

    private void prepareTranspose(PreparedNode node) {
        if (node.getOperator() != OperatorType.TRANSPOSE) return;
        int[] inputs = node.inputs;
        int[] outputs = node.outputs;
        if (inputs.length != 1 || outputs.length != 1) return;
        TensorShape inputShape = execution.shapes().get(inputs[0]);
        TensorShape outputShape = execution.shapes().get(outputs[0]);
        ByteBuffer params = node.parameters;
        int rank = params.getShort(2) & 0xffff;
        if (rank != inputShape.getRank() || rank != outputShape.getRank()) return;
        int[] permutation = new int[rank];
        for (int i = 0; i < rank; i++) permutation[i] = params.getInt(4 + i * 4);
        transposePermutations[node.index] = permutation;
        transposeInputStrides[node.index] = strides(inputShape);
    }

    private void prepareReduceMean(PreparedNode node) {
        if (node.getOperator() != OperatorType.REDUCE_MEAN) return;
        ByteBuffer params = node.parameters;
        int count = params.getShort(2) & 0xffff;
        if (12L + count * 4L > params.limit()) return;
        int[] axes = new int[count];
        for (int i = 0; i < count; i++) axes[i] = params.getInt(12 + i * 4);
        reduceAxes[node.index] = axes;
    }

    private void prepareSlice(PreparedNode node) {
        if (node.getOperator() != OperatorType.SLICE) return;
        ByteBuffer params = node.parameters;
        int count = params.getShort(2) & 0xffff;
        if (4L + count * 4L > params.limit() || 68L + count * 4L > params.limit() ||
                100L + count * 4L > params.limit()) return;
        int[] starts = new int[count];
        int[] axes = new int[count];
        int[] steps = new int[count];
        for (int i = 0; i < count; i++) {
            starts[i] = params.getInt(4 + i * 4);
            axes[i] = params.getInt(68 + i * 4);
            steps[i] = params.getInt(100 + i * 4);
        }
        sliceStarts[node.index] = starts;
        sliceAxes[node.index] = axes;
        sliceSteps[node.index] = steps;
    }

    private void prepareConcat(PreparedNode node) {
        if (node.getOperator() != OperatorType.CONCAT) return;
        int[] inputs = node.inputs;
        if (inputs.length == 0) return;
        ByteBuffer params = node.parameters;
        int axis = params.getInt(4);
        TensorShape shape = execution.shapes().get(inputs[0]);
        if (axis < 0) axis += shape.getRank();
        if (axis < 0 || axis >= shape.getRank()) return;
        int[] outputs = node.outputs;
        if (outputs.length != 1) return;
        TensorShape outputShape = execution.shapes().get(outputs[0]);
        if (outputShape.getRank() != shape.getRank()) return;
        ConcatPlan plan = new ConcatPlan(inputs.length, axis);
        long axisTotal = 0;
        for (int i = 0; i < inputs.length; i++) {
            TensorShape inputShape = execution.shapes().get(inputs[i]);
            if (inputShape.getRank() != shape.getRank()) return;
            for (int dimension = 0; dimension < shape.getRank(); dimension++) {
                if (dimension != axis && inputShape.get(dimension) != shape.get(dimension)) return;
            }
            plan.offsets[i] = offset(inputs[i]);
            plan.axisSizes[i] = inputShape.get(axis);
            axisTotal += inputShape.get(axis);
        }
        if (axisTotal != outputShape.get(axis)) return;
        for (int dimension = 0; dimension < shape.getRank(); dimension++) {
            if (dimension != axis && outputShape.get(dimension) != shape.get(dimension)) return;
        }
        concatPlans[node.index] = plan;
    }

    private void prepareLayout(PreparedNode node) {
        if (node.getOperator() != OperatorType.SQUEEZE && node.getOperator() != OperatorType.UNSQUEEZE) return;
        int[] inputs = node.inputs;
        int[] outputs = node.outputs;
        if (inputs.length != 1 || outputs.length != 1) return;
        ByteBuffer params = node.parameters;
        int count = params.getShort(2) & 0xffff;
        if (4L + count * 4L > params.limit()) return;
        int[] axes = new int[count];
        for (int i = 0; i < count; i++) axes[i] = params.getInt(4 + i * 4);
        try {
            TensorShape inputShape = execution.shapes().get(inputs[0]);
            TensorShape outputShape = execution.shapes().get(outputs[0]);
            if (node.getOperator() == OperatorType.SQUEEZE) {
                validateSqueeze(inputShape, outputShape, axes);
            } else {
                validateUnsqueeze(inputShape, outputShape, axes);
            }
            layoutAxes[node.index] = axes;
        } catch (RuntimeException ignored) {
            // Preserve the existing run-time validation and error reporting path.
        }
    }

    private static final class ConcatPlan {
        private final int axis;
        private final float[][] values;
        private final int[] offsets;
        private final int[] axisSizes;

        private ConcatPlan(int inputCount, int axis) {
            this.axis = axis;
            this.values = new float[inputCount][];
            this.offsets = new int[inputCount];
            this.axisSizes = new int[inputCount];
        }
    }

    private static final class GeluPlan {
        private final int input;
        private final int output;
        private final float divisor;
        private final float addend;
        private final float multiplier;

        private GeluPlan(int input, int output, float divisor, float addend,
                         float multiplier) {
            this.input = input;
            this.output = output;
            this.divisor = divisor;
            this.addend = addend;
            this.multiplier = multiplier;
        }
    }

    private int[] strides(TensorShape shape) {
        int[] strides = new int[shape.getRank()];
        int stride = 1;
        for (int axis = shape.getRank() - 1; axis >= 0; axis--) {
            strides[axis] = stride;
            stride *= shape.get(axis);
        }
        return strides;
    }

    private float[] data(int tensorIndex, float[] storage) {
        float[] constant = execution.constant(tensorIndex);
        return constant == null ? storage : constant;
    }

    private int offset(int tensorIndex) {
        float[] constant = execution.constant(tensorIndex);
        return constant == null ? execution.offset(tensorIndex) : 0;
    }

    private void requireLength(float[] values, int expected, String name) {
        if (values == null || values.length != expected) throw new OcrException(OcrErrorCode.INVALID_ARGUMENT, name + " length does not match tensor shape");
    }

    private OcrException unsupported(PreparedNode node, String detail) {
        return new OcrException(OcrErrorCode.UNSUPPORTED_OPERATOR, node.getOperator() + ": " + detail);
    }

    private OcrException unsupported(PreparedNode node, String detail, Throwable cause) {
        return new OcrException(OcrErrorCode.UNSUPPORTED_OPERATOR,
                node.getOperator() + ": " + detail, cause);
    }

    private void ensureOpen() {
        if (closed) throw new OcrException(OcrErrorCode.INVALID_ARGUMENT, "inference session is closed");
    }
}
