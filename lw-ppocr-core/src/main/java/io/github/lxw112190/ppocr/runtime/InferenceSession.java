package io.github.lxw112190.ppocr.runtime;

import io.github.lxw112190.ppocr.kernels.KernelBackend;
import io.github.lxw112190.ppocr.kernels.ScalarBackend;
import io.github.lxw112190.ppocr.model.LwmModel;
import io.github.lxw112190.ppocr.model.OcrErrorCode;
import io.github.lxw112190.ppocr.model.OcrException;
import io.github.lxw112190.ppocr.model.OperatorType;
import io.github.lxw112190.ppocr.model.NodeInfo;
import java.nio.ByteBuffer;
import java.util.Collections;
import java.util.IdentityHashMap;
import java.util.List;

/** Prepared, reusable scalar execution session for the initial operator subset. */
public final class InferenceSession implements AutoCloseable {
    private final PreparedExecution execution;
    private final Workspace workspace;
    private final KernelBackend backend;
    private final ByteBuffer[] parameters;
    private final IdentityHashMap<NodeInfo, Integer> nodeIndexes;
    private final IdentityHashMap<NodeInfo, int[]> nodeInputs;
    private final IdentityHashMap<NodeInfo, int[]> nodeOutputs;
    private final IdentityHashMap<NodeInfo, int[]> transposePermutations;
    private final IdentityHashMap<NodeInfo, int[]> transposeInputStrides;
    private final IdentityHashMap<NodeInfo, int[]> reduceAxes;
    private final IdentityHashMap<NodeInfo, int[]> sliceStarts;
    private final IdentityHashMap<NodeInfo, int[]> sliceAxes;
    private final IdentityHashMap<NodeInfo, int[]> sliceSteps;
    private final IdentityHashMap<NodeInfo, ConcatPlan> concatPlans;
    private boolean closed;

    public InferenceSession(LwmModel model) {
        this(model, Collections.<TensorShape>emptyList(), new ScalarBackend());
    }

    public InferenceSession(LwmModel model, List<TensorShape> inputShapes) {
        this(model, inputShapes, new ScalarBackend());
    }

    public InferenceSession(LwmModel model, List<TensorShape> inputShapes, KernelBackend backend) {
        if (model == null || inputShapes == null || backend == null) {
            throw new OcrException(OcrErrorCode.INVALID_ARGUMENT, "model, input shapes, and backend are required");
        }
        this.execution = new PreparedExecution(model, inputShapes);
        this.workspace = new Workspace(execution.workspacePlan());
        this.backend = backend;
        List<NodeInfo> nodes = model.getNodes();
        this.parameters = new ByteBuffer[nodes.size()];
        this.nodeIndexes = new IdentityHashMap<NodeInfo, Integer>(nodes.size());
        this.nodeInputs = new IdentityHashMap<NodeInfo, int[]>(nodes.size());
        this.nodeOutputs = new IdentityHashMap<NodeInfo, int[]>(nodes.size());
        this.transposePermutations = new IdentityHashMap<NodeInfo, int[]>(nodes.size());
        this.transposeInputStrides = new IdentityHashMap<NodeInfo, int[]>(nodes.size());
        this.reduceAxes = new IdentityHashMap<NodeInfo, int[]>(nodes.size());
        this.sliceStarts = new IdentityHashMap<NodeInfo, int[]>(nodes.size());
        this.sliceAxes = new IdentityHashMap<NodeInfo, int[]>(nodes.size());
        this.sliceSteps = new IdentityHashMap<NodeInfo, int[]>(nodes.size());
        this.concatPlans = new IdentityHashMap<NodeInfo, ConcatPlan>(nodes.size());
        for (int i = 0; i < nodes.size(); i++) {
            NodeInfo node = nodes.get(i);
            this.nodeIndexes.put(node, i);
            this.nodeInputs.put(node, node.getInputs());
            this.nodeOutputs.put(node, node.getOutputs());
            this.parameters[i] = model.parameterData(i);
            prepareTranspose(node);
            prepareReduceMean(node);
            prepareSlice(node);
            prepareConcat(node);
        }
    }

    public void run(float[] input, float[] output) {
        ensureOpen();
        if (execution.model().getGraphInputs().size() != 1 || execution.model().getGraphOutputs().size() != 1) {
            throw new OcrException(OcrErrorCode.INVALID_MODEL, "session currently requires one input and one output");
        }
        int inputIndex = execution.model().getGraphInputs().get(0);
        int outputIndex = execution.model().getGraphOutputs().get(0);
        requireLength(input, execution.length(inputIndex), "input");
        requireLength(output, execution.length(outputIndex), "output");
        float[] storage = workspace.fp32();
        System.arraycopy(input, 0, storage, execution.offset(inputIndex), input.length);
        for (NodeInfo node : execution.model().getNodes()) {
            executeNode(node, storage);
        }
        System.arraycopy(storage, execution.offset(outputIndex), output, 0, output.length);
    }

    public PreparedExecution execution() { ensureOpen(); return execution; }

    @Override
    public void close() { closed = true; }

    private void executeNode(NodeInfo node, float[] storage) {
        try {
            executeNodeUnchecked(node, storage);
        } catch (OcrException e) {
            throw e;
        } catch (RuntimeException e) {
            throw unsupported(node, "invalid operator parameters or tensor layout", e);
        }
    }

    private void executeNodeUnchecked(NodeInfo node, float[] storage) {
        int[] inputs = nodeInputs.get(node);
        int[] outputs = nodeOutputs.get(node);
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
                if (inputs.length != 2 || execution.length(inputs[0]) != execution.length(inputs[1]) ||
                        execution.length(inputs[0]) != execution.length(output)) {
                    throw unsupported(node, "only equal-length binary tensors are supported");
                }
                right = data(inputs[1], storage);
                rightOffset = offset(inputs[1]);
                int length = execution.length(output);
                if (node.getOperator() == OperatorType.ADD) backend.add(left, leftOffset, right, rightOffset, storage, offset(output), length);
                else if (node.getOperator() == OperatorType.MUL) backend.mul(left, leftOffset, right, rightOffset, storage, offset(output), length);
                else if (node.getOperator() == OperatorType.DIV) backend.div(left, leftOffset, right, rightOffset, storage, offset(output), length);
                else backend.sub(left, leftOffset, right, rightOffset, storage, offset(output), length);
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
                ByteBuffer hardSigmoid = parameterData(node);
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
                if (inputs.length != 2 || execution.length(inputs[0]) != execution.length(inputs[1]) ||
                        execution.length(inputs[0]) != execution.length(output)) throw unsupported(node, "only equal-length Pow tensors are supported");
                right = data(inputs[1], storage);
                rightOffset = offset(inputs[1]);
                backend.pow(left, leftOffset, right, rightOffset, storage, offset(output), execution.length(output));
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
                if (inputs.length != 2 || execution.shapes().get(inputs[0]).getRank() != 2 ||
                        execution.shapes().get(inputs[1]).getRank() != 2 || execution.shapes().get(output).getRank() != 2) {
                    throw unsupported(node, "MatMul requires three rank-2 tensors");
                }
                right = data(inputs[1], storage);
                rightOffset = offset(inputs[1]);
                TensorShape a = execution.shapes().get(inputs[0]);
                TensorShape b = execution.shapes().get(inputs[1]);
                TensorShape c = execution.shapes().get(output);
                if (a.get(1) != b.get(0) || c.get(0) != a.get(0) || c.get(1) != b.get(1)) throw unsupported(node, "MatMul shape mismatch");
                backend.matMul(left, leftOffset, right, rightOffset, storage, offset(output), a.get(0), a.get(1), b.get(1));
                break;
            default:
                throw unsupported(node, "operator not implemented by scalar executor");
        }
    }

    private void executeConv(NodeInfo node, float[] storage, int output) {
        int[] inputs = nodeInputs.get(node);
        if (inputs.length < 2 || inputs.length > 3 || execution.shapes().get(inputs[0]).getRank() != 4 ||
                execution.shapes().get(inputs[1]).getRank() != 4 || execution.shapes().get(output).getRank() != 4) {
            throw unsupported(node, "Conv requires rank-4 input, weights, and output");
        }
        TensorShape inputShape = execution.shapes().get(inputs[0]);
        TensorShape weightShape = execution.shapes().get(inputs[1]);
        TensorShape outputShape = execution.shapes().get(output);
        ByteBuffer params = parameterData(node);
        int groups = params.getInt(4);
        int kernelHeight = params.getInt(8);
        int kernelWidth = params.getInt(12);
        int strideHeight = params.getInt(16);
        int strideWidth = params.getInt(20);
        int dilationHeight = params.getInt(24);
        int dilationWidth = params.getInt(28);
        int padTop = params.getInt(32);
        int padLeft = params.getInt(36);
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
                dilationHeight, dilationWidth, padTop, padLeft, groups,
                outputShape.get(2), outputShape.get(3));
    }

    private void executeSoftmax(NodeInfo node, float[] storage, int output) {
        int[] inputs = nodeInputs.get(node);
        if (inputs.length != 1 || execution.length(inputs[0]) != execution.length(output)) throw unsupported(node, "shape mismatch");
        ByteBuffer params = parameterData(node);
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

    private void executePool(NodeInfo node, float[] storage, int output) {
        int[] inputs = nodeInputs.get(node);
        if (inputs.length != 1 || execution.shapes().get(inputs[0]).getRank() != 4 || execution.shapes().get(output).getRank() != 4) {
            throw unsupported(node, "pool requires rank-4 input and output");
        }
        ByteBuffer params = parameterData(node);
        TensorShape inputShape = execution.shapes().get(inputs[0]);
        TensorShape outputShape = execution.shapes().get(output);
        backend.pool(data(inputs[0], storage), offset(inputs[0]), storage, offset(output),
                inputShape.get(0), inputShape.get(1), inputShape.get(2), inputShape.get(3),
                params.getInt(8), params.getInt(12), params.getInt(16), params.getInt(20),
                params.getInt(24), params.getInt(28), outputShape.get(2), outputShape.get(3),
                node.getOperator() == OperatorType.MAX_POOL, params.getInt(44) != 0);
    }

    private void executeResize(NodeInfo node, float[] storage, int output) {
        int[] inputs = nodeInputs.get(node);
        if (inputs.length != 1 || execution.shapes().get(inputs[0]).getRank() != 4 || execution.shapes().get(output).getRank() != 4) {
            throw unsupported(node, "Resize requires rank-4 input and output");
        }
        ByteBuffer params = parameterData(node);
        TensorShape inputShape = execution.shapes().get(inputs[0]);
        TensorShape outputShape = execution.shapes().get(output);
        backend.resizeNearest(data(inputs[0], storage), offset(inputs[0]), storage, offset(output),
                inputShape.get(0), inputShape.get(1), inputShape.get(2), inputShape.get(3),
                outputShape.get(2), outputShape.get(3), params.getFloat(12), params.getFloat(16));
    }

    private void executeConvTranspose(NodeInfo node, float[] storage, int output) {
        int[] inputs = nodeInputs.get(node);
        if (inputs.length < 2 || inputs.length > 3 || execution.shapes().get(inputs[0]).getRank() != 4 ||
                execution.shapes().get(inputs[1]).getRank() != 4 || execution.shapes().get(output).getRank() != 4) {
            throw unsupported(node, "ConvTranspose requires rank-4 input, weights, and output");
        }
        TensorShape inputShape = execution.shapes().get(inputs[0]);
        TensorShape weightShape = execution.shapes().get(inputs[1]);
        TensorShape outputShape = execution.shapes().get(output);
        ByteBuffer params = parameterData(node);
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

    private void executeReduceMean(NodeInfo node, float[] storage, int output) {
        int[] inputs = nodeInputs.get(node);
        if (inputs.length != 1) throw unsupported(node, "ReduceMean requires one input");
        ByteBuffer params = parameterData(node);
        int[] axes = reduceAxes.get(node);
        if (axes == null) {
            int count = params.getShort(2) & 0xffff;
            axes = new int[count];
            for (int i = 0; i < count; i++) axes[i] = params.getInt(12 + i * 4);
        }
        backend.reduceMean(data(inputs[0], storage), offset(inputs[0]), storage, offset(output),
                execution.shapes().get(inputs[0]).dimensionsUnsafe(), axes, params.getInt(4) != 0);
    }

    private void executeBatchNormalization(NodeInfo node, float[] storage, int output) {
        int[] inputs = nodeInputs.get(node);
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
        ByteBuffer params = parameterData(node);
        float epsilon = params.getFloat(4);
        if (!Float.isFinite(epsilon) || epsilon <= 0.0f) {
            throw unsupported(node, "BatchNormalization epsilon is invalid");
        }
        backend.batchNormalization(data(inputs[0], storage), offset(inputs[0]),
                data(inputs[1], storage), offset(inputs[1]), data(inputs[2], storage), offset(inputs[2]),
                data(inputs[3], storage), offset(inputs[3]), data(inputs[4], storage), offset(inputs[4]),
                epsilon, storage, offset(output), inputShape.dimensionsUnsafe());
    }

    private void executeSqueezeOrUnsqueeze(NodeInfo node, float[] storage, int output) {
        int[] inputs = nodeInputs.get(node);
        if (inputs.length != 1 || execution.length(inputs[0]) != execution.length(output)) {
            throw unsupported(node, "layout operator shape mismatch");
        }
        TensorShape inputShape = execution.shapes().get(inputs[0]);
        TensorShape outputShape = execution.shapes().get(output);
        ByteBuffer params = parameterData(node);
        int count = params.getShort(2) & 0xffff;
        int[] axes = new int[count];
        for (int i = 0; i < count; i++) axes[i] = params.getInt(4 + i * 4);
        if (node.getOperator() == OperatorType.SQUEEZE) {
            validateSqueeze(inputShape, outputShape, axes);
        } else {
            validateUnsqueeze(inputShape, outputShape, axes);
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

    private void executeConcat(NodeInfo node, float[] storage, int output) {
        int[] inputs = nodeInputs.get(node);
        if (inputs.length == 0) throw unsupported(node, "Concat requires inputs");
        ConcatPlan plan = concatPlans.get(node);
        TensorShape shape = execution.shapes().get(inputs[0]);
        if (plan != null) {
            for (int i = 0; i < inputs.length; i++) plan.values[i] = data(inputs[i], storage);
            backend.concat(plan.values, plan.offsets, storage, offset(output),
                    shape.dimensionsUnsafe(), plan.axis, plan.axisSizes);
            return;
        }
        ByteBuffer params = parameterData(node);
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

    private void executeSlice(NodeInfo node, float[] storage, int output) {
        int[] inputs = nodeInputs.get(node);
        if (inputs.length != 1) throw unsupported(node, "Slice requires one input");
        int[] starts = sliceStarts.get(node);
        int[] axes = sliceAxes.get(node);
        int[] steps = sliceSteps.get(node);
        if (starts == null) {
            ByteBuffer params = parameterData(node);
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

    private void executeTranspose(NodeInfo node, float[] storage, int output) {
        int[] inputs = nodeInputs.get(node);
        if (inputs.length != 1 || execution.length(inputs[0]) != execution.length(output)) throw unsupported(node, "shape mismatch");
        TensorShape inputShape = execution.shapes().get(inputs[0]);
        TensorShape outputShape = execution.shapes().get(output);
        int[] permutation = transposePermutations.get(node);
        int[] inputStrides = transposeInputStrides.get(node);
        int rank;
        if (permutation == null) {
            ByteBuffer params = parameterData(node);
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

    private void prepareTranspose(NodeInfo node) {
        if (node.getOperator() != OperatorType.TRANSPOSE) return;
        int[] inputs = nodeInputs.get(node);
        int[] outputs = nodeOutputs.get(node);
        if (inputs.length != 1 || outputs.length != 1) return;
        TensorShape inputShape = execution.shapes().get(inputs[0]);
        TensorShape outputShape = execution.shapes().get(outputs[0]);
        ByteBuffer params = parameterData(node);
        int rank = params.getShort(2) & 0xffff;
        if (rank != inputShape.getRank() || rank != outputShape.getRank()) return;
        int[] permutation = new int[rank];
        for (int i = 0; i < rank; i++) permutation[i] = params.getInt(4 + i * 4);
        transposePermutations.put(node, permutation);
        transposeInputStrides.put(node, strides(inputShape));
    }

    private void prepareReduceMean(NodeInfo node) {
        if (node.getOperator() != OperatorType.REDUCE_MEAN) return;
        ByteBuffer params = parameterData(node);
        int count = params.getShort(2) & 0xffff;
        if (12L + count * 4L > params.limit()) return;
        int[] axes = new int[count];
        for (int i = 0; i < count; i++) axes[i] = params.getInt(12 + i * 4);
        reduceAxes.put(node, axes);
    }

    private void prepareSlice(NodeInfo node) {
        if (node.getOperator() != OperatorType.SLICE) return;
        ByteBuffer params = parameterData(node);
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
        sliceStarts.put(node, starts);
        sliceAxes.put(node, axes);
        sliceSteps.put(node, steps);
    }

    private void prepareConcat(NodeInfo node) {
        if (node.getOperator() != OperatorType.CONCAT) return;
        int[] inputs = nodeInputs.get(node);
        if (inputs.length == 0) return;
        ByteBuffer params = parameterData(node);
        int axis = params.getInt(4);
        TensorShape shape = execution.shapes().get(inputs[0]);
        if (axis < 0) axis += shape.getRank();
        if (axis < 0 || axis >= shape.getRank()) return;
        int[] outputs = nodeOutputs.get(node);
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
        concatPlans.put(node, plan);
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

    private int[] strides(TensorShape shape) {
        int[] strides = new int[shape.getRank()];
        int stride = 1;
        for (int axis = shape.getRank() - 1; axis >= 0; axis--) {
            strides[axis] = stride;
            stride *= shape.get(axis);
        }
        return strides;
    }

    private int indexOf(NodeInfo target) {
        Integer index = nodeIndexes.get(target);
        if (index == null) throw new IllegalStateException("prepared node is not owned by model");
        return index;
    }

    private ByteBuffer parameterData(NodeInfo node) {
        return parameters[indexOf(node)];
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

    private OcrException unsupported(NodeInfo node, String detail) {
        return new OcrException(OcrErrorCode.UNSUPPORTED_OPERATOR, node.getOperator() + ": " + detail);
    }

    private OcrException unsupported(NodeInfo node, String detail, Throwable cause) {
        return new OcrException(OcrErrorCode.UNSUPPORTED_OPERATOR,
                node.getOperator() + ": " + detail, cause);
    }

    private void ensureOpen() {
        if (closed) throw new OcrException(OcrErrorCode.INVALID_ARGUMENT, "inference session is closed");
    }
}
