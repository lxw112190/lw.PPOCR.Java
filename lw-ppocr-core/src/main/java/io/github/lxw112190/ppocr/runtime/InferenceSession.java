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
import java.util.List;

/** Prepared, reusable scalar execution session for the initial operator subset. */
public final class InferenceSession implements AutoCloseable {
    private final PreparedExecution execution;
    private final Workspace workspace;
    private final KernelBackend backend;
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
        int[] inputs = node.getInputs();
        int[] outputs = node.getOutputs();
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
                ByteBuffer hardSigmoid = execution.model().parameterData(indexOf(node));
                backend.hardSigmoid(left, leftOffset, storage, offset(output), execution.length(output),
                        hardSigmoid.getFloat(4), hardSigmoid.getFloat(8));
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
            case TRANSPOSE:
                executeTranspose(node, storage, output);
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
        int[] inputs = node.getInputs();
        if (inputs.length < 2 || inputs.length > 3 || execution.shapes().get(inputs[0]).getRank() != 4 ||
                execution.shapes().get(inputs[1]).getRank() != 4 || execution.shapes().get(output).getRank() != 4) {
            throw unsupported(node, "Conv requires rank-4 input, weights, and output");
        }
        TensorShape inputShape = execution.shapes().get(inputs[0]);
        TensorShape weightShape = execution.shapes().get(inputs[1]);
        TensorShape outputShape = execution.shapes().get(output);
        ByteBuffer params = execution.model().parameterData(indexOf(node));
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
        int[] inputs = node.getInputs();
        if (inputs.length != 1 || execution.length(inputs[0]) != execution.length(output)) throw unsupported(node, "shape mismatch");
        ByteBuffer params = execution.model().parameterData(indexOf(node));
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

    private void executeReduceMean(NodeInfo node, float[] storage, int output) {
        int[] inputs = node.getInputs();
        if (inputs.length != 1) throw unsupported(node, "ReduceMean requires one input");
        ByteBuffer params = execution.model().parameterData(indexOf(node));
        int count = params.getShort(2) & 0xffff;
        int[] axes = new int[count];
        for (int i = 0; i < count; i++) axes[i] = params.getInt(12 + i * 4);
        backend.reduceMean(data(inputs[0], storage), offset(inputs[0]), storage, offset(output),
                execution.shapes().get(inputs[0]).getDimensions(), axes, params.getInt(4) != 0);
    }

    private void executeConcat(NodeInfo node, float[] storage, int output) {
        int[] inputs = node.getInputs();
        if (inputs.length == 0) throw unsupported(node, "Concat requires inputs");
        ByteBuffer params = execution.model().parameterData(indexOf(node));
        int axis = params.getInt(4);
        TensorShape shape = execution.shapes().get(inputs[0]);
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
        backend.concat(values, offsets, storage, offset(output), shape.getDimensions(), axis, axisSizes);
    }

    private void executeSlice(NodeInfo node, float[] storage, int output) {
        int[] inputs = node.getInputs();
        if (inputs.length != 1) throw unsupported(node, "Slice requires one input");
        ByteBuffer params = execution.model().parameterData(indexOf(node));
        int count = params.getShort(2) & 0xffff;
        int[] starts = new int[count];
        int[] axes = new int[count];
        int[] steps = new int[count];
        for (int i = 0; i < count; i++) {
            starts[i] = params.getInt(4 + i * 4);
            axes[i] = params.getInt(68 + i * 4);
            steps[i] = params.getInt(100 + i * 4);
        }
        backend.slice(data(inputs[0], storage), offset(inputs[0]), storage, offset(output),
                execution.shapes().get(inputs[0]).getDimensions(), starts, axes, steps);
    }

    private void executeTranspose(NodeInfo node, float[] storage, int output) {
        int[] inputs = node.getInputs();
        if (inputs.length != 1 || execution.length(inputs[0]) != execution.length(output)) throw unsupported(node, "shape mismatch");
        TensorShape inputShape = execution.shapes().get(inputs[0]);
        TensorShape outputShape = execution.shapes().get(output);
        ByteBuffer params = execution.model().parameterData(indexOf(node));
        int rank = params.getShort(2) & 0xffff;
        if (rank != inputShape.getRank() || rank != outputShape.getRank()) throw unsupported(node, "Transpose rank mismatch");
        int[] permutation = new int[rank];
        for (int i = 0; i < rank; i++) permutation[i] = params.getInt(4 + i * 4);
        int[] inputStrides = strides(inputShape);
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
        List<NodeInfo> nodes = execution.model().getNodes();
        for (int i = 0; i < nodes.size(); i++) if (nodes.get(i) == target) return i;
        throw new IllegalStateException("prepared node is not owned by model");
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

    private void ensureOpen() {
        if (closed) throw new OcrException(OcrErrorCode.INVALID_ARGUMENT, "inference session is closed");
    }
}
