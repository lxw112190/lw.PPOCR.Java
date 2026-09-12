package io.github.lxw112190.ppocr.runtime;

import io.github.lxw112190.ppocr.kernels.KernelBackend;
import io.github.lxw112190.ppocr.kernels.ScalarBackend;
import io.github.lxw112190.ppocr.model.LwmModel;
import io.github.lxw112190.ppocr.model.OcrErrorCode;
import io.github.lxw112190.ppocr.model.OcrException;
import io.github.lxw112190.ppocr.model.OperatorType;
import io.github.lxw112190.ppocr.model.NodeInfo;
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
