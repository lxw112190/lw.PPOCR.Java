package io.github.lxw112190.ppocr.runtime;

import io.github.lxw112190.ppocr.model.LwmModel;
import io.github.lxw112190.ppocr.model.NodeInfo;
import io.github.lxw112190.ppocr.model.OcrErrorCode;
import io.github.lxw112190.ppocr.model.OcrException;
import io.github.lxw112190.ppocr.model.OperatorType;
import io.github.lxw112190.ppocr.model.TensorInfo;
import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/** Resolves model metadata into concrete shapes for a prepared execution. */
public final class ShapeResolver {
    private ShapeResolver() { }

    public static List<TensorShape> resolveStatic(LwmModel model) {
        if (model == null) {
            throw new OcrException(OcrErrorCode.INVALID_ARGUMENT, "model is required");
        }
        return resolve(model, Collections.<TensorShape>emptyList());
    }

    public static List<TensorShape> resolve(LwmModel model, List<TensorShape> inputShapes) {
        if (model == null || inputShapes == null) {
            throw new OcrException(OcrErrorCode.INVALID_ARGUMENT, "model and input shapes are required");
        }
        if (inputShapes.size() != model.getGraphInputs().size() && !inputShapes.isEmpty()) {
            throw new OcrException(OcrErrorCode.INVALID_ARGUMENT, "input shape count does not match model");
        }
        List<TensorShape> resolved = new ArrayList<TensorShape>(model.getTensors().size());
        for (int i = 0; i < model.getTensors().size(); i++) {
            TensorInfo tensor = model.getTensors().get(i);
            int inputPosition = model.getGraphInputs().indexOf(i);
            if (inputPosition >= 0 && !inputShapes.isEmpty()) {
                resolved.add(bindInput(tensor.getDimensions(), inputShapes.get(inputPosition), i));
            } else if (isStatic(tensor.getDimensions())) {
                resolved.add(new TensorShape(tensor.getDimensions()));
            } else {
                resolved.add(null);
            }
        }

        List<NodeInfo> nodes = model.getNodes();
        for (int nodeIndex = 0; nodeIndex < nodes.size(); nodeIndex++) {
            NodeInfo node = nodes.get(nodeIndex);
            int[] inputs = node.getInputs();
            int[] outputs = node.getOutputs();
            if (outputs.length != 1) {
                throw invalid("shape inference requires one output for node " + nodeIndex);
            }
            TensorShape[] inputValues = new TensorShape[inputs.length];
            for (int i = 0; i < inputs.length; i++) {
                TensorShape value = resolved.get(inputs[i]);
                if (value == null) {
                    throw invalid("input tensor " + inputs[i] + " is unresolved at node " + nodeIndex);
                }
                inputValues[i] = value;
            }
            int outputIndex = outputs[0];
            if (resolved.get(outputIndex) == null) {
                TensorShape inferred = inferNode(node, nodeIndex, inputValues, model.parameterData(nodeIndex));
                resolved.set(outputIndex, bindOutput(model.getTensors().get(outputIndex).getDimensions(), inferred,
                        outputIndex));
            }
        }
        for (int i = 0; i < resolved.size(); i++) {
            if (resolved.get(i) == null) {
                throw invalid("tensor " + i + " still has unresolved dynamic dimensions");
            }
        }
        return Collections.unmodifiableList(resolved);
    }

    private static TensorShape bindInput(int[] declared, TensorShape actual, int index) {
        if (actual == null || actual.getRank() != declared.length) {
            throw invalid("input rank does not match model tensor " + index);
        }
        int[] dimensions = actual.getDimensions();
        for (int axis = 0; axis < declared.length; axis++) {
            if (dimensions[axis] <= 0 || (declared[axis] != -1 && declared[axis] != dimensions[axis])) {
                throw invalid("input dimension does not match model tensor " + index);
            }
        }
        return actual;
    }

    private static TensorShape bindOutput(int[] declared, TensorShape inferred, int index) {
        int[] dimensions = inferred.getDimensions();
        if (declared.length != dimensions.length) {
            throw invalid("inferred rank does not match model tensor " + index);
        }
        for (int axis = 0; axis < declared.length; axis++) {
            if (declared[axis] != -1 && declared[axis] != dimensions[axis]) {
                throw invalid("inferred dimension does not match model tensor " + index);
            }
        }
        return inferred;
    }

    private static TensorShape inferNode(NodeInfo node, int nodeIndex, TensorShape[] inputs, ByteBuffer parameters) {
        OperatorType type = node.getOperator();
        switch (type) {
            case ADD:
            case SUB:
            case MUL:
            case DIV:
            case POW:
                requireInputCount(inputs, 2, nodeIndex);
                requireSameShape(inputs[0], inputs[1], nodeIndex);
                return inputs[0];
            case RELU:
            case SIGMOID:
            case ERF:
            case HARD_SIGMOID:
            case SQRT:
            case BATCH_NORMALIZATION:
            case SOFTMAX:
                requireInputCount(inputs, 1, nodeIndex);
                return inputs[0];
            case REDUCE_MEAN:
                requireInputCount(inputs, 1, nodeIndex);
                return reduceMeanShape(inputs[0], parameters, nodeIndex);
            case CONCAT:
                return concatShape(inputs, parameters, nodeIndex);
            case CONV:
                requireInputRange(inputs, 2, 3, nodeIndex);
                return convShape(inputs[0], inputs[1], parameters, nodeIndex, false);
            case CONV_TRANSPOSE:
                requireInputRange(inputs, 2, 3, nodeIndex);
                return convShape(inputs[0], inputs[1], parameters, nodeIndex, true);
            case AVERAGE_POOL:
            case MAX_POOL:
                requireInputCount(inputs, 1, nodeIndex);
                return poolShape(inputs[0], parameters, nodeIndex);
            case RESIZE:
                requireInputCount(inputs, 1, nodeIndex);
                return resizeShape(inputs[0], parameters, nodeIndex);
            case TRANSPOSE:
                requireInputCount(inputs, 1, nodeIndex);
                return transposeShape(inputs[0], parameters, nodeIndex);
            case SQUEEZE:
                requireInputCount(inputs, 1, nodeIndex);
                return squeezeShape(inputs[0], parameters, nodeIndex, false);
            case UNSQUEEZE:
                requireInputCount(inputs, 1, nodeIndex);
                return squeezeShape(inputs[0], parameters, nodeIndex, true);
            case RESHAPE:
                requireInputCount(inputs, 1, nodeIndex);
                return inputs[0];
            case MAT_MUL:
                requireInputCount(inputs, 2, nodeIndex);
                requireRank(inputs[0], 2, nodeIndex);
                requireRank(inputs[1], 2, nodeIndex);
                if (inputs[0].get(1) != inputs[1].get(0)) {
                    throw invalid("MAT_MUL dimensions do not match at node " + nodeIndex);
                }
                return new TensorShape(inputs[0].get(0), inputs[1].get(1));
            default:
                throw invalid("shape inference is not implemented for " + type + " at node " + nodeIndex);
        }
    }

    private static TensorShape convShape(TensorShape input, TensorShape weights, ByteBuffer p,
                                         int nodeIndex, boolean transpose) {
        requireRank(input, 4, nodeIndex);
        requireRank(weights, 4, nodeIndex);
        int groups = positive(p.getInt(4), "groups", nodeIndex);
        int kernelHeight = positive(p.getInt(8), "kernel height", nodeIndex);
        int kernelWidth = positive(p.getInt(12), "kernel width", nodeIndex);
        int strideHeight = positive(p.getInt(16), "stride height", nodeIndex);
        int strideWidth = positive(p.getInt(20), "stride width", nodeIndex);
        int dilationHeight = positive(p.getInt(24), "dilation height", nodeIndex);
        int dilationWidth = positive(p.getInt(28), "dilation width", nodeIndex);
        int padTop = nonNegative(p.getInt(32), "top padding", nodeIndex);
        int padLeft = nonNegative(p.getInt(36), "left padding", nodeIndex);
        int height = input.get(2);
        int width = input.get(3);
        int channels;
        if (transpose) {
            if (weights.get(0) != input.get(1)) {
                throw invalid("CONV_TRANSPOSE input channels do not match at node " + nodeIndex);
            }
            channels = Math.multiplyExact(weights.get(1), groups);
            height = transposedOut(height, strideHeight, padTop, dilationHeight, kernelHeight, nodeIndex);
            width = transposedOut(width, strideWidth, padLeft, dilationWidth, kernelWidth, nodeIndex);
        } else {
            if (weights.get(1) * groups != input.get(1)) {
                throw invalid("CONV input channels do not match at node " + nodeIndex);
            }
            channels = weights.get(0);
            height = convOut(height, strideHeight, padTop, dilationHeight, kernelHeight, nodeIndex);
            width = convOut(width, strideWidth, padLeft, dilationWidth, kernelWidth, nodeIndex);
        }
        return new TensorShape(input.get(0), channels, height, width);
    }

    private static TensorShape poolShape(TensorShape input, ByteBuffer p, int nodeIndex) {
        requireRank(input, 4, nodeIndex);
        int kernelHeight = positive(p.getInt(8), "kernel height", nodeIndex);
        int kernelWidth = positive(p.getInt(12), "kernel width", nodeIndex);
        int strideHeight = positive(p.getInt(16), "stride height", nodeIndex);
        int strideWidth = positive(p.getInt(20), "stride width", nodeIndex);
        int padTop = nonNegative(p.getInt(24), "top padding", nodeIndex);
        int padLeft = nonNegative(p.getInt(28), "left padding", nodeIndex);
        return new TensorShape(input.get(0), input.get(1),
                convOut(input.get(2), strideHeight, padTop, 1, kernelHeight, nodeIndex),
                convOut(input.get(3), strideWidth, padLeft, 1, kernelWidth, nodeIndex));
    }

    private static TensorShape resizeShape(TensorShape input, ByteBuffer p, int nodeIndex) {
        requireRank(input, 4, nodeIndex);
        float scaleHeight = p.getFloat(12);
        float scaleWidth = p.getFloat(16);
        if (!(scaleHeight > 0.0f) || !(scaleWidth > 0.0f)) {
            throw invalid("RESIZE scales must be positive at node " + nodeIndex);
        }
        int height = positiveDimension((int) Math.floor(input.get(2) * scaleHeight), nodeIndex);
        int width = positiveDimension((int) Math.floor(input.get(3) * scaleWidth), nodeIndex);
        return new TensorShape(input.get(0), input.get(1), height, width);
    }

    private static TensorShape reduceMeanShape(TensorShape input, ByteBuffer p, int nodeIndex) {
        int axesCount = Short.toUnsignedInt(p.getShort(2));
        boolean keepDimensions = p.getInt(4) != 0;
        boolean[] reduced = new boolean[input.getRank()];
        for (int i = 0; i < axesCount; i++) {
            int axis = normalizeAxis(p.getInt(12 + i * 4), input.getRank(), nodeIndex);
            reduced[axis] = true;
        }
        int outputRank = keepDimensions ? input.getRank() : input.getRank() - axesCount;
        int[] output = new int[outputRank];
        int offset = 0;
        for (int axis = 0; axis < input.getRank(); axis++) {
            if (!reduced[axis] || keepDimensions) {
                output[offset++] = reduced[axis] ? 1 : input.get(axis);
            }
        }
        return new TensorShape(output);
    }

    private static TensorShape concatShape(TensorShape[] inputs, ByteBuffer p, int nodeIndex) {
        if (inputs.length == 0) throw invalid("CONCAT requires inputs at node " + nodeIndex);
        int axis = normalizeAxis(p.getInt(4), inputs[0].getRank(), nodeIndex);
        int[] output = inputs[0].getDimensions();
        long total = output[axis];
        for (int i = 1; i < inputs.length; i++) {
            requireSameRank(inputs[0], inputs[i], nodeIndex);
            for (int dimension = 0; dimension < output.length; dimension++) {
                if (dimension != axis && output[dimension] != inputs[i].get(dimension)) {
                    throw invalid("CONCAT dimensions do not match at node " + nodeIndex);
                }
            }
            total += inputs[i].get(axis);
        }
        if (total > Integer.MAX_VALUE) throw invalid("CONCAT output is too large at node " + nodeIndex);
        output[axis] = (int) total;
        return new TensorShape(output);
    }

    private static TensorShape transposeShape(TensorShape input, ByteBuffer p, int nodeIndex) {
        int rank = Short.toUnsignedInt(p.getShort(2));
        requireRank(input, rank, nodeIndex);
        int[] output = new int[rank];
        boolean[] used = new boolean[rank];
        for (int i = 0; i < rank; i++) {
            int axis = p.getInt(4 + i * 4);
            if (axis < 0 || axis >= rank || used[axis]) throw invalid("invalid TRANSPOSE permutation at node " + nodeIndex);
            used[axis] = true;
            output[i] = input.get(axis);
        }
        return new TensorShape(output);
    }

    private static TensorShape squeezeShape(TensorShape input, ByteBuffer p, int nodeIndex, boolean unsqueeze) {
        int axesCount = Short.toUnsignedInt(p.getShort(2));
        int[] axes = new int[axesCount];
        for (int i = 0; i < axesCount; i++) axes[i] = p.getInt(4 + i * 4);
        if (unsqueeze) {
            int[] output = input.getDimensions();
            for (int axis : axes) {
                int normalized = axis < 0 ? axis + output.length + 1 : axis;
                if (normalized < 0 || normalized > output.length) throw invalid("invalid UNSQUEEZE axis at node " + nodeIndex);
                int[] next = new int[output.length + 1];
                System.arraycopy(output, 0, next, 0, normalized);
                next[normalized] = 1;
                System.arraycopy(output, normalized, next, normalized + 1, output.length - normalized);
                output = next;
            }
            return new TensorShape(output);
        }
        boolean[] remove = new boolean[input.getRank()];
        for (int axis : axes) {
            int normalized = normalizeAxis(axis, input.getRank(), nodeIndex);
            if (input.get(normalized) != 1) throw invalid("SQUEEZE axis is not one at node " + nodeIndex);
            remove[normalized] = true;
        }
        int[] output = new int[input.getRank() - axesCount];
        int offset = 0;
        for (int axis = 0; axis < input.getRank(); axis++) if (!remove[axis]) output[offset++] = input.get(axis);
        return new TensorShape(output);
    }

    private static int convOut(int input, int stride, int pad, int dilation, int kernel, int nodeIndex) {
        long numerator = (long) input + 2L * pad - (long) dilation * (kernel - 1) - 1L;
        return positiveDimension((int) (numerator / stride + 1L), nodeIndex);
    }

    private static int transposedOut(int input, int stride, int pad, int dilation, int kernel, int nodeIndex) {
        long result = (long) (input - 1) * stride - 2L * pad + (long) dilation * (kernel - 1) + 1L;
        return positiveDimension((int) result, nodeIndex);
    }

    private static boolean isStatic(int[] dimensions) {
        for (int dimension : dimensions) if (dimension == -1) return false;
        return true;
    }

    private static int positive(int value, String name, int nodeIndex) {
        if (value <= 0) throw invalid(name + " must be positive at node " + nodeIndex);
        return value;
    }

    private static int nonNegative(int value, String name, int nodeIndex) {
        if (value < 0) throw invalid(name + " must not be negative at node " + nodeIndex);
        return value;
    }

    private static int positiveDimension(int value, int nodeIndex) {
        if (value <= 0) throw invalid("inferred dimension is invalid at node " + nodeIndex);
        return value;
    }

    private static int normalizeAxis(int axis, int rank, int nodeIndex) {
        int normalized = axis < 0 ? axis + rank : axis;
        if (normalized < 0 || normalized >= rank) throw invalid("axis is out of range at node " + nodeIndex);
        return normalized;
    }

    private static void requireInputCount(TensorShape[] inputs, int count, int nodeIndex) {
        if (inputs.length != count) throw invalid("unexpected input count at node " + nodeIndex);
    }

    private static void requireInputRange(TensorShape[] inputs, int minimum, int maximum, int nodeIndex) {
        if (inputs.length < minimum || inputs.length > maximum) {
            throw invalid("unexpected input count at node " + nodeIndex);
        }
    }

    private static void requireRank(TensorShape shape, int rank, int nodeIndex) {
        if (shape.getRank() != rank) throw invalid("unexpected tensor rank at node " + nodeIndex);
    }

    private static void requireSameRank(TensorShape left, TensorShape right, int nodeIndex) {
        if (left.getRank() != right.getRank()) throw invalid("tensor ranks do not match at node " + nodeIndex);
    }

    private static void requireSameShape(TensorShape left, TensorShape right, int nodeIndex) {
        requireSameRank(left, right, nodeIndex);
        for (int axis = 0; axis < left.getRank(); axis++) {
            if (left.get(axis) != right.get(axis)) throw invalid("tensor shapes do not match at node " + nodeIndex);
        }
    }

    private static OcrException invalid(String message) {
        return new OcrException(OcrErrorCode.INVALID_MODEL, message);
    }
}
