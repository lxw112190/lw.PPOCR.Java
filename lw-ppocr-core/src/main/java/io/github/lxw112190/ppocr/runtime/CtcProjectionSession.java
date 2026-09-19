package io.github.lxw112190.ppocr.runtime;

import io.github.lxw112190.ppocr.kernels.KernelBackend;
import io.github.lxw112190.ppocr.kernels.ProjectionArgMaxBackend;
import io.github.lxw112190.ppocr.model.LwmModel;
import io.github.lxw112190.ppocr.model.NodeInfo;
import io.github.lxw112190.ppocr.model.OcrErrorCode;
import io.github.lxw112190.ppocr.model.OcrException;
import io.github.lxw112190.ppocr.model.OperatorType;
import io.github.lxw112190.ppocr.model.TensorInfo;
import java.nio.ByteBuffer;
import java.util.List;

/** Specialized REC session which avoids materializing the terminal class matrix. */
public final class CtcProjectionSession implements AutoCloseable {
    public static final String DISABLE_PROPERTY = "lwppocr.disableProjectionFusion";

    private final InferenceSession prefix;
    private final ProjectionArgMaxBackend backend;
    private final PreparedMatMulWeights weights;
    private final float[] bias;
    private final float[] rowScratch;
    private final int rows;
    private float[] compatibilityOutput;
    private boolean closed;

    private CtcProjectionSession(InferenceSession prefix, ProjectionArgMaxBackend backend,
                                 PreparedMatMulWeights weights, float[] bias,
                                 int rows) {
        this.prefix = prefix;
        this.backend = backend;
        this.weights = weights;
        this.bias = bias;
        this.rowScratch = new float[Math.min(rows, 4) * weights.getColumns()];
        this.rows = rows;
    }

    /** Returns null when the graph/backend does not safely support this specialization. */
    public static CtcProjectionSession tryCreate(LwmModel model, List<TensorShape> inputShapes,
                                                 KernelBackend kernelBackend, int classCount) {
        if (Boolean.getBoolean(DISABLE_PROPERTY) ||
                !(kernelBackend instanceof ProjectionArgMaxBackend)) return null;
        Tail tail = Tail.detect(model, inputShapes, classCount);
        if (tail == null) return null;
        ProjectionArgMaxBackend projection = (ProjectionArgMaxBackend) kernelBackend;
        if (!projection.supportsProjectionArgMax(tail.rows, tail.inner, tail.columns)) return null;

        InferenceSession prefix = new InferenceSession(model, inputShapes, kernelBackend,
                tail.projectionNodeIndex, tail.activationTensor);
        try {
            PreparedExecution execution = prefix.execution();
            float[] canonicalWeights = execution.constant(tail.weightTensor);
            float[] bias = execution.constant(tail.biasTensor);
            if (canonicalWeights == null || bias == null) {
                prefix.close();
                return null;
            }
            PreparedMatMulWeights prepared = new PreparedMatMulWeights(canonicalWeights, 0,
                    tail.inner, tail.columns);
            return new CtcProjectionSession(prefix, projection, prepared, bias,
                    tail.rows);
        } catch (RuntimeException e) {
            prefix.close();
            throw e;
        }
    }

    public void run(float[] input, int[] bestIndices, float[] bestLogits,
                    float[] bestProbabilities) {
        ensureOpen();
        validateOutputs(bestIndices, bestLogits, bestProbabilities);
        if (compatibilityOutput == null) {
            compatibilityOutput = new float[prefix.outputView().length()];
        }
        prefix.run(input, compatibilityOutput);
        project(compatibilityOutput, 0, bestIndices, bestLogits, bestProbabilities);
    }

    /** Returns the input view of the prefix execution without allocating a copy. */
    public FloatTensorView inputView() {
        ensureOpen();
        return prefix.inputView();
    }

    /** Executes the prefix and projects its bound output without materializing activations. */
    public void runBound(int[] bestIndices, float[] bestLogits,
                         float[] bestProbabilities) {
        ensureOpen();
        validateOutputs(bestIndices, bestLogits, bestProbabilities);
        prefix.runBound();
        FloatTensorView activation = prefix.outputView();
        project(activation.array(), activation.offset(), bestIndices, bestLogits,
                bestProbabilities);
    }

    private void project(float[] activation, int activationOffset, int[] bestIndices,
                         float[] bestLogits, float[] bestProbabilities) {
        backend.projectionArgMax(activation, activationOffset, weights.canonical(), weights.offset(),
                bias, 0, rows, weights.getInner(), weights.getColumns(), bestIndices,
                bestLogits, bestProbabilities, rowScratch);
    }

    private void validateOutputs(int[] bestIndices, float[] bestLogits,
                                 float[] bestProbabilities) {
        if (bestIndices == null || bestLogits == null || bestProbabilities == null ||
                bestIndices.length < rows || bestLogits.length < rows ||
                bestProbabilities.length < rows) {
            throw new OcrException(OcrErrorCode.INVALID_ARGUMENT,
                    "compact CTC output is too small");
        }
    }

    public int getTimeSteps() { return rows; }
    public int getClassCount() { return weights.getColumns(); }
    public long getDenseOutputBytesAvoided() { return (long) rows * weights.getColumns() * 4L; }
    public long getWorkspaceBytes() { ensureOpen(); return prefix.execution().workspacePlan().getTotalBytes(); }

    @Override
    public void close() {
        if (!closed) {
            closed = true;
            prefix.close();
        }
    }

    private void ensureOpen() {
        if (closed) throw new OcrException(OcrErrorCode.INVALID_ARGUMENT,
                "CTC projection session is closed");
    }

    private static final class Tail {
        final int projectionNodeIndex;
        final int activationTensor;
        final int weightTensor;
        final int biasTensor;
        final int rows;
        final int inner;
        final int columns;

        Tail(int projectionNodeIndex, int activationTensor, int weightTensor, int biasTensor,
             int rows, int inner, int columns) {
            this.projectionNodeIndex = projectionNodeIndex;
            this.activationTensor = activationTensor;
            this.weightTensor = weightTensor;
            this.biasTensor = biasTensor;
            this.rows = rows;
            this.inner = inner;
            this.columns = columns;
        }

        static Tail detect(LwmModel model, List<TensorShape> inputShapes, int classCount) {
            if (model == null || inputShapes == null || classCount <= 0 ||
                    model.getGraphInputs().size() != 1 || model.getGraphOutputs().size() != 1) return null;
            CompiledModel compiled = CompiledModel.acquire(model);
            if (compiled.nodeCount() < 3) return null;
            int projectionIndex = compiled.nodeCount() - 3;
            int addIndex = compiled.nodeCount() - 2;
            int softmaxIndex = compiled.nodeCount() - 1;
            NodeInfo projection = compiled.node(projectionIndex);
            NodeInfo add = compiled.node(addIndex);
            NodeInfo softmax = compiled.node(softmaxIndex);
            if (projection.getOperator() != OperatorType.MAT_MUL ||
                    add.getOperator() != OperatorType.ADD ||
                    softmax.getOperator() != OperatorType.SOFTMAX) return null;
            int[] projectionInputs = compiled.nodeInputs(projectionIndex);
            int[] projectionOutputs = compiled.nodeOutputs(projectionIndex);
            int[] addInputs = compiled.nodeInputs(addIndex);
            int[] addOutputs = compiled.nodeOutputs(addIndex);
            int[] softmaxInputs = compiled.nodeInputs(softmaxIndex);
            int[] softmaxOutputs = compiled.nodeOutputs(softmaxIndex);
            if (projectionInputs.length != 2 || projectionOutputs.length != 1 ||
                    addInputs.length != 2 || addOutputs.length != 1 ||
                    softmaxInputs.length != 1 || softmaxOutputs.length != 1 ||
                    softmaxInputs[0] != addOutputs[0] ||
                    model.getGraphOutputs().get(0) != softmaxOutputs[0]) return null;
            int projectionOutput = projectionOutputs[0];
            int biasTensor;
            if (addInputs[0] == projectionOutput) biasTensor = addInputs[1];
            else if (addInputs[1] == projectionOutput) biasTensor = addInputs[0];
            else return null;

            List<TensorInfo> tensors = model.getTensors();
            int activationTensor = projectionInputs[0];
            int weightTensor = projectionInputs[1];
            if (!tensors.get(weightTensor).isConstant() || !tensors.get(biasTensor).isConstant()) return null;
            if (compiled.tensorConsumerCount(projectionOutput) != 1 ||
                    compiled.tensorLastUse(projectionOutput) != addIndex ||
                    compiled.tensorConsumerCount(addOutputs[0]) != 1 ||
                    compiled.tensorLastUse(addOutputs[0]) != softmaxIndex ||
                    compiled.tensorConsumerCount(softmaxOutputs[0]) != 0 ||
                    compiled.tensorLastUse(softmaxOutputs[0]) != compiled.nodeCount()) return null;

            List<TensorShape> shapes = ShapeResolver.resolve(model, inputShapes);
            TensorShape activation = shapes.get(activationTensor);
            TensorShape weights = shapes.get(weightTensor);
            TensorShape bias = shapes.get(biasTensor);
            TensorShape projected = shapes.get(projectionOutput);
            TensorShape probabilities = shapes.get(softmaxOutputs[0]);
            if ((activation.getRank() != 2 && activation.getRank() != 3) ||
                    (activation.getRank() == 3 && activation.get(0) != 1) ||
                    weights.getRank() != 2 || bias.getRank() != 1 ||
                    !projected.equals(shapes.get(addOutputs[0])) ||
                    !projected.equals(probabilities)) return null;
            int inner = activation.get(activation.getRank() - 1);
            int rows = activation.get(activation.getRank() - 2);
            int columns = weights.get(1);
            if (weights.get(0) != inner || bias.get(0) != columns ||
                    columns != classCount || projected.get(projected.getRank() - 2) != rows ||
                    projected.get(projected.getRank() - 1) != columns) return null;
            ByteBuffer parameters = compiled.parameterData(softmaxIndex);
            int axis = parameters.getInt(4);
            if (axis < 0) axis += projected.getRank();
            if (axis != projected.getRank() - 1) return null;
            return new Tail(projectionIndex, activationTensor, weightTensor, biasTensor,
                    rows, inner, columns);
        }
    }
}
