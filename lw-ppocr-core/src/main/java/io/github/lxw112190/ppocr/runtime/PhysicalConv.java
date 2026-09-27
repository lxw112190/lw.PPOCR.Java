package io.github.lxw112190.ppocr.runtime;

import io.github.lxw112190.ppocr.model.OperatorType;
import io.github.lxw112190.ppocr.kernels.KernelBackend;
import io.github.lxw112190.ppocr.kernels.PreparedConvBackend;
import java.nio.ByteBuffer;

/** Fully bound Conv instruction: run does not resolve shapes or decode parameters. */
final class PhysicalConv {
    final int last;
    final ConvFusionPlan fusion;
    private final float[] input, weights, bias, storage, residual;
    private final int inputOffset, weightOffset, biasOffset, outputOffset, residualOffset;
    private final int[] p;
    private final PreparedConvBackend.Kernel prepared;

    private PhysicalConv(PreparedExecution execution, PreparedNode node, float[] storage,
                         ConvFusionPlan fusion, int[] parameters, KernelBackend backend) {
        int[] inputs = node.inputs;
        this.storage = storage; this.fusion = fusion;
        this.last = fusion == null ? node.index : fusion.last;
        this.input = binding(execution, inputs[0], storage);
        this.weights = binding(execution, inputs[1], storage);
        this.bias = inputs.length == 3 ? binding(execution, inputs[2], storage) : null;
        this.inputOffset = execution.offset(inputs[0]); this.weightOffset = execution.offset(inputs[1]);
        this.biasOffset = inputs.length == 3 ? execution.offset(inputs[2]) : 0;
        this.outputOffset = execution.offset(fusion == null ? node.outputs[0] : fusion.output);
        this.residual = fusion != null && fusion.residual >= 0 ? binding(execution, fusion.residual, storage) : null;
        this.residualOffset = residual == null ? 0 : execution.offset(fusion.residual);
        this.p = parameters;
        this.prepared = backend instanceof PreparedConvBackend
                && execution.model().getTensors().get(inputs[1]).isConstant()
                ? ((PreparedConvBackend) backend).prepareConv(weights, weightOffset, parameters) : null;
    }

    static PhysicalConv prepare(PreparedExecution execution, PreparedNode node, float[] storage,
                                KernelBackend backend) {
        if (node.getOperator() != OperatorType.CONV || node.inputs.length < 2
                || node.inputs.length > 3 || node.outputs.length != 1) return null;
        TensorShape a = execution.shapes().get(node.inputs[0]);
        TensorShape w = execution.shapes().get(node.inputs[1]);
        TensorShape c = execution.shapes().get(node.outputs[0]);
        if (a.getRank() != 4 || w.getRank() != 4 || c.getRank() != 4) return null;
        ByteBuffer params = node.parameters;
        int groups = params.getInt(4), kh = params.getInt(8), kw = params.getInt(12);
        if (groups <= 0 || a.get(1) % groups != 0 || w.get(0) % groups != 0
                || w.get(1) != a.get(1) / groups || w.get(2) != kh || w.get(3) != kw
                || c.get(0) != a.get(0) || c.get(1) != w.get(0)) return null;
        int[] p = {a.get(0), a.get(1), a.get(2), a.get(3), w.get(0), kh, kw,
                params.getInt(16), params.getInt(20), params.getInt(24), params.getInt(28),
                params.getInt(32), params.getInt(36), params.getInt(40), params.getInt(44),
                groups, c.get(2), c.get(3)};
        return new PhysicalConv(execution, node, storage, execution.convFusionPlans()[node.index], p, backend);
    }

    int scratchFloats() { return prepared == null ? 0 : prepared.scratchFloats(); }
    boolean usesSpatialPanel() { return prepared != null; }

    void run(ParallelKernels kernels) {
        if (prepared != null) {
            kernels.preparedConv(prepared, input, inputOffset, bias, biasOffset, storage, outputOffset,
                    p[0], p[4], p[16] * p[17], fusion == null ? null : fusion.epilogue,
                    residual, residualOffset);
            return;
        }
        kernels.conv(input, inputOffset, weights, weightOffset, bias, biasOffset, storage, outputOffset,
                p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8], p[9], p[10], p[11], p[12],
                p[13], p[14], p[15], p[16], p[17], fusion == null ? null : fusion.epilogue,
                residual, residualOffset);
    }

    private static float[] binding(PreparedExecution execution, int tensor, float[] storage) {
        return execution.model().getTensors().get(tensor).isConstant() ? execution.constant(tensor) : storage;
    }
}
