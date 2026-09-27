package io.github.lxw112190.ppocr.vector;

import io.github.lxw112190.ppocr.kernels.ConvEpilogue;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorSpecies;

/** Allocation-free post-ops over the physical instruction's final NCHW allocation. */
final class VectorConvEpilogue {
    private static final VectorSpecies<Float> SPECIES = VectorSupport.F32;
    private VectorConvEpilogue() { }

    static void finish(float[] output, int offset, int batch, int channels, int plane,
                        ConvEpilogue ep, int channelBase, float[] residual, int residualOffset,
                        VectorBackend backend) {
        // Keep optional channel transforms scalar: mixing all activation kinds in
        // one Vector loop prevented escape elimination on JDK 25 during warmup.
        if (ep.factor != null || ep.postBias != null) {
            for (int n = 0; n < batch; n++) for (int c = 0; c < channels; c++) {
                int base = offset + (n * channels + c) * plane, ec = channelBase + c;
                if (ep.factor != null) {
                    float mean = ep.mean[ec], factor = ep.factor[ec], bias = ep.normalizationBias[ec];
                    for (int i = 0; i < plane; i++) output[base + i] = (output[base + i] - mean) * factor + bias;
                }
                if (ep.postBias != null) {
                    float value = ep.postBias[ep.scalarPostBias ? 0 : ec];
                    for (int i = 0; i < plane; i++) output[base + i] += value;
                }
            }
        }
        int length = batch * channels * plane;
        if (residual != null) backend.add(output, offset, residual, residualOffset, output, offset, length);
        switch (ep.activation) {
            case ConvEpilogue.RELU: backend.relu(output, offset, output, offset, length); break;
            case ConvEpilogue.GELU:
                backend.gelu(output, offset, output, offset, length, ep.divisor, ep.addend, ep.multiplier); break;
            case ConvEpilogue.HARD_SIGMOID:
                backend.hardSigmoid(output, offset, output, offset, length, ep.alpha, ep.beta); break;
            case ConvEpilogue.HARD_SWISH: hardSwish(output, offset, length, ep.alpha, ep.beta); break;
            default: break;
        }
    }

    private static void hardSwish(float[] output, int offset, int length, float alpha, float beta) {
        int bound = SPECIES.loopBound(length), i = 0;
        for (; i < bound; i += SPECIES.length()) {
            FloatVector value = FloatVector.fromArray(SPECIES, output, offset + i);
            value.mul(alpha).add(beta).max(0.0f).min(1.0f).mul(value).intoArray(output, offset + i);
        }
        for (; i < length; i++) {
            float value = output[offset + i];
            output[offset + i] = Math.min(1.0f, Math.max(0.0f, value * alpha + beta)) * value;
        }
    }
}
