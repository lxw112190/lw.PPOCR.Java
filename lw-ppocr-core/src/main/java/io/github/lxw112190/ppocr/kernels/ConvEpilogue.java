package io.github.lxw112190.ppocr.kernels;

/** Immutable channel-wise post-convolution expression; no weight folding or FMA. */
public final class ConvEpilogue {
    public static final int NONE = 0, RELU = 1, GELU = 2, HARD_SWISH = 3, HARD_SIGMOID = 4;
    public final float[] mean, factor, normalizationBias, postBias;
    public final boolean scalarPostBias;
    public final int activation;
    public final float divisor, addend, multiplier, alpha, beta;

    public ConvEpilogue(float[] mean, float[] factor, float[] normalizationBias,
                        float[] postBias, boolean scalarPostBias, int activation,
                        float divisor, float addend, float multiplier, float alpha, float beta) {
        this.mean = mean; this.factor = factor; this.normalizationBias = normalizationBias;
        this.postBias = postBias; this.scalarPostBias = scalarPostBias;
        this.activation = activation; this.divisor = divisor; this.addend = addend;
        this.multiplier = multiplier; this.alpha = alpha; this.beta = beta;
    }

    public float apply(float value, int channel, float residual, boolean addResidual) {
        value = linear(value, channel, residual, addResidual);
        switch (activation) {
            case RELU: return Math.max(0.0f, value);
            case GELU: return ((erf(value / divisor) + addend) * value) * multiplier;
            case HARD_SWISH: return Math.min(1.0f, Math.max(0.0f, value * alpha + beta)) * value;
            case HARD_SIGMOID: return Math.min(1.0f, Math.max(0.0f, value * alpha + beta));
            default: return value;
        }
    }

    public float linear(float value, int channel, float residual, boolean addResidual) {
        if (factor != null) value = (value - mean[channel]) * factor[channel] + normalizationBias[channel];
        if (postBias != null) value += postBias[scalarPostBias ? 0 : channel];
        if (addResidual) value += residual;
        return value;
    }

    private static float erf(float input) {
        double value = input, sign = value < 0 ? -1.0 : 1.0;
        value = Math.abs(value);
        double t = 1.0 / (1.0 + 0.3275911 * value);
        double polynomial = (((((1.061405429 * t - 1.453152027) * t)
                + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t;
        return (float) (sign * (1.0 - polynomial * Math.exp(-value * value)));
    }
}
