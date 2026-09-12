package io.github.lxw112190.ppocr.kernels;

/** Optional backend capability for an exactly matched five-node GELU expression. */
public interface FusedGeluBackend {
    void gelu(float[] input, int inputOffset, float[] output, int outputOffset,
              int length, float divisor, float addend, float multiplier);
}
