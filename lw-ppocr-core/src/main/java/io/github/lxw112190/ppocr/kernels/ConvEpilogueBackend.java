package io.github.lxw112190.ppocr.kernels;

/** Optional fused convolution capability. Residual/output must not overlap inputs. */
public interface ConvEpilogueBackend {
    /** Keeps flat activation vector/scalar boundaries unchanged across channel shards. */
    default int convEpilogueAlignment() { return 1; }
    void convEpilogue(float[] input, int inputOffset, float[] weights, int weightOffset,
                      float[] bias, int biasOffset, float[] output, int outputOffset,
                      int batch, int channels, int height, int width, int outputChannels,
                      int kh, int kw, int sh, int sw, int dh, int dw,
                      int pt, int pl, int pb, int pr, int groups, int oh, int ow,
                      ConvEpilogue epilogue, int channelBase, float[] residual, int residualOffset);
}
