package io.github.lxw112190.ppocr.kernels;

/** Optional model-weight packing and session-bound spatial convolution capability. */
public interface PreparedConvBackend {
    interface Kernel {
        int scratchFloats();
        /** Independent kernel-defined work units; may be spatial panels rather than image rows. */
        int outputRows();
        long operations();
        void runRows(float[] input, int inputOffset, float[] bias, int biasOffset,
                     float[] output, int outputOffset, float[] scratch, int firstRow, int endRow);
    }

    /** Parameters have the same order as KernelBackend.conv, starting at batch. */
    Kernel prepareConv(float[] weights, int weightOffset, int[] parameters);

    void finishPreparedConv(float[] output, int offset, int batch, int channels, int plane,
                            ConvEpilogue epilogue, float[] residual, int residualOffset);

    /** Unique packed arrays owned by this backend, excluding session scratch. */
    long preparedConvWeightBytes();
}
