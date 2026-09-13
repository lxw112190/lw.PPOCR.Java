package io.github.lxw112190.ppocr.kernels;

/** Optional backend capability for a REC projection followed by greedy selection. */
public interface ProjectionArgMaxBackend {
    boolean supportsProjectionArgMax(int rows, int inner, int columns);

    /**
     * Computes {@code activations * weights + bias}, then the row-wise argmax and
     * the winning softmax probability without retaining the dense output matrix.
     * The scratch buffer must hold up to four class rows.
     */
    void projectionArgMax(float[] activations, int activationOffset,
                          float[] weights, int weightOffset,
                          float[] bias, int biasOffset,
                          int rows, int inner, int columns,
                          int[] bestIndices, float[] bestLogits,
                          float[] bestProbabilities, float[] rowScratch);
}
