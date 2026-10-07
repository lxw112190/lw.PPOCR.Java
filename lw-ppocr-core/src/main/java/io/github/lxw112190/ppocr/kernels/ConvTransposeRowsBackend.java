package io.github.lxw112190.ppocr.kernels;

/** Optional disjoint batch/output-channel shards; weights remain IC-major.
 * Parameters use the same 18-field layout as PreparedConvBackend (bottom/right pads unused).
 * The parameter array is read-only, session-owned and valid until all shards join.
 */
public interface ConvTransposeRowsBackend {
    boolean supportsConvTransposeRows(int[] parameters);
    void convTransposeRows(float[] input, int inputOffset, float[] weights, int weightOffset,
                           float[] bias, int biasOffset, float[] output, int outputOffset,
                           int[] parameters, int firstRow, int endRow);
}
