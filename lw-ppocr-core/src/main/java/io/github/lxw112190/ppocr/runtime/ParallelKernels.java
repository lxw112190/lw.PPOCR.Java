package io.github.lxw112190.ppocr.runtime;

import io.github.lxw112190.ppocr.kernels.KernelBackend;
import io.github.lxw112190.ppocr.kernels.ConvEpilogue;
import io.github.lxw112190.ppocr.kernels.ConvEpilogueBackend;
import io.github.lxw112190.ppocr.kernels.PreparedConvBackend;

/** Disjoint NCHW channel / matrix row shards, preserving reduction order. */
final class ParallelKernels implements OperatorParallelExecutor.Action {
    private final KernelBackend backend;
    private final OperatorParallelExecutor executor = new OperatorParallelExecutor();
    private int parallelism = 1;
    private float[] input, weights, bias, output;
    private int inputOffset, weightOffset, biasOffset, outputOffset;
    private final int[] p = new int[18];
    private boolean matrix;
    private ConvEpilogue epilogue;
    private float[] residual;
    private int residualOffset;
    private PreparedConvBackend.Kernel prepared;
    private float[][] spatialScratch;
    private int spatialScratchSize;
    ParallelKernels(KernelBackend backend) { this.backend = backend; }
    void setParallelism(int count) {
        if (count <= 0) throw new IllegalArgumentException("operator parallelism must be positive");
        parallelism = Boolean.getBoolean("lwppocr.disableIntraOp") ? 1
                : Math.min(count, Math.min(8, Runtime.getRuntime().availableProcessors()));
        ensureSpatialScratch();
    }

    void prepareSpatialScratch(int size) { spatialScratchSize = size; ensureSpatialScratch(); }

    private void ensureSpatialScratch() {
        if (spatialScratchSize == 0) return;
        if (spatialScratch == null) spatialScratch = new float[Math.min(8, Runtime.getRuntime().availableProcessors())][];
        for (int i = 0; i < parallelism; i++) if (spatialScratch[i] == null) {
            spatialScratch[i] = new float[spatialScratchSize];
        }
    }

    long spatialScratchBytes() {
        long size = 0;
        if (spatialScratch != null) for (float[] array : spatialScratch) if (array != null) size += (long) array.length * 4;
        return size;
    }

    void preparedConv(PreparedConvBackend.Kernel kernel, float[] a, int ao, float[] b, int bo,
                      float[] c, int co, int batch, int channels, int plane,
                      ConvEpilogue epilogue, float[] residual, int residualOffset) {
        int count = kernel.operations() < 2000000L ? 1 : Math.min(parallelism, kernel.outputRows());
        if (count == 1) {
            kernel.runRows(a, ao, b, bo, c, co, spatialScratch[0], 0, kernel.outputRows());
        } else {
            bind(a, ao, null, 0, b, bo, c, co); prepared = kernel;
            try { executor.run(count, this); } finally { prepared = null; clear(); }
        }
        ((PreparedConvBackend) backend).finishPreparedConv(c, co, batch, channels, plane,
                epilogue, residual, residualOffset);
    }
    void matMul(float[] a, int ao, float[] b, int bo, float[] c, int co,
                int rows, int inner, int columns) {
        int count = Math.min(parallelism, rows / 4);
        if (count <= 1 || (long) rows * inner * columns < 2000000L) {
            backend.matMul(a, ao, b, bo, c, co, rows, inner, columns); return;
        }
        bind(a, ao, b, bo, null, 0, c, co);
        matrix = true; p[0] = rows; p[1] = inner; p[2] = columns;
        try { executor.run(count, this); } finally { clear(); }
    }
    void conv(float[] a, int ao, float[] w, int wo, float[] b, int bo, float[] c, int co,
              int batch, int channels, int height, int width, int outChannels,
              int kh, int kw, int sh, int sw, int dh, int dw, int pt, int pl,
              int pb, int pr, int groups, int oh, int ow) {
        conv(a, ao, w, wo, b, bo, c, co, batch, channels, height, width, outChannels,
                kh, kw, sh, sw, dh, dw, pt, pl, pb, pr, groups, oh, ow, null, null, 0);
    }

    void conv(float[] a, int ao, float[] w, int wo, float[] b, int bo, float[] c, int co,
              int batch, int channels, int height, int width, int outChannels,
              int kh, int kw, int sh, int sw, int dh, int dw, int pt, int pl,
              int pb, int pr, int groups, int oh, int ow,
              ConvEpilogue epilogue, float[] residual, int residualOffset) {
        boolean depthwise = groups == channels && outChannels == channels;
        // Stride-two dense kernels require >=16 output channels. Do not turn a
        // specialized 16-channel kernel into an allocating generic 8-channel path.
        int grain = !depthwise && sh == 2 && sw == 2 ? 16 : 8;
        if (epilogue != null && epilogue.activation == ConvEpilogue.GELU) {
            grain = Math.max(grain, ((ConvEpilogueBackend) backend).convEpilogueAlignment());
        }
        int count = Math.min(parallelism, outChannels / grain);
        long work = (long) outChannels * oh * ow * (channels / groups) * kh * kw;
        if (count <= 1 || batch != 1 || (!depthwise && groups != 1) || work < 2000000L) {
            if (epilogue == null) {
                backend.conv(a, ao, w, wo, b, bo, c, co, batch, channels, height, width,
                        outChannels, kh, kw, sh, sw, dh, dw, pt, pl, pb, pr, groups, oh, ow);
            } else {
                ((ConvEpilogueBackend) backend).convEpilogue(a, ao, w, wo, b, bo, c, co,
                        batch, channels, height, width, outChannels, kh, kw, sh, sw, dh, dw,
                        pt, pl, pb, pr, groups, oh, ow, epilogue, 0, residual, residualOffset);
            }
            return;
        }
        bind(a, ao, w, wo, b, bo, c, co); matrix = false;
        this.epilogue = epilogue; this.residual = residual; this.residualOffset = residualOffset;
        p[0] = channels; p[1] = height; p[2] = width; p[3] = outChannels;
        p[4] = kh; p[5] = kw; p[6] = sh; p[7] = sw; p[8] = dh; p[9] = dw;
        p[10] = pt; p[11] = pl; p[12] = pb; p[13] = pr; p[14] = groups;
        p[15] = oh; p[16] = ow;
        p[17] = grain;
        try { executor.run(count, this); } finally { clear(); }
    }
    public void run(int shard, int shards) {
        if (prepared != null) {
            int rows = prepared.outputRows();
            prepared.runRows(input, inputOffset, bias, biasOffset, output, outputOffset,
                    spatialScratch[shard], rows * shard / shards, rows * (shard + 1) / shards);
        } else if (matrix) {
            int blocks = p[0] / 4, first = blocks * shard / shards * 4;
            int end = shard == shards - 1 ? p[0] : blocks * (shard + 1) / shards * 4;
            backend.matMul(input, inputOffset + first * p[1], weights, weightOffset,
                    output, outputOffset + first * p[2], end - first, p[1], p[2]);
        } else {
            int blocks = p[3] / p[17], first = blocks * shard / shards * p[17];
            int end = shard == shards - 1 ? p[3] : blocks * (shard + 1) / shards * p[17];
            boolean depthwise = p[14] != 1;
            int channels = depthwise ? end - first : p[0];
            int weightStride = (depthwise ? 1 : p[0]) * p[4] * p[5];
            if (epilogue == null) {
                backend.conv(input, inputOffset + (depthwise ? first * p[1] * p[2] : 0),
                    weights, weightOffset + first * weightStride, bias, biasOffset + first,
                    output, outputOffset + first * p[15] * p[16], 1, channels, p[1], p[2],
                    end - first, p[4], p[5], p[6], p[7], p[8], p[9], p[10], p[11], p[12], p[13],
                    depthwise ? channels : 1, p[15], p[16]);
            } else {
                ((ConvEpilogueBackend) backend).convEpilogue(input,
                        inputOffset + (depthwise ? first * p[1] * p[2] : 0),
                        weights, weightOffset + first * weightStride, bias, biasOffset + first,
                        output, outputOffset + first * p[15] * p[16], 1, channels, p[1], p[2],
                        end - first, p[4], p[5], p[6], p[7], p[8], p[9], p[10], p[11], p[12], p[13],
                        depthwise ? channels : 1, p[15], p[16], epilogue, first, residual,
                        residualOffset + first * p[15] * p[16]);
            }
        }
    }
    private void bind(float[] a, int ao, float[] w, int wo, float[] b, int bo, float[] c, int co) {
        input = a; inputOffset = ao; weights = w; weightOffset = wo;
        bias = b; biasOffset = bo; output = c; outputOffset = co;
    }
    private void clear() { input = weights = bias = output = residual = null; epilogue = null; }
}
