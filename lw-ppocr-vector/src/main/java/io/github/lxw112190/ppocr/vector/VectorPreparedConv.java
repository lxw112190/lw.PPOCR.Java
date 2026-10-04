package io.github.lxw112190.ppocr.vector;

import io.github.lxw112190.ppocr.kernels.PreparedConvBackend;
import java.util.HashMap;
import java.util.Map;
import java.util.WeakHashMap;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorSpecies;

/** Pixel-panel input packing with output-channel-contiguous model weights. */
final class VectorPreparedConv implements PreparedConvBackend.Kernel {
    private static final VectorSpecies<Float> SPECIES = VectorSupport.F32;
    private static final int LANES = SPECIES.length();
    private static final int PIXELS = 4;
    private final int channels, height, width, outChannels, kh, kw, sh, sw, dh, dw, pt, pl, oh, ow;
    private final int inner;
    private final float[] packed;
    private final float[] sourceWeights;
    private final int sourceOffset;

    /** Cache lifetime follows canonical model weights, not shape-specialized sessions. */
    static final class Weights {
        // Bound extra packing model-wide, not once per width/worker Session.
        private static final long LIMIT = 80L * 1024;
        private final Map<float[], Map<String, float[]>> cache =
                new WeakHashMap<float[], Map<String, float[]>>();

        synchronized float[] pack(float[] source, int offset, int channels, int outChannels, int taps) {
            String key = offset + ":" + channels + ":" + outChannels + ":" + taps;
            Map<String, float[]> entries = cache.get(source);
            if (entries == null) { entries = new HashMap<String, float[]>(); cache.put(source, entries); }
            float[] result = entries.get(key);
            if (result != null) return result;
            int inner = Math.multiplyExact(channels, taps);
            if ((long) inner * outChannels * 4 > LIMIT - bytes()) return null;
            result = new float[Math.multiplyExact(inner, outChannels)];
            packInto(source, offset, result, 0, inner, outChannels);
            entries.put(key, result);
            return result;
        }

        synchronized long bytes() {
            long size = 0;
            for (Map<String, float[]> entries : cache.values()) for (float[] array : entries.values()) {
                size += (long) array.length * 4;
            }
            return size;
        }
    }

    static VectorPreparedConv prepare(Weights cache, float[] weights, int offset, int[] p) {
        if (p[0] != 1 || p[15] != 1 || p[4] < 16 || p[4] % LANES != 0
                || p[5] * p[6] <= 1 || p[5] > 5 || p[6] > 5
                || p[7] <= 0 || p[8] <= 0 || p[9] <= 0 || p[10] <= 0
                || p[16] <= 0 || p[17] <= 0) return null;
        float[] packed = cache.pack(weights, offset, p[1], p[4], p[5] * p[6]);
        return new VectorPreparedConv(p, packed, packed == null ? weights : null, offset);
    }

    private VectorPreparedConv(int[] p, float[] packed, float[] sourceWeights, int sourceOffset) {
        channels = p[1]; height = p[2]; width = p[3]; outChannels = p[4];
        kh = p[5]; kw = p[6]; sh = p[7]; sw = p[8]; dh = p[9]; dw = p[10];
        pt = p[11]; pl = p[12]; oh = p[16]; ow = p[17];
        inner = channels * kh * kw; this.packed = packed;
        this.sourceWeights = sourceWeights; this.sourceOffset = sourceOffset;
    }

    public int scratchFloats() {
        int panel = Math.addExact(Math.multiplyExact(inner, PIXELS), PIXELS * LANES + kh * kw * PIXELS);
        return packed == null ? Math.addExact(panel, Math.multiplyExact(inner, outChannels)) : panel;
    }
    public int outputRows() { return oh; }
    public long operations() { return (long) oh * ow * outChannels * inner; }

    public void runRows(float[] input, int inputOffset, float[] bias, int biasOffset,
                        float[] output, int outputOffset, float[] scratch, int firstRow, int endRow) {
        int validity = inner * PIXELS + PIXELS * LANES;
        int plane = oh * ow;
        float[] weights = packed;
        int weightBase = 0;
        if (weights == null) {
            // Keep the same arithmetic path when the model cache budget is full.
            // Repack into session-reused scratch, never a per-call allocation.
            weights = scratch; weightBase = validity + kh * kw * PIXELS;
            packInto(sourceWeights, sourceOffset, weights, weightBase, inner, outChannels);
        }
        for (int y = firstRow; y < endRow; y++) {
            for (int x = 0; x < ow; x += PIXELS) {
                int count = Math.min(PIXELS, ow - x);
                int y0 = y * sh - pt, x0 = x * sw - pl;
                boolean interior = count == PIXELS && y0 >= 0 && x0 >= 0
                        && y0 + (kh - 1) * dh < height
                        && x0 + (PIXELS - 1) * sw + (kw - 1) * dw < width;
                // k follows original IC/KH/KW accumulation order, not NHWC tap/IC order.
                if (interior) {
                    int dst = 0;
                    for (int ic = 0; ic < channels; ic++) for (int ky = 0; ky < kh; ky++) {
                        int sourceRow = inputOffset + (ic * height + y0 + ky * dh) * width + x0;
                        for (int kx = 0; kx < kw; kx++) {
                            int source = sourceRow + kx * dw;
                            scratch[dst++] = input[source];
                            scratch[dst++] = input[source + sw];
                            scratch[dst++] = input[source + sw * 2];
                            scratch[dst++] = input[source + sw * 3];
                        }
                    }
                } else for (int ky = 0; ky < kh; ky++) for (int kx = 0; kx < kw; kx++) {
                    int iy = y * sh - pt + ky * dh;
                    int tap = ky * kw + kx;
                    for (int pixel = 0; pixel < PIXELS; pixel++) {
                        int ix = (x + pixel) * sw - pl + kx * dw;
                        boolean valid = pixel < count && iy >= 0 && iy < height && ix >= 0 && ix < width;
                        scratch[validity + tap * PIXELS + pixel] = valid ? 1.0f : 0.0f;
                        for (int ic = 0; ic < channels; ic++) {
                            scratch[(ic * kh * kw + tap) * PIXELS + pixel] = valid
                                    ? input[inputOffset + (ic * height + iy) * width + ix] : 0.0f;
                        }
                    }
                }
                for (int oc = 0; oc < outChannels; oc += LANES) {
                    if (interior) interiorTile(weights, weightBase + oc * inner, scratch, inner, bias, biasOffset + oc);
                    else microtile(weights, weightBase + oc * inner, scratch, inner, kh * kw,
                                bias, biasOffset + oc, count);
                    for (int pixel = 0; pixel < count; pixel++) for (int lane = 0; lane < LANES; lane++) {
                        output[outputOffset + (oc + lane) * plane + y * ow + x + pixel] =
                                scratch[inner * PIXELS + pixel * LANES + lane];
                    }
                }
            }
        }
    }

    private static void packInto(float[] source, int offset, float[] destination, int target,
                                 int inner, int outChannels) {
        for (int oc = 0; oc < outChannels; oc += LANES) for (int k = 0; k < inner; k++) {
            for (int lane = 0; lane < LANES; lane++) {
                destination[target + oc * inner + k * LANES + lane] = source[offset + (oc + lane) * inner + k];
            }
        }
    }

    private static void interiorTile(float[] weights, int weightOffset, float[] scratch,
                                      int inner, float[] bias, int biasOffset) {
        FloatVector initial = bias == null ? FloatVector.zero(SPECIES)
                : FloatVector.fromArray(SPECIES, bias, biasOffset);
        FloatVector s0 = initial, s1 = initial, s2 = initial, s3 = initial;
        for (int k = 0; k < inner; k++) {
            FloatVector w = FloatVector.fromArray(SPECIES, weights, weightOffset + k * LANES);
            int base = k * PIXELS;
            s0 = s0.add(w.mul(scratch[base]));
            s1 = s1.add(w.mul(scratch[base + 1]));
            s2 = s2.add(w.mul(scratch[base + 2]));
            s3 = s3.add(w.mul(scratch[base + 3]));
        }
        s0.intoArray(scratch, inner * PIXELS);
        s1.intoArray(scratch, inner * PIXELS + LANES);
        s2.intoArray(scratch, inner * PIXELS + LANES * 2);
        s3.intoArray(scratch, inner * PIXELS + LANES * 3);
    }

    /** Boundary pixels have independent, single-accumulator Vector loops. */
    private static void microtile(float[] weights, int weightOffset, float[] scratch,
                                  int inner, int taps, float[] bias, int biasOffset, int count) {
        int validity = inner * PIXELS + PIXELS * LANES;
        for (int pixel = 0; pixel < count; pixel++) {
            boundaryPixel(weights, weightOffset, scratch, inner, taps, bias, biasOffset,
                    validity, pixel);
        }
    }

    private static void boundaryPixel(float[] weights, int weightOffset, float[] scratch,
                                      int inner, int taps, float[] bias, int biasOffset,
                                      int validity, int pixel) {
        // Four separately conditional Vector accumulators caused persistent vector
        // boxing in the two-CPU JDK 25 full-OCR workload. Keep the array-only method
        // boundary and one accumulator, including for partial four-pixel tiles.
        FloatVector sum = bias == null ? FloatVector.zero(SPECIES)
                : FloatVector.fromArray(SPECIES, bias, biasOffset);
        for (int k = 0; k < inner; k++) {
            // Skip padding before loading weights, rather than multiplying by zero:
            // nonfinite weights and signed-zero arithmetic must match Scalar Conv.
            if (scratch[validity + (k % taps) * PIXELS + pixel] == 0) continue;
            FloatVector w = FloatVector.fromArray(SPECIES, weights, weightOffset + k * LANES);
            sum = sum.add(w.mul(scratch[k * PIXELS + pixel]));
        }
        sum.intoArray(scratch, inner * PIXELS + pixel * LANES);
    }
}
