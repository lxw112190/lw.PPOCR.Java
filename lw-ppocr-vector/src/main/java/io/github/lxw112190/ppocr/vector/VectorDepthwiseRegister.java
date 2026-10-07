package io.github.lxw112190.ppocr.vector;

import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorSpecies;

/** One tile stays in a register across all depthwise taps; no output RMW. */
final class VectorDepthwiseRegister {
    private static final VectorSpecies<Float> S = VectorSupport.F32;
    private static final int L = S.length();

    private VectorDepthwiseRegister() { }

    static void apply(float[] a, int ao, float[] w, int wo, float[] b, int bo,
                      float[] c, int co, int batch, int channels, int h, int width,
                      int kernel, int sh, int oh) {
        int pad = kernel / 2, plane = h * width, outPlane = oh * width;
        int left = Math.min(pad, width), right = Math.max(left, width - pad);
        for (int n = 0; n < batch; n++) for (int ch = 0; ch < channels; ch++) {
            int in = ao + (n * channels + ch) * plane;
            int out = co + (n * channels + ch) * outPlane;
            int weight = wo + ch * kernel * kernel;
            float initial = b == null ? 0 : b[bo + ch];
            for (int y = 0; y < oh; y++) {
                int iy = y * sh - pad;
                int ky0 = Math.max(0, -iy), ky1 = Math.min(kernel, h - iy);
                int x = left;
                for (; x + L * 2 <= right; x += L * 2)
                    pair(a, in, w, weight, c, out + y * width + x, initial,
                            width, kernel, iy, x - pad, ky0, ky1);
                for (; x + L <= right; x += L)
                    tile(a, in, w, weight, c, out + y * width + x, initial,
                            width, kernel, iy, x - pad, ky0, ky1);
                edge(a, in, w, weight, c, out + y * width, initial,
                        width, kernel, iy, pad, ky0, ky1, 0, left);
                edge(a, in, w, weight, c, out + y * width, initial,
                        width, kernel, iy, pad, ky0, ky1, x, width);
            }
        }
    }

    private static void pair(float[] a, int ao, float[] w, int wo, float[] c, int co,
                             float initial, int width, int kernel, int iy, int ix,
                             int ky0, int ky1) {
        FloatVector s0 = FloatVector.broadcast(S, initial), s1 = s0;
        for (int ky = ky0; ky < ky1; ky++) {
            int source = ao + (iy + ky) * width + ix, weight = wo + ky * kernel;
            for (int kx = 0; kx < kernel; kx++) {
                FloatVector weightVector = FloatVector.broadcast(S, w[weight + kx]);
                s0 = s0.add(FloatVector.fromArray(S, a, source + kx).mul(weightVector));
                s1 = s1.add(FloatVector.fromArray(S, a, source + kx + L).mul(weightVector));
            }
        }
        s0.intoArray(c, co); s1.intoArray(c, co + L);
    }

    private static void tile(float[] a, int ao, float[] w, int wo, float[] c, int co,
                             float initial, int width, int kernel, int iy, int ix,
                             int ky0, int ky1) {
        FloatVector sum = FloatVector.broadcast(S, initial);
        for (int ky = ky0; ky < ky1; ky++) {
            int source = ao + (iy + ky) * width + ix, weight = wo + ky * kernel;
            for (int kx = 0; kx < kernel; kx++)
                sum = sum.add(FloatVector.fromArray(S, a, source + kx).mul(w[weight + kx]));
        }
        sum.intoArray(c, co);
    }

    private static void edge(float[] a, int ao, float[] w, int wo, float[] c, int co,
                             float initial, int width, int kernel, int iy, int pad,
                             int ky0, int ky1, int start, int end) {
        for (int x = start; x < end; x++) {
            float sum = initial;
            int kx0 = Math.max(0, pad - x), kx1 = Math.min(kernel, width + pad - x);
            for (int ky = ky0; ky < ky1; ky++) {
                int source = ao + (iy + ky) * width + x - pad, weight = wo + ky * kernel;
                for (int kx = kx0; kx < kx1; kx++) sum += a[source + kx] * w[weight + kx];
            }
            c[co + x] = sum;
        }
    }
}
