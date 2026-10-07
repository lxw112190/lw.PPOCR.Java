package io.github.lxw112190.ppocr.vector;

import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorSpecies;

/** Register-resident spatial tiles for wide, same-size NCHW convolutions. */
final class VectorWideConv {
    private static final VectorSpecies<Float> S = VectorSupport.F32;
    private static final int L = S.length();

    private VectorWideConv() { }

    static void apply(float[] a, int ao, float[] w, int wo, float[] b, int bo,
                      float[] c, int co, int batch, int ic, int h, int width, int oc,
                      int kh, int kw, int pt, int pl) {
        int plane = h * width, inner = ic * kh * kw;
        int left = Math.min(pl, width), right = Math.max(left, width - (kw - 1 - pl));
        for (int n = 0; n < batch; n++) for (int o = 0; o < oc; o += 4) {
            int count = Math.min(4, oc - o);
            int out = co + (n * oc + o) * plane, in = ao + n * ic * plane;
            for (int y = 0; y < h; y++) {
                int ky0 = Math.max(0, pt - y), ky1 = Math.min(kh, h + pt - y);
                int x = left;
                // Four output channels share each input load and write sums only once.
                for (; x + L <= right; x += L) {
                    if (count == 4) four(a, in, w, wo + o * inner, b, bo + o,
                            c, out + y * width + x, ic, h, width, inner, plane,
                            kh, kw, y - pt, x - pl, ky0, ky1);
                    else for (int j = 0; j < count; j++) one(a, in, w, wo + (o + j) * inner,
                            b, bo + o + j, c, out + j * plane + y * width + x,
                            ic, h, width, kh, kw, y - pt, x - pl, ky0, ky1);
                }
                for (int j = 0; j < count; j++) {
                    edge(a, in, w, wo + (o + j) * inner, b, bo + o + j, c,
                            out + j * plane + y * width, ic, h, width, kh, kw, pt, pl,
                            y, ky0, ky1, 0, left);
                    edge(a, in, w, wo + (o + j) * inner, b, bo + o + j, c,
                            out + j * plane + y * width, ic, h, width, kh, kw, pt, pl,
                            y, ky0, ky1, x, width);
                }
            }
        }
    }

    private static void four(float[] a, int ao, float[] w, int wo, float[] b, int bo,
                             float[] c, int co, int channels, int h, int width, int inner,
                             int plane, int kh, int kw, int y, int x, int ky0, int ky1) {
        FloatVector s0 = FloatVector.broadcast(S, b == null ? 0 : b[bo]);
        FloatVector s1 = FloatVector.broadcast(S, b == null ? 0 : b[bo + 1]);
        FloatVector s2 = FloatVector.broadcast(S, b == null ? 0 : b[bo + 2]);
        FloatVector s3 = FloatVector.broadcast(S, b == null ? 0 : b[bo + 3]);
        for (int i = 0; i < channels; i++) for (int ky = ky0; ky < ky1; ky++) {
            int source = ao + (i * h + y + ky) * width + x;
            int weight = wo + (i * kh + ky) * kw;
            for (int kx = 0; kx < kw; kx++) {
                FloatVector v = FloatVector.fromArray(S, a, source + kx);
                s0 = s0.add(v.mul(w[weight + kx]));
                s1 = s1.add(v.mul(w[weight + inner + kx]));
                s2 = s2.add(v.mul(w[weight + inner * 2 + kx]));
                s3 = s3.add(v.mul(w[weight + inner * 3 + kx]));
            }
        }
        s0.intoArray(c, co); s1.intoArray(c, co + plane);
        s2.intoArray(c, co + plane * 2); s3.intoArray(c, co + plane * 3);
    }

    private static void one(float[] a, int ao, float[] w, int wo, float[] b, int bo,
                            float[] c, int co, int channels, int h, int width, int kh, int kw,
                            int y, int x, int ky0, int ky1) {
        FloatVector sum = FloatVector.broadcast(S, b == null ? 0 : b[bo]);
        for (int i = 0; i < channels; i++) for (int ky = ky0; ky < ky1; ky++) {
            int source = ao + (i * h + y + ky) * width + x;
            int weight = wo + (i * kh + ky) * kw;
            for (int kx = 0; kx < kw; kx++)
                sum = sum.add(FloatVector.fromArray(S, a, source + kx).mul(w[weight + kx]));
        }
        sum.intoArray(c, co);
    }

    private static void edge(float[] a, int ao, float[] w, int wo, float[] b, int bo,
                             float[] c, int co, int channels, int h, int width,
                             int kh, int kw, int pt, int pl, int y, int ky0, int ky1,
                             int start, int end) {
        for (int x = start; x < end; x++) {
            float sum = b == null ? 0 : b[bo];
            int kx0 = Math.max(0, pl - x), kx1 = Math.min(kw, width + pl - x);
            for (int i = 0; i < channels; i++) for (int ky = ky0; ky < ky1; ky++) {
                int source = ao + (i * h + y - pt + ky) * width + x - pl;
                int weight = wo + (i * kh + ky) * kw;
                for (int kx = kx0; kx < kx1; kx++) sum += a[source + kx] * w[weight + kx];
            }
            c[co + x] = sum;
        }
    }
}
