package io.github.lxw112190.ppocr.vector;

import io.github.lxw112190.ppocr.kernels.PreparedConvBackend;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorSpecies;

/** Bounded spatial-panel NCHW 1x1 Conv for large Small/Medium channels. */
final class VectorLargePointwise implements PreparedConvBackend.Kernel {
    private static final VectorSpecies<Float> SPECIES = VectorSupport.F32;
    private static final int LANES = SPECIES.length();
    private static final int PANEL = 32;
    private final float[] weights;
    private final int weightOffset, channels, outputChannels, plane;
    private final boolean fma;
    private final boolean six = !Boolean.getBoolean("lwppocr.disableSixPointwise");


    static VectorLargePointwise prepare(float[] weights, int offset, int[] p) {
        // Wider coverage is exclusive to explicitly requested fused arithmetic.
        int minimum = Boolean.getBoolean("lwppocr.vectorFma")
                && !Boolean.getBoolean("lwppocr.disableExtendedFmaPointwise")
                ? Boolean.getBoolean("lwppocr.smallFmaPointwise") ? 32 : 64 : 256;
        if (p[0] != 1 || p[1] < minimum || p[4] < minimum || p[2] <= 0 || p[3] <= 0
                || p[5] != 1 || p[6] != 1 || p[9] <= 0 || p[10] <= 0
                || p[7] != 1 || p[8] != 1 || p[15] != 1
                || p[11] != 0 || p[12] != 0 || p[13] != 0 || p[14] != 0
                || p[2] != p[16] || p[3] != p[17]
                || (long) p[2] * p[3] < 64
                || (long) p[1] * PANEL > 262144) return null;
        return new VectorLargePointwise(weights, offset, p[1], p[4], p[2] * p[3]);
    }

    private VectorLargePointwise(float[] weights, int offset, int channels, int outputChannels, int plane) {
        this.weights = weights; this.weightOffset = offset;
        this.channels = channels; this.outputChannels = outputChannels; this.plane = plane;
        // Opt-in: fused rounding is not Scalar-bit-identical.
        this.fma = Boolean.getBoolean("lwppocr.vectorFma");
    }

    // Per-worker scratch is session-reused. Canonical weights are never copied.
    // Tiles include independent tails.
    public int scratchFloats() { return channels * PANEL; }
    public int outputRows() { return (plane + PANEL - 1) / PANEL; }
    public long operations() { return (long) channels * outputChannels * plane; }

    public void runRows(float[] input, int inputOffset, float[] bias, int biasOffset,
                        float[] output, int outputOffset, float[] scratch, int firstRow, int endRow) {
        for (int row = firstRow; row < endRow; row++) {
            int first = row * PANEL, pixels = Math.min(PANEL, plane - first);
            for (int k = 0; k < channels; k++)
                System.arraycopy(input, inputOffset + k * plane + first, scratch, k * PANEL, pixels);
            int oc = 0;
            if(fma && six) {
                for(;oc+5<outputChannels;oc+=6) {
                    int x=0;
                    for(;x+2*LANES<=pixels;x+=2*LANES)
                        sixPairFma(scratch,x,bias,biasOffset,output,outputOffset+first+x,oc);
                    if(x<pixels)for(int c=oc;c<oc+6;c++)oneFma(scratch,x,pixels,bias,biasOffset,output,outputOffset+first,c);
                }
            }
            for (; oc + 3 < outputChannels; oc += 4) {
                int x = 0;
                for (; x + 2 * LANES <= pixels; x += 2 * LANES) {
                    if (fma) fourPairFma(scratch, x, bias, biasOffset, output, outputOffset + first + x, oc);
                    else fourPair(scratch, x, bias, biasOffset, output, outputOffset + first + x, oc);
                }
                if (fma) {
                    for (int c = oc; x < pixels && c < oc + 4; c++)
                        oneFma(scratch, x, pixels, bias, biasOffset, output, outputOffset + first, c);
                    continue;
                }
                for (; x < pixels; x++)
                    fourScalar(scratch, x, bias, biasOffset, output, outputOffset + first + x, oc);
            }
            for (; oc < outputChannels; oc++) {
                if (fma) {
                    oneFma(scratch, 0, pixels, bias, biasOffset, output, outputOffset + first, oc);
                    continue;
                }
                int dest = outputOffset + oc * plane + first, w = weightOffset + oc * channels;
                int x = 0;
                for (; x + LANES <= pixels; x += LANES) {
                    FloatVector sum = FloatVector.broadcast(SPECIES, bias == null ? 0 : bias[biasOffset + oc]);
                    for (int k = 0; k < channels; k++) sum = sum.add(
                            FloatVector.fromArray(SPECIES, scratch, k * PANEL + x).mul(weights[w + k]));
                    sum.intoArray(output, dest + x);
                }
                for (; x < pixels; x++) {
                    float sum = bias == null ? 0 : bias[biasOffset + oc];
                    for (int k = 0; k < channels; k++) sum += scratch[k * PANEL + x] * weights[w + k];
                    output[dest + x] = sum;
                }
            }
        }
    }

    // Keep the vector values inside a compact, frequently invoked microkernel
    // so C2 compiles the arithmetic before a large whole-graph warmup completes.
    private void fourPair(float[] input, int x, float[] bias, int bo, float[] output, int dest, int oc) {
        dest += oc * plane;
        int w0 = weightOffset + oc * channels, w1 = w0 + channels, w2 = w1 + channels, w3 = w2 + channels;
        FloatVector s00 = FloatVector.broadcast(SPECIES, bias == null ? 0 : bias[bo + oc]), s01 = s00;
        FloatVector s10 = FloatVector.broadcast(SPECIES, bias == null ? 0 : bias[bo + oc + 1]), s11 = s10;
        FloatVector s20 = FloatVector.broadcast(SPECIES, bias == null ? 0 : bias[bo + oc + 2]), s21 = s20;
        FloatVector s30 = FloatVector.broadcast(SPECIES, bias == null ? 0 : bias[bo + oc + 3]), s31 = s30;
        for (int k = 0; k < channels; k++) {
            FloatVector a0 = FloatVector.fromArray(SPECIES, input, k * PANEL + x);
            FloatVector a1 = FloatVector.fromArray(SPECIES, input, k * PANEL + x + LANES);
            float w = weights[w0 + k]; s00 = s00.add(a0.mul(w)); s01 = s01.add(a1.mul(w));
            w = weights[w1 + k]; s10 = s10.add(a0.mul(w)); s11 = s11.add(a1.mul(w));
            w = weights[w2 + k]; s20 = s20.add(a0.mul(w)); s21 = s21.add(a1.mul(w));
            w = weights[w3 + k]; s30 = s30.add(a0.mul(w)); s31 = s31.add(a1.mul(w));
        }
        s00.intoArray(output, dest); s01.intoArray(output, dest + LANES);
        s10.intoArray(output, dest + plane); s11.intoArray(output, dest + plane + LANES);
        s20.intoArray(output, dest + 2 * plane); s21.intoArray(output, dest + 2 * plane + LANES);
        s30.intoArray(output, dest + 3 * plane); s31.intoArray(output, dest + 3 * plane + LANES);
    }

    private void fourPairFma(float[] input, int x, float[] bias, int bo, float[] output, int dest, int oc) {
        dest += oc * plane;
        int w0 = weightOffset + oc * channels, w1 = w0 + channels, w2 = w1 + channels, w3 = w2 + channels;
        FloatVector s00 = FloatVector.broadcast(SPECIES, bias == null ? 0 : bias[bo + oc]), s01 = s00;
        FloatVector s10 = FloatVector.broadcast(SPECIES, bias == null ? 0 : bias[bo + oc + 1]), s11 = s10;
        FloatVector s20 = FloatVector.broadcast(SPECIES, bias == null ? 0 : bias[bo + oc + 2]), s21 = s20;
        FloatVector s30 = FloatVector.broadcast(SPECIES, bias == null ? 0 : bias[bo + oc + 3]), s31 = s30;
        for (int k = 0; k < channels; k++) {
            FloatVector a0 = FloatVector.fromArray(SPECIES, input, k * PANEL + x);
            FloatVector a1 = FloatVector.fromArray(SPECIES, input, k * PANEL + x + LANES);
            FloatVector w = FloatVector.broadcast(SPECIES, weights[w0 + k]);
            s00 = a0.fma(w, s00); s01 = a1.fma(w, s01);
            w = FloatVector.broadcast(SPECIES, weights[w1 + k]);
            s10 = a0.fma(w, s10); s11 = a1.fma(w, s11);
            w = FloatVector.broadcast(SPECIES, weights[w2 + k]);
            s20 = a0.fma(w, s20); s21 = a1.fma(w, s21);
            w = FloatVector.broadcast(SPECIES, weights[w3 + k]);
            s30 = a0.fma(w, s30); s31 = a1.fma(w, s31);
        }
        s00.intoArray(output, dest); s01.intoArray(output, dest + LANES);
        s10.intoArray(output, dest + plane); s11.intoArray(output, dest + plane + LANES);
        s20.intoArray(output, dest + 2 * plane); s21.intoArray(output, dest + 2 * plane + LANES);
        s30.intoArray(output, dest + 3 * plane); s31.intoArray(output, dest + 3 * plane + LANES);
    }



    // Twelve accumulators share two input loads; keep this compact for C2 escape elimination.
    private void sixPairFma(float[] input,int x,float[] bias,int bo,float[] output,int dest,int oc) {
        dest+=oc*plane;
        FloatVector s00=FloatVector.broadcast(SPECIES,bias==null?0:bias[bo+oc+0]),s01=s00;
        FloatVector s10=FloatVector.broadcast(SPECIES,bias==null?0:bias[bo+oc+1]),s11=s10;
        FloatVector s20=FloatVector.broadcast(SPECIES,bias==null?0:bias[bo+oc+2]),s21=s20;
        FloatVector s30=FloatVector.broadcast(SPECIES,bias==null?0:bias[bo+oc+3]),s31=s30;
        FloatVector s40=FloatVector.broadcast(SPECIES,bias==null?0:bias[bo+oc+4]),s41=s40;
        FloatVector s50=FloatVector.broadcast(SPECIES,bias==null?0:bias[bo+oc+5]),s51=s50;
        for(int k=0;k<channels;k++) {
            FloatVector a0=FloatVector.fromArray(SPECIES,input,k*PANEL+x),a1=FloatVector.fromArray(SPECIES,input,k*PANEL+x+LANES);
            FloatVector w;
            w=FloatVector.broadcast(SPECIES,weights[weightOffset+(oc+0)*channels+k]);s00=a0.fma(w,s00);s01=a1.fma(w,s01);
            w=FloatVector.broadcast(SPECIES,weights[weightOffset+(oc+1)*channels+k]);s10=a0.fma(w,s10);s11=a1.fma(w,s11);
            w=FloatVector.broadcast(SPECIES,weights[weightOffset+(oc+2)*channels+k]);s20=a0.fma(w,s20);s21=a1.fma(w,s21);
            w=FloatVector.broadcast(SPECIES,weights[weightOffset+(oc+3)*channels+k]);s30=a0.fma(w,s30);s31=a1.fma(w,s31);
            w=FloatVector.broadcast(SPECIES,weights[weightOffset+(oc+4)*channels+k]);s40=a0.fma(w,s40);s41=a1.fma(w,s41);
            w=FloatVector.broadcast(SPECIES,weights[weightOffset+(oc+5)*channels+k]);s50=a0.fma(w,s50);s51=a1.fma(w,s51);
        }
        s00.intoArray(output,dest+0*plane);s01.intoArray(output,dest+0*plane+LANES);
        s10.intoArray(output,dest+1*plane);s11.intoArray(output,dest+1*plane+LANES);
        s20.intoArray(output,dest+2*plane);s21.intoArray(output,dest+2*plane+LANES);
        s30.intoArray(output,dest+3*plane);s31.intoArray(output,dest+3*plane+LANES);
        s40.intoArray(output,dest+4*plane);s41.intoArray(output,dest+4*plane+LANES);
        s50.intoArray(output,dest+5*plane);s51.intoArray(output,dest+5*plane+LANES);
    }


    private void oneFma(float[] input, int first, int pixels, float[] bias, int bo,
                        float[] output, int dest, int oc) {
        int w = weightOffset + oc * channels, x = first;
        dest += oc * plane;
        for (; x + LANES <= pixels; x += LANES) {
            oneVectorFma(input,x,bias==null?0:bias[bo+oc],output,dest+x,w);
        }
        for (; x < pixels; x++) {
            float sum = bias == null ? 0 : bias[bo + oc];
            for (int k = 0; k < channels; k++) sum = Math.fma(input[k * PANEL + x], weights[w + k], sum);
            output[dest + x] = sum;
        }
    }

    // Keep vector state out of the outer vector/scalar-tail loop and skip empty
    // work before entering it, so profiling is driven by actual tail reductions.
    private void oneVectorFma(float[] input,int x,float initial,float[] output,int dest,int weightBase) {
        FloatVector sum=FloatVector.broadcast(SPECIES,initial);
        for(int k=0;k<channels;k++)sum=FloatVector.fromArray(SPECIES,input,k*PANEL+x)
                .fma(FloatVector.broadcast(SPECIES,weights[weightBase+k]),sum);
        sum.intoArray(output,dest);
    }

    private void fourScalar(float[] input, int x, float[] bias, int bo, float[] output, int dest, int oc) {
        dest += oc * plane;
        int w0 = weightOffset + oc * channels, w1 = w0 + channels, w2 = w1 + channels, w3 = w2 + channels;
        float s0 = bias == null ? 0 : bias[bo + oc], s1 = bias == null ? 0 : bias[bo + oc + 1];
        float s2 = bias == null ? 0 : bias[bo + oc + 2], s3 = bias == null ? 0 : bias[bo + oc + 3];
        for (int k = 0; k < channels; k++) {
            float a = input[k * PANEL + x];
            s0 += a * weights[w0 + k]; s1 += a * weights[w1 + k];
            s2 += a * weights[w2 + k]; s3 += a * weights[w3 + k];
        }
        output[dest] = s0; output[dest + plane] = s1;
        output[dest + 2 * plane] = s2; output[dest + 3 * plane] = s3;
    }
}
