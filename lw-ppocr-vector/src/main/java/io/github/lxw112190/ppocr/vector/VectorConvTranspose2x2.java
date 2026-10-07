package io.github.lxw112190.ppocr.vector;

import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorShuffle;
import jdk.incubator.vector.VectorMask;
import jdk.incubator.vector.VectorSpecies;

/** Exact non-overlapping upsampling: two output channels share each input load. */
final class VectorConvTranspose2x2 {
    private static final VectorSpecies<Float> S=VectorSupport.F32;
    private static final int L=S.length();
    // Two-source rearrange recomputes laneIsValid() through integer vectors on JDK 25.
    // Positive single-source indices and one cached blend mask avoid that allocation.
    private static final VectorShuffle<Float> LOW=VectorShuffle.fromOp(S,i->i/2);
    private static final VectorShuffle<Float> HIGH=VectorShuffle.fromOp(S,i->L/2+i/2);
    private static final VectorMask<Float> ODD=VectorMask.fromLong(S,0xAAAAAAAAAAAAAAAAL);

    static void run(float[] a,int ao,float[] w,int wo,float[] b,int bo,float[] out,int oo,
                    int ic,int h,int width,int oc,int firstRow,int endRow) {
        int plane=h*width, outWidth=width*2, outPlane=plane*4, bound=S.loopBound(width);
        for(int row=firstRow;row<endRow;) {
            int n=row/oc,c=row%oc;
            boolean pair=c+1<oc && row+1<endRow;
            float b0=b==null?0:b[bo+c],b1=pair && b!=null?b[bo+c+1]:0;
            for(int y=0;y<h;y++) {
                int source=ao+n*ic*plane+y*width;
                int dest=oo+row*outPlane+y*2*outWidth;
                int x=0;
                for(;x<bound;x+=L) {
                    if(pair) two(a,source+x,w,wo+c*4,ic,plane,oc*4,b0,b1,out,dest+x*2,outWidth,outPlane);
                    else one(a,source+x,w,wo+c*4,ic,plane,oc*4,b0,out,dest+x*2,outWidth);
                }
                for(;x<width;x++) {
                    scalar(a,source+x,w,wo+c*4,ic,plane,oc*4,b0,out,dest+x*2,outWidth);
                    if(pair) scalar(a,source+x,w,wo+(c+1)*4,ic,plane,oc*4,b1,out,dest+outPlane+x*2,outWidth);
                }
            }
            row+=pair?2:1;
        }
    }

    private static void two(float[] a,int source,float[] w,int weight,int ic,int plane,int stride,
                            float bias0,float bias1,float[] out,int dest,int width,int outPlane) {
        FloatVector s00=FloatVector.broadcast(S,bias0),s01=s00,s10=s00,s11=s00;
        FloatVector t00=FloatVector.broadcast(S,bias1),t01=t00,t10=t00,t11=t00;
        for(int k=0;k<ic;k++,source+=plane,weight+=stride) {
            FloatVector v=FloatVector.fromArray(S,a,source);
            s00=s00.add(v.mul(w[weight])); s01=s01.add(v.mul(w[weight+1]));
            s10=s10.add(v.mul(w[weight+2])); s11=s11.add(v.mul(w[weight+3]));
            t00=t00.add(v.mul(w[weight+4])); t01=t01.add(v.mul(w[weight+5]));
            t10=t10.add(v.mul(w[weight+6])); t11=t11.add(v.mul(w[weight+7]));
        }
        s00.rearrange(LOW).blend(s01.rearrange(LOW),ODD).intoArray(out,dest); s00.rearrange(HIGH).blend(s01.rearrange(HIGH),ODD).intoArray(out,dest+L);
        s10.rearrange(LOW).blend(s11.rearrange(LOW),ODD).intoArray(out,dest+width); s10.rearrange(HIGH).blend(s11.rearrange(HIGH),ODD).intoArray(out,dest+width+L);
        dest+=outPlane;
        t00.rearrange(LOW).blend(t01.rearrange(LOW),ODD).intoArray(out,dest); t00.rearrange(HIGH).blend(t01.rearrange(HIGH),ODD).intoArray(out,dest+L);
        t10.rearrange(LOW).blend(t11.rearrange(LOW),ODD).intoArray(out,dest+width); t10.rearrange(HIGH).blend(t11.rearrange(HIGH),ODD).intoArray(out,dest+width+L);
    }

    private static void one(float[] a,int source,float[] w,int weight,int ic,int plane,int stride,
                            float bias,float[] out,int dest,int width) {
        FloatVector s00=FloatVector.broadcast(S,bias),s01=s00,s10=s00,s11=s00;
        for(int k=0;k<ic;k++,source+=plane,weight+=stride) {
            FloatVector v=FloatVector.fromArray(S,a,source);
            s00=s00.add(v.mul(w[weight])); s01=s01.add(v.mul(w[weight+1]));
            s10=s10.add(v.mul(w[weight+2])); s11=s11.add(v.mul(w[weight+3]));
        }
        s00.rearrange(LOW).blend(s01.rearrange(LOW),ODD).intoArray(out,dest); s00.rearrange(HIGH).blend(s01.rearrange(HIGH),ODD).intoArray(out,dest+L);
        s10.rearrange(LOW).blend(s11.rearrange(LOW),ODD).intoArray(out,dest+width); s10.rearrange(HIGH).blend(s11.rearrange(HIGH),ODD).intoArray(out,dest+width+L);
    }

    private static void scalar(float[] a,int source,float[] w,int weight,int ic,int plane,int stride,
                                float bias,float[] out,int dest,int width) {
        float s00=bias,s01=bias,s10=bias,s11=bias;
        for(int k=0;k<ic;k++,source+=plane,weight+=stride) {
            float v=a[source];
            s00+=v*w[weight];s01+=v*w[weight+1];s10+=v*w[weight+2];s11+=v*w[weight+3];
        }
        out[dest]=s00;out[dest+1]=s01;out[dest+width]=s10;out[dest+width+1]=s11;
    }
}
