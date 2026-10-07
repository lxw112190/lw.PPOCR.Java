package io.github.lxw112190.ppocr.vector;

import io.github.lxw112190.ppocr.kernels.ScalarBackend;
import java.util.Arrays;
import java.util.Random;
import org.junit.Test;
import static org.junit.Assert.*;

public final class VectorRegisterConvTest {
    @Test public void wideShapesOffsetsBatchesTailsAndReuseMatchScalarBits() {
        for (int[] shape : new int[][] {{7,7}, {1,7}, {7,1}})
            for (int width : new int[] {1,3,7,17,35})
                for (int oc : new int[] {1,4,7,8})
                    check(2,3,5,width,oc,shape[0],shape[1],1,1,false);
    }

    @Test public void depthwiseStrideHeightAndSmallImagesMatchScalarBits() {
        for (int kernel : new int[] {3,7,9}) for (int sh : new int[] {1,2})
            for (int width : new int[] {1,3,7,17,35})
                for (int h : new int[] {1,3,11})
                    check(2,3,h,width,3,kernel,kernel,sh,3,false);
    }

    @Test public void paddingSkipsNonfiniteWeightsAndPreservesSignedZero() {
        check(2,3,3,35,7,7,7,1,1,true);
        check(1,3,3,35,3,9,9,2,3,true);
        check(1,3,3,35,3,3,3,1,3,true);
    }

    @Test public void nonEligibleGeometryStillFallsBack() {
        float[] a=values(3*7*35+5), w=values(7*3*49+7), c=new float[7*4*18+11];
        float[] expected=new float[c.length];
        new ScalarBackend().conv(a,5,w,7,null,0,expected,11,1,3,7,35,7,
                7,7,2,2,1,1,3,3,3,3,1,4,18);
        new VectorBackend().conv(a,5,w,7,null,0,c,11,1,3,7,35,7,
                7,7,2,2,1,1,3,3,3,3,1,4,18);
        bits(expected,c);
    }

    private static void check(int batch,int ic,int h,int width,int oc,int kh,int kw,
                              int sh,int groups,boolean special) {
        int oh=(h+sh-1)/sh;
        float[] a=values(batch*ic*h*width+5), w=values(oc*(ic/groups)*kh*kw+7);
        float[] b=values(oc+3), c=new float[batch*oc*oh*width+20];
        if(special) {
            Arrays.fill(a,-0.0f); Arrays.fill(w,1.0f); Arrays.fill(b,-0.0f);
            w[7]=Float.POSITIVE_INFINITY; w[8]=Float.NaN;
            a[5+Math.min(3,batch*ic*h*width-1)]=Float.NEGATIVE_INFINITY;
        }
        VectorBackend backend=new VectorBackend();
        for(int iteration=0;iteration<2;iteration++) {
            float[] bias=iteration==0?b:null;
            float[] expected=new float[c.length];
            Arrays.fill(expected, -73); Arrays.fill(c,-73);
            new ScalarBackend().conv(a,5,w,7,bias,3,expected,11,batch,ic,h,width,oc,
                    kh,kw,sh,1,1,1,kh/2,kw/2,kh/2,kw/2,groups,oh,width);
            backend.conv(a,5,w,7,bias,3,c,11,batch,ic,h,width,oc,
                    kh,kw,sh,1,1,1,kh/2,kw/2,kh/2,kw/2,groups,oh,width);
            bits(expected,c);
        }
    }

    private static float[] values(int size) {
        Random r=new Random(size);float[] a=new float[size];
        for(int i=0;i<size;i++)a[i]=(r.nextFloat()-.5f)*.0625f;
        return a;
    }

    private static void bits(float[] expected,float[] actual) {
        for(int i=0;i<expected.length;i++)
            assertEquals("index "+i,Float.floatToIntBits(expected[i]),Float.floatToIntBits(actual[i]));
    }
}
