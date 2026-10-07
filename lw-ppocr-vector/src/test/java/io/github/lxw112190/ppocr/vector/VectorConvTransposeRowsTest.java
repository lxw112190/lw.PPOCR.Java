package io.github.lxw112190.ppocr.vector;

import io.github.lxw112190.ppocr.kernels.ScalarBackend;
import java.util.Arrays;
import java.util.Random;
import java.util.concurrent.*;
import org.junit.Test;
import static org.junit.Assert.*;

public final class VectorConvTransposeRowsTest {
    @Test public void batchChannelShardsOffsetsTailsAndReuseAreBitExact() throws Exception {
        for(int width:new int[]{5,17,33}) for(boolean biased:new boolean[]{false,true}) compare(width,biased,false);
    }
    @Test public void specialValuesKeepScalarRounding() throws Exception {compare(17,true,true);}
    @Test public void unsupportedShapesRetainFallback() {
        int[] p={2,9,3,17,7,2,2,2,2,1,1,0,0,0,0,1,6,34};
        VectorBackend backend=new VectorBackend(); assertTrue(backend.supportsConvTransposeRows(p));
        for(int index:new int[]{5,6,7,8,9,10,11,12,15,16,17}) {
            int[] unsupported=p.clone();unsupported[index]++;
            assertFalse(backend.supportsConvTransposeRows(unsupported));
        }
    }
    private static void compare(int width,boolean biased,boolean special) throws Exception {
        int batch=2,ic=9,oc=7,h=3,plane=h*width;
        Random random=new Random(width+901);
        float[] a=new float[batch*ic*plane+7],w=new float[ic*oc*4+9],b=biased?new float[oc+5]:null;
        for(int i=0;i<a.length;i++)a[i]=(random.nextFloat()-.5f)*4;
        for(int i=0;i<w.length;i++)w[i]=(random.nextFloat()-.5f)*4;
        if(b!=null)for(int i=0;i<b.length;i++)b[i]=random.nextFloat()-.5f;
        if(special) {Arrays.fill(a,-0.0f);Arrays.fill(b,-0.0f);w[9]=Float.POSITIVE_INFINITY;w[13]=Float.NaN;a[7+4*plane+3]=Float.NEGATIVE_INFINITY;}
        float[] expected=new float[batch*oc*plane*4+22],actual=new float[expected.length];
        Arrays.fill(expected,-31);Arrays.fill(actual,-31);
        new ScalarBackend().convTranspose(a,7,w,9,b,5,expected,11,batch,ic,h,width,oc,2,2,2,2,1,1,0,0,1,h*2,width*2);
        VectorBackend backend=new VectorBackend();
        int[] p={batch,ic,h,width,oc,2,2,2,2,1,1,0,0,0,0,1,h*2,width*2};
        ExecutorService workers=Executors.newFixedThreadPool(4);
        try {
            for(int repeat=0;repeat<2;repeat++) {
                Future<?>[] tasks=new Future<?>[4];
                for(int shard=0;shard<4;shard++) {
                    final int s=shard;
                    tasks[s]=workers.submit(()->backend.convTransposeRows(a,7,w,9,b,5,actual,11,p,batch*oc*s/4,batch*oc*(s+1)/4));
                }
                for(Future<?> task:tasks)task.get();
                bits(expected,actual);
            }
            backend.convTranspose(a,7,w,9,b,5,actual,11,batch,ic,h,width,oc,2,2,2,2,1,1,0,0,1,h*2,width*2);
            bits(expected,actual);
        } finally {workers.shutdownNow();}
    }
    private static void bits(float[] expected,float[] actual) {
        for(int i=0;i<actual.length;i++)assertEquals("index "+i,Float.floatToIntBits(expected[i]),Float.floatToIntBits(actual[i]));
    }
}
