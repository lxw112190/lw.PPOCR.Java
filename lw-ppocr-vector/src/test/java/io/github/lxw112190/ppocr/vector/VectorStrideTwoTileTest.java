package io.github.lxw112190.ppocr.vector;

import com.sun.management.ThreadMXBean;
import io.github.lxw112190.ppocr.kernels.ScalarBackend;
import java.lang.management.ManagementFactory;
import java.util.Arrays;
import jdk.incubator.vector.FloatVector;
import org.junit.Assume;
import org.junit.Test;
import static org.junit.Assert.*;

public final class VectorStrideTwoTileTest {
    @Test public void outlinedTilesKeepBatchOffsetsPaddingAndTailsBitIdentical() {
        for(int channels:new int[]{3,32})for(int width:new int[]{3,17,35,49})
            check(channels,9,width,false);
    }

    @Test public void nonfiniteWeightsDoNotTurnPaddingIntoProducts() {
        check(3,3,49,true);
    }

    @Test public void genericRowsKeepGroupsDilationOffsetsAndChannelTailsBitIdentical() {
        for (int groups : new int[]{1,2}) for (int oc : new int[]{1,4,8,12,16,25})
            for (int width : new int[]{3,17,35,49}) checkGeneric(groups,oc,width,false);
        checkGeneric(2,25,35,true);
    }

    @Test public void hotPreferredGenericRowsHaveBoundedAllocation() {
        Assume.assumeTrue(VectorSupport.F32.equals(FloatVector.SPECIES_PREFERRED));
        java.lang.management.ThreadMXBean raw=ManagementFactory.getThreadMXBean();
        Assume.assumeTrue(raw instanceof ThreadMXBean);
        ThreadMXBean bean=(ThreadMXBean)raw;
        Assume.assumeTrue(bean.isThreadAllocatedMemorySupported());
        if(!bean.isThreadAllocatedMemoryEnabled())bean.setThreadAllocatedMemoryEnabled(true);
        VectorBackend backend=new VectorBackend();
        for(int oc:new int[]{1,4,8,12,16,25}) {
            float[] a=new float[3*32*160],w=new float[oc*3*9],b=new float[oc],out=new float[oc*16*80];
            Arrays.fill(a,.25f);Arrays.fill(w,.5f);Arrays.fill(b,.125f);
            for(int i=0;i<100;i++)generic(backend,a,w,b,out,oc);
            long id=Thread.currentThread().threadId(),before=bean.getThreadAllocatedBytes(id);
            for(int i=0;i<5;i++)generic(backend,a,w,b,out,oc);
            long bytes=bean.getThreadAllocatedBytes(id)-before;
            System.out.println("hot generic stride-two: species_bits="+VectorSupport.F32.vectorBitSize()
                    +", output_channels="+oc+", calls=5, allocated_bytes="+bytes);
            assertTrue("generic stride-two allocation for OC="+oc+": "+bytes,bytes>=0 && bytes<=100000);
            assertEquals(3.5f,out[8*80+40],0);
        }
    }

    @Test public void hotPreferredStemHasBoundedAllocation() {
        Assume.assumeTrue(VectorSupport.F32.equals(FloatVector.SPECIES_PREFERRED));
        java.lang.management.ThreadMXBean raw=ManagementFactory.getThreadMXBean();
        Assume.assumeTrue(raw instanceof ThreadMXBean);
        ThreadMXBean bean=(ThreadMXBean)raw;
        Assume.assumeTrue(bean.isThreadAllocatedMemorySupported());
        if(!bean.isThreadAllocatedMemoryEnabled())bean.setThreadAllocatedMemoryEnabled(true);
        float[] a=new float[3*320*320],w=new float[16*3*9],bias=new float[16],out=new float[16*160*160];
        Arrays.fill(a,.25f);Arrays.fill(w,.5f);Arrays.fill(bias,.125f);
        VectorBackend backend=new VectorBackend();
        for(int i=0;i<100;i++)stem(backend,a,w,bias,out);
        long id=Thread.currentThread().threadId(),before=bean.getThreadAllocatedBytes(id);
        for(int i=0;i<5;i++)stem(backend,a,w,bias,out);
        long bytes=bean.getThreadAllocatedBytes(id)-before;
        System.out.println("hot stride-two tile: species_bits="+VectorSupport.F32.vectorBitSize()
                +", calls=5, allocated_bytes="+bytes);
        assertTrue("hot stride-two tile allocation: "+bytes,bytes>=0 && bytes<=100000);
        assertEquals(3.5f,out[80*160+80],0);
    }

    private static void stem(VectorBackend backend,float[] a,float[] w,float[] b,float[] c) {
        backend.conv(a,0,w,0,b,0,c,0,1,3,320,320,16,3,3,2,2,1,1,1,1,1,1,1,160,160);
    }

    private static void generic(VectorBackend backend,float[] a,float[] w,float[] b,float[] c,int oc) {
        backend.conv(a,0,w,0,b,0,c,0,1,3,32,160,oc,3,3,2,2,2,2,2,2,2,2,1,16,80);
    }

    private static void checkGeneric(int groups,int ocPerGroup,int width,boolean special) {
        int batch=2,channels=6,height=7,oc=groups*ocPerGroup,oh=4,ow=(width+1)/2;
        float[] a=values(batch*channels*height*width+7),w=values(oc*channels/groups*9+9),b=values(oc+5);
        if(special){Arrays.fill(a,-0.0f);Arrays.fill(w,1);Arrays.fill(b,-0.0f);w[9]=Float.NaN;w[10]=Float.POSITIVE_INFINITY;}
        VectorBackend backend=new VectorBackend();
        for(int repeat=0;repeat<2;repeat++) {
            float[] bias=repeat==0?b:null;
            float[] expected=new float[batch*oc*oh*ow+22],actual=new float[expected.length];
            Arrays.fill(expected,-31);Arrays.fill(actual,-31);
            new ScalarBackend().conv(a,7,w,9,bias,5,expected,11,batch,channels,height,width,oc,
                    3,3,2,2,2,2,2,2,2,2,groups,oh,ow);
            backend.conv(a,7,w,9,bias,5,actual,11,batch,channels,height,width,oc,
                    3,3,2,2,2,2,2,2,2,2,groups,oh,ow);
            for(int i=0;i<actual.length;i++)assertEquals("groups="+groups+", OC="+ocPerGroup+", index="+i,
                    Float.floatToIntBits(expected[i]),Float.floatToIntBits(actual[i]));
        }
    }

    private static void check(int channels,int h,int width,boolean special) {
        int oc=16,batch=2,oh=(h+1)/2,ow=(width+1)/2;
        float[] a=values(batch*channels*h*width+7),w=values(oc*channels*9+9),b=values(oc+5);
        if(special){Arrays.fill(a,-0.0f);Arrays.fill(w,1);Arrays.fill(b,-0.0f);w[9]=Float.NaN;w[10]=Float.POSITIVE_INFINITY;}
        VectorBackend backend=new VectorBackend();
        for(int repeat=0;repeat<2;repeat++) {
            float[] bias=repeat==0?b:null;
            float[] expected=new float[batch*oc*oh*ow+22],actual=new float[expected.length];
            Arrays.fill(expected,-31);Arrays.fill(actual,-31);
            new ScalarBackend().conv(a,7,w,9,bias,5,expected,11,batch,channels,h,width,oc,
                    3,3,2,2,1,1,1,1,1,1,1,oh,ow);
            backend.conv(a,7,w,9,bias,5,actual,11,batch,channels,h,width,oc,
                    3,3,2,2,1,1,1,1,1,1,1,oh,ow);
            for(int i=0;i<actual.length;i++)assertEquals("index "+i,
                    Float.floatToIntBits(expected[i]),Float.floatToIntBits(actual[i]));
        }
    }
    private static float[] values(int n){float[] a=new float[n];for(int i=0;i<n;i++)a[i]=(i%103-51)*.0012345f;return a;}
}
