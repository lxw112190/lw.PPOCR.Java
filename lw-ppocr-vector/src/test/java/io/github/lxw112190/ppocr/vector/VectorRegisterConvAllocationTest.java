package io.github.lxw112190.ppocr.vector;

import com.sun.management.ThreadMXBean;
import java.lang.management.ManagementFactory;
import java.util.Arrays;
import jdk.incubator.vector.FloatVector;
import org.junit.Assume;
import org.junit.Test;
import static org.junit.Assert.*;

/** Keep the new spatial kernels allocation-free once their Vector loops are hot. */
public final class VectorRegisterConvAllocationTest {
    @Test public void warmedWideAndDepthwiseKernelsHaveBoundedAllocation() {
        Assume.assumeTrue(VectorSupport.F32.equals(FloatVector.SPECIES_PREFERRED));
        java.lang.management.ThreadMXBean raw=ManagementFactory.getThreadMXBean();
        Assume.assumeTrue(raw instanceof ThreadMXBean);
        ThreadMXBean bean=(ThreadMXBean)raw;
        Assume.assumeTrue(bean.isThreadAllocatedMemorySupported());
        if(!bean.isThreadAllocatedMemoryEnabled())bean.setThreadAllocatedMemoryEnabled(true);
        for(boolean depthwise:new boolean[]{false,true}) {
            int channels=32,size=64,kernel=depthwise?9:7,groups=depthwise?channels:1;
            float[] a=new float[channels*size*size],w=new float[channels*(channels/groups)*kernel*kernel];
            float[] c=new float[a.length];Arrays.fill(a,.25f);Arrays.fill(w,.5f);
            VectorBackend backend=new VectorBackend();
            for(int i=0;i<30;i++)run(backend,a,w,c,channels,size,kernel,groups);
            long id=Thread.currentThread().threadId(),before=bean.getThreadAllocatedBytes(id);
            for(int i=0;i<5;i++)run(backend,a,w,c,channels,size,kernel,groups);
            long bytes=bean.getThreadAllocatedBytes(id)-before;
            assertTrue("hot spatial allocation: "+bytes,bytes>=0 && bytes<=100000);
            assertEquals((channels/groups)*kernel*kernel*.125f,c[size*32+32],0);
        }
    }
    private static void run(VectorBackend b,float[] a,float[] w,float[] c,int ch,int size,int k,int groups) {
        b.conv(a,0,w,0,null,0,c,0,1,ch,size,size,ch,k,k,1,1,1,1,k/2,k/2,k/2,k/2,groups,size,size);
    }
}
