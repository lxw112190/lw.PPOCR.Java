package io.github.lxw112190.ppocr.vector;

import com.sun.management.ThreadMXBean;
import java.lang.management.ManagementFactory;
import java.util.Arrays;
import jdk.incubator.vector.FloatVector;
import org.junit.Assume;
import org.junit.Test;
import static org.junit.Assert.*;

/** Real DET upsampling size: dynamic shuffle masks must not allocate per vector. */
public final class VectorConvTransposeAllocationTest {
    @Test public void warmedPreferredSpeciesHasBoundedAllocation() {
        Assume.assumeTrue(VectorSupport.F32.equals(FloatVector.SPECIES_PREFERRED));
        java.lang.management.ThreadMXBean raw=ManagementFactory.getThreadMXBean();
        Assume.assumeTrue(raw instanceof ThreadMXBean);
        ThreadMXBean bean=(ThreadMXBean)raw;
        Assume.assumeTrue(bean.isThreadAllocatedMemorySupported());
        if(!bean.isThreadAllocatedMemoryEnabled())bean.setThreadAllocatedMemoryEnabled(true);
        int c=64,size=128;
        float[] input=new float[c*size*size],w=new float[c*c*4],bias=new float[c];
        float[] output=new float[c*size*size*4];
        Arrays.fill(input,.25f);Arrays.fill(w,.5f);Arrays.fill(bias,.125f);
        VectorBackend backend=new VectorBackend();
        for(int i=0;i<30;i++)run(backend,input,w,bias,output,c,size);
        long id=Thread.currentThread().threadId(),before=bean.getThreadAllocatedBytes(id);
        for(int i=0;i<5;i++)run(backend,input,w,bias,output,c,size);
        long bytes=bean.getThreadAllocatedBytes(id)-before;
        assertTrue("unexpected hot transpose allocation: "+bytes,bytes>=0 && bytes<=100000);
        assertEquals(8.125f,output[output.length/2],0);
    }
    private static void run(VectorBackend backend,float[] input,float[] w,float[] b,float[] out,int c,int size) {
        backend.convTranspose(input,0,w,0,b,0,out,0,1,c,size,size,c,2,2,2,2,1,1,0,0,1,size*2,size*2);
    }
}
