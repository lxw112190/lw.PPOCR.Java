package io.github.lxw112190.ppocr.runtime;
import io.github.lxw112190.ppocr.kernels.*;
import io.github.lxw112190.ppocr.vector.VectorBackend;
import java.lang.reflect.*;
import java.util.Arrays;
import java.util.concurrent.atomic.*;
import org.junit.*;
import static org.junit.Assert.*;

public final class VectorParallelTransposeTest {
    @Test public void parallelTransposeJoinsFailuresAndKeepsOtherModesReusable() {
        Assume.assumeTrue(Runtime.getRuntime().availableProcessors()>1);
        VectorBackend delegate=new VectorBackend(); AtomicBoolean fail=new AtomicBoolean(true);
        AtomicInteger calls=new AtomicInteger();
        KernelBackend backend=(KernelBackend)Proxy.newProxyInstance(getClass().getClassLoader(),
            new Class<?>[]{KernelBackend.class,ConvTransposeRowsBackend.class},(proxy,method,args)->{
                if(method.getName().equals("convTransposeRows")) {
                    calls.incrementAndGet();
                    if(fail.get() && (int)args[9]==0)throw new IllegalArgumentException("transpose shard failure");
                }
                try {return method.invoke(delegate,args);} catch(InvocationTargetException e){throw e.getCause();}
            });
        ParallelKernels parallel=new ParallelKernels(backend);parallel.setParallelism(4);
        int ic=16,oc=17,h=64,w=65,plane=h*w;
        float[] a=values(ic*plane+7),weights=values(ic*oc*4+9),bias=values(oc+5);
        float[] expected=new float[oc*plane*4+22],actual=new float[expected.length];
        Arrays.fill(expected,-31);Arrays.fill(actual,-31);
        new ScalarBackend().convTranspose(a,7,weights,9,bias,5,expected,11,1,ic,h,w,oc,2,2,2,2,1,1,0,0,1,h*2,w*2);
        try {
            parallel.convTranspose(a,7,weights,9,bias,5,actual,11,1,ic,h,w,oc,2,2,2,2,1,1,0,0,1,h*2,w*2);
            Assert.fail("failure must reach caller");
        } catch(IllegalArgumentException e) {assertEquals("transpose shard failure",e.getMessage());}
        fail.set(false);
        for(int repeat=0;repeat<2;repeat++) {
            parallel.convTranspose(a,7,weights,9,bias,5,actual,11,1,ic,h,w,oc,2,2,2,2,1,1,0,0,1,h*2,w*2);
            for(int i=0;i<actual.length;i++)assertEquals(Float.floatToIntBits(expected[i]),Float.floatToIntBits(actual[i]));
            float[] left=values(35*192),right=values(192*321),want=new float[35*321],got=new float[want.length];
            new ScalarBackend().matMul(left,0,right,0,want,0,35,192,321);
            parallel.matMul(left,0,right,0,got,0,35,192,321);
            assertArrayEquals(want,got,0);
        }
        assertTrue(calls.get()>=6);
    }
    private static float[] values(int n){float[] a=new float[n];for(int i=0;i<n;i++)a[i]=(i%97-48)*.015625f;return a;}
}
