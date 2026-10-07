package io.github.lxw112190.ppocr.runtime;

import io.github.lxw112190.ppocr.kernels.*;
import io.github.lxw112190.ppocr.vector.VectorBackend;
import java.lang.reflect.*;
import java.util.Arrays;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;
import org.junit.Test;
import org.junit.Assume;
import static org.junit.Assert.*;

public final class VectorParallelEpilogueTest {
    @Test public void channelShardsPreserveTransformsTailsAndRecoverAfterFailure() throws Exception {
        Assume.assumeTrue(Runtime.getRuntime().availableProcessors()>1);
        VectorBackend real=new VectorBackend();AtomicBoolean fail=new AtomicBoolean();
        AtomicInteger calls=new AtomicInteger();
        KernelBackend proxy=(KernelBackend)Proxy.newProxyInstance(getClass().getClassLoader(),
                new Class<?>[]{KernelBackend.class,PreparedConvBackend.class,PreparedConvEpilogueRowsBackend.class},
                (object,method,args)->{
                    if(method.getName().equals("finishPreparedConvChannels")) {
                        calls.incrementAndGet();if(fail.compareAndSet(true,false))throw new IllegalStateException("test epilogue failure");
                    }
                    try{return method.invoke(real,args);}catch(InvocationTargetException e){throw e.getCause();}
                });
        ParallelKernels parallel=new ParallelKernels(proxy);parallel.setParallelism(4);
        int ic=256,oc=259,plane=321;
        float[] a=values(ic*plane+7),w=values(ic*oc+9),b=values(oc+5),residual=values(oc*plane+13);
        PreparedConvBackend.Kernel plan=real.prepareConv(w,9,
                new int[]{1,ic,1,plane,oc,1,1,1,1,1,1,0,0,0,0,1,1,plane});
        parallel.prepareSpatialScratch(plan.scratchFloats());
        float[] expected=new float[oc*plane+22],actual=new float[expected.length];
        float[] mean=values(oc),factor=values(oc),nb=values(oc),pb=values(oc);
        for(int activation=0;activation<=4;activation++) {
            ConvEpilogue ep=new ConvEpilogue(mean,factor,nb,pb,false,activation,1.4142135f,1,.5f,.2f,.5f);
            Arrays.fill(expected,-31);Arrays.fill(actual,-31);
            plan.runRows(a,7,b,5,expected,11,new float[plan.scratchFloats()],0,plan.outputRows());
            real.finishPreparedConv(expected,11,1,oc,plane,ep,residual,13);
            for(int repeat=0;repeat<2;repeat++) {
                parallel.preparedConv(plan,a,7,b,5,actual,11,1,oc,plane,ep,residual,13);
                check(expected,actual,activation);
            }
            fail.set(true);
            try {parallel.preparedConv(plan,a,7,b,5,actual,11,1,oc,plane,ep,residual,13);fail("expected shard failure");}
            catch(IllegalStateException expectedFailure){assertEquals("test epilogue failure",expectedFailure.getMessage());}
            parallel.preparedConv(plan,a,7,b,5,actual,11,1,oc,plane,ep,residual,13);
            check(expected,actual,activation);
        }
        assertTrue(calls.get()>=20);
    }
    private static float[] values(int n){float[] a=new float[n];for(int i=0;i<n;i++)a[i]=(i%67-33)*.001953125f;return a;}
    private static void check(float[] expected,float[] actual,int activation) {
        for(int i=0;i<actual.length;i++) {
            String message="activation "+activation+" index "+i;
            // EXP may change its last bit when HotSpot transitions from interpreted
            // Math.exp to the compiled Vector intrinsic during this reuse test.
            if(activation==ConvEpilogue.GELU && Float.isFinite(expected[i]))
                assertEquals(message,expected[i],actual[i],2e-7f);
            else assertEquals(message,Float.floatToIntBits(expected[i]),Float.floatToIntBits(actual[i]));
        }
    }
}
