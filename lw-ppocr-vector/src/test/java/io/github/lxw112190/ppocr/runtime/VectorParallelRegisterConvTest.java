package io.github.lxw112190.ppocr.runtime;

import io.github.lxw112190.ppocr.kernels.ScalarBackend;
import io.github.lxw112190.ppocr.vector.VectorBackend;
import java.util.Arrays;
import org.junit.Test;
import static org.junit.Assert.*;

public final class VectorParallelRegisterConvTest {
    @Test public void channelShardsPreserveWideAndDepthwiseResultsAndReuse() {
        ParallelKernels parallel=new ParallelKernels(new VectorBackend());
        parallel.setParallelism(4);
        for(int kernel:new int[]{7,9}) {
            int channels=32,h=47,width=49,groups=kernel==9?channels:1;
            float[] a=values(channels*h*width+5),w=values(channels*(channels/groups)*kernel*kernel+7);
            float[] b=values(channels+3),want=new float[channels*h*width+20],got=new float[want.length];
            Arrays.fill(want,-17);Arrays.fill(got,-17);
            new ScalarBackend().conv(a,5,w,7,b,3,want,11,1,channels,h,width,channels,
                    kernel,kernel,1,1,1,1,kernel/2,kernel/2,kernel/2,kernel/2,groups,h,width);
            for(int repeat=0;repeat<2;repeat++) {
                parallel.conv(a,5,w,7,b,3,got,11,1,channels,h,width,channels,
                        kernel,kernel,1,1,1,1,kernel/2,kernel/2,kernel/2,kernel/2,groups,h,width);
                for(int i=0;i<got.length;i++)assertEquals("index "+i,
                        Float.floatToIntBits(want[i]),Float.floatToIntBits(got[i]));
            }
        }
    }
    private static float[] values(int n){float[] a=new float[n];for(int i=0;i<n;i++)a[i]=(i%97-48)*.001953125f;return a;}
}
