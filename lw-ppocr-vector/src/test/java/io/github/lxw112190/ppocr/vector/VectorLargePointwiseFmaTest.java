package io.github.lxw112190.ppocr.vector;
import io.github.lxw112190.ppocr.kernels.PreparedConvBackend;
import java.util.Arrays;
import java.util.Random;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.*;
import org.junit.Test;
import static org.junit.Assert.*;
public final class VectorLargePointwiseFmaTest {
    @Test public void extendedCoverageRequiresFmaAndMatchesItsReference() {
        String oldFma=System.getProperty("lwppocr.vectorFma"),oldExtended=System.getProperty("lwppocr.disableExtendedFmaPointwise");
        try {
            System.setProperty("lwppocr.disableExtendedFmaPointwise","false");
            System.setProperty("lwppocr.vectorFma","false");
            assertNull(VectorLargePointwise.prepare(new float[128*129],0,
                    new int[]{1,128,5,17,129,1,1,1,1,1,1,0,0,0,0,1,5,17}));
            System.setProperty("lwppocr.vectorFma","true");
            int[] inputs=Boolean.getBoolean("lwppocr.smallFmaPointwise")?new int[]{32,48,64,80,128,160,255}:new int[]{64,80,128,160,255};
            int[] outputs=Boolean.getBoolean("lwppocr.smallFmaPointwise")?new int[]{32,33,64,65,80,128,129,259}:new int[]{64,65,80,128,129,259};
            for(int c:inputs)for(int oc:outputs)for(boolean biased:new boolean[]{false,true}) {
                int plane=85;Random random=new Random(91);
                float[] a=new float[c*plane+7],w=new float[c*oc+9],b=biased?new float[oc+5]:null;
                for(int i=0;i<a.length;i++)a[i]=random.nextFloat()*2-1;
                for(int i=0;i<w.length;i++)w[i]=random.nextFloat()*2-1;
                if(b!=null)for(int i=0;i<b.length;i++)b[i]=random.nextFloat()*2-1;
                float[] expected=new float[oc*plane+22],actual=new float[expected.length];
                Arrays.fill(expected,-31);Arrays.fill(actual,-31);
                for(int o=0;o<oc;o++)for(int x=0;x<plane;x++) {
                    float sum=b==null?0:b[5+o];for(int k=0;k<c;k++)sum=Math.fma(a[7+k*plane+x],w[9+o*c+k],sum);
                    expected[11+o*plane+x]=sum;
                }
                PreparedConvBackend.Kernel p=VectorLargePointwise.prepare(w,9,
                        new int[]{1,c,5,17,oc,1,1,1,1,1,1,0,0,0,0,1,5,17});
                assertNotNull(p);float[] scratch=new float[p.scratchFloats()];
                for(int repeat=0;repeat<2;repeat++) {
                    for(int row=p.outputRows()-1;row>=0;row--)p.runRows(a,7,b,5,actual,11,scratch,row,row+1);
                    for(int i=0;i<actual.length;i++)assertEquals(Float.floatToIntBits(expected[i]),Float.floatToIntBits(actual[i]));
                }
            }
        } finally {
            if(oldFma==null)System.clearProperty("lwppocr.vectorFma");else System.setProperty("lwppocr.vectorFma",oldFma);
            if(oldExtended==null)System.clearProperty("lwppocr.disableExtendedFmaPointwise");else System.setProperty("lwppocr.disableExtendedFmaPointwise",oldExtended);
        }
    }
    @Test public void smallChannelCandidateMatchesFmaReference() {
        String old=System.getProperty("lwppocr.smallFmaPointwise");
        try {
            System.setProperty("lwppocr.smallFmaPointwise","true");
            extendedCoverageRequiresFmaAndMatchesItsReference();
        } finally {
            if(old==null)System.clearProperty("lwppocr.smallFmaPointwise");else System.setProperty("lwppocr.smallFmaPointwise",old);
        }
    }
    @Test public void coverageFallbackRetainsOriginalEligibility() {
        String old=System.getProperty("lwppocr.disableExtendedFmaPointwise"),oldFma=System.getProperty("lwppocr.vectorFma");
        try {
            System.setProperty("lwppocr.disableExtendedFmaPointwise","true");System.setProperty("lwppocr.vectorFma","true");
            assertNull(VectorLargePointwise.prepare(new float[128*129],0,
                    new int[]{1,128,5,17,129,1,1,1,1,1,1,0,0,0,0,1,5,17}));
            fusedRoundingMatchesMathFmaWithOffsetsTailsAndReuse();
        } finally {
            if(old==null)System.clearProperty("lwppocr.disableExtendedFmaPointwise");else System.setProperty("lwppocr.disableExtendedFmaPointwise",old);
            if(oldFma==null)System.clearProperty("lwppocr.vectorFma");else System.setProperty("lwppocr.vectorFma",oldFma);
        }
    }
    @Test public void fourChannelFallbackMatchesFmaReference() {
        String old=System.getProperty("lwppocr.disableSixPointwise");
        try {
            System.setProperty("lwppocr.disableSixPointwise","true");
            fusedRoundingMatchesMathFmaWithOffsetsTailsAndReuse();
        } finally {
            if(old==null)System.clearProperty("lwppocr.disableSixPointwise");else System.setProperty("lwppocr.disableSixPointwise",old);
        }
    }
    @Test public void concurrentReuseMatchesFusedSpecialValues() throws Exception {
        String old=System.getProperty("lwppocr.vectorFma");
        ExecutorService workers=Executors.newFixedThreadPool(4);
        try {
            System.setProperty("lwppocr.vectorFma","true");
            int c=256,oc=259,plane=258;
            float[] input=new float[c*plane],weights=new float[oc*c],bias=new float[oc];
            Arrays.fill(input,-0.0f); Arrays.fill(weights,1.0f); Arrays.fill(bias,-0.0f);
            weights[0]=Float.POSITIVE_INFINITY; weights[c]=Float.NaN;
            input[4*plane+7]=Float.NEGATIVE_INFINITY;
            float[] expected=new float[oc*plane+22],actual=new float[expected.length];
            Arrays.fill(expected,-31); Arrays.fill(actual,-31);
            for(int o=0;o<oc;o++) for(int x=0;x<plane;x++) {
                float sum=bias[o];
                for(int k=0;k<c;k++) sum=Math.fma(input[k*plane+x],weights[o*c+k],sum);
                expected[11+o*plane+x]=sum;
            }
            PreparedConvBackend.Kernel plan=new VectorBackend().prepareConv(weights,0,
                new int[]{1,c,2,129,oc,1,1,1,1,1,1,0,0,0,0,1,2,129});
            float[][] scratch=new float[4][plan.scratchFloats()];
            for(int repeat=0;repeat<2;repeat++) {
                List<Future<?>> tasks=new ArrayList<>();
                for(int shard=0;shard<4;shard++) {
                    final int worker=shard;
                    tasks.add(workers.submit(()->plan.runRows(input,0,bias,0,actual,11,scratch[worker],
                        plan.outputRows()*worker/4,plan.outputRows()*(worker+1)/4)));
                }
                for(Future<?> task:tasks) task.get();
                for(int i=0;i<actual.length;i++) assertEquals("index "+i,
                    Float.floatToIntBits(expected[i]),Float.floatToIntBits(actual[i]));
            }
        } finally {
            workers.shutdownNow();
            if(old==null) System.clearProperty("lwppocr.vectorFma"); else System.setProperty("lwppocr.vectorFma",old);
        }
    }
    @Test public void fusedRoundingMatchesMathFmaWithOffsetsTailsAndReuse() {
        String old = System.getProperty("lwppocr.vectorFma");
        try {
            System.setProperty("lwppocr.vectorFma", "true");
            for (int c : new int[] {256,321,1024}) for (boolean biased : new boolean[] {false,true}) {
                int oc=259, plane=85; Random random = new Random(901+c);
                float[] input = new float[c*plane+7], weights = new float[oc*c+9];
                float[] bias = biased ? new float[oc+5] : null;
                for (int i=0;i<input.length;i++) input[i]=(random.nextFloat()-.5f)*4;
                for (int i=0;i<weights.length;i++) weights[i]=(random.nextFloat()-.5f)*4;
                if (bias!=null) for (int i=0;i<bias.length;i++) bias[i]=(random.nextFloat()-.5f);
                float[] expected=new float[oc*plane+22],actual=new float[expected.length];
                Arrays.fill(expected,-31); Arrays.fill(actual,-31);
                for (int o=0;o<oc;o++) for(int x=0;x<plane;x++) {
                    float sum=bias==null?0:bias[5+o];
                    for(int k=0;k<c;k++) sum=Math.fma(input[7+k*plane+x],weights[9+o*c+k],sum);
                    expected[11+o*plane+x]=sum;
                }
                PreparedConvBackend.Kernel plan = new VectorBackend().prepareConv(weights,9,
                    new int[]{1,c,5,17,oc,1,1,1,1,1,1,0,0,0,0,1,5,17});
                float[] scratch = new float[plan.scratchFloats()];
                for(int repeat=0;repeat<2;repeat++) {
                    int split=plan.outputRows()/2;
                    plan.runRows(input,7,bias,5,actual,11,scratch,split,plan.outputRows());
                    plan.runRows(input,7,bias,5,actual,11,scratch,0,split);
                    for(int i=0;i<actual.length;i++) assertEquals("index "+i,
                        Float.floatToIntBits(expected[i]),Float.floatToIntBits(actual[i]));
                }
            }
        } finally {
            if(old==null) System.clearProperty("lwppocr.vectorFma"); else System.setProperty("lwppocr.vectorFma",old);
        }
    }
}
