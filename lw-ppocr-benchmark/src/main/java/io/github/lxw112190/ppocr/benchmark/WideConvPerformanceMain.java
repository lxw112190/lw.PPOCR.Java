package io.github.lxw112190.ppocr.benchmark;

import io.github.lxw112190.ppocr.kernels.KernelBackend;
import io.github.lxw112190.ppocr.kernels.ScalarBackend;
import java.util.Arrays;
import java.util.Locale;

/** Diagnostic shapes from Medium DET; not an additional CI speed gate. */
public final class WideConvPerformanceMain {
    private static volatile float sink;
    private WideConvPerformanceMain() { }
    public static void main(String[] args) throws Exception {
        String backendName=args.length>0?args[0]:"vector";
        String profile=args.length>1?args[1]:"dense7";
        int warmup=args.length>2?positive(args[2]):30, iterations=args.length>3?positive(args[3]):30;
        int kh,kw,ic=32,oc=32,h=128,width=128,groups=1;
        if("dense7".equals(profile)) {kh=7;kw=7;}
        else if("strip7".equals(profile)) {kh=1;kw=7;}
        else if("depthwise9".equals(profile)) {kh=9;kw=9;ic=oc=groups=256;}
        else if("depthwise7".equals(profile)) {kh=7;kw=7;ic=oc=groups=96;}
        else if("depthwise3".equals(profile)) {kh=3;kw=3;ic=oc=groups=128;h=6;width=240;}
        else throw new IllegalArgumentException("unknown profile: "+profile);
        KernelBackend backend="scalar".equals(backendName)?new ScalarBackend():
                "vector".equals(backendName)?(KernelBackend)Class.forName(
                "io.github.lxw112190.ppocr.vector.VectorBackend").getDeclaredConstructor().newInstance():null;
        if(backend==null)throw new IllegalArgumentException("backend must be scalar or vector");
        float[] a=fixture(ic*h*width), w=fixture(oc*(ic/groups)*kh*kw), b=fixture(oc), c=new float[oc*h*width];
        long[] times=new long[iterations];
        for(int i=-warmup;i<iterations;i++) {
            long start=System.nanoTime();
            backend.conv(a,0,w,0,b,0,c,0,1,ic,h,width,oc,kh,kw,1,1,1,1,
                    kh/2,kw/2,kh/2,kw/2,groups,h,width);
            if(i>=0)times[i]=System.nanoTime()-start;
            sink=c[(i+warmup)*7919%c.length];
        }
        long total=0,hash=-3750763034362895579L;
        for(long t:times)total+=t;Arrays.sort(times);
        for(float f:c){hash^=Float.floatToIntBits(f)&0xffffffffL;hash*=1099511628211L;}
        System.out.printf(Locale.ROOT,"{\"benchmark\":\"wide-conv\",\"profile\":\"%s\",\"backend\":\"%s\","
                +"\"warmup\":%d,\"iterations\":%d,\"mean_ms\":%.3f,\"median_ms\":%.3f,"
                +"\"checksum\":\"%s\"}%n",profile,backendName,warmup,iterations,
                total/(iterations*1e6),times[iterations/2]/1e6,Long.toUnsignedString(hash));
    }
    private static int positive(String s){int n=Integer.parseInt(s);if(n<1)throw new IllegalArgumentException("must be positive");return n;}
    private static float[] fixture(int n){float[] a=new float[n];for(int i=0;i<n;i++)a[i]=(i%101-50)*.0009765625f;return a;}
}
