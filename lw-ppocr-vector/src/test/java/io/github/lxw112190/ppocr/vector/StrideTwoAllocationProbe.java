package io.github.lxw112190.ppocr.vector;

import com.sun.management.ThreadMXBean;
import java.lang.management.ManagementFactory;
import java.util.Arrays;

/** Failure diagnostics, not a replacement for the strict allocation assertions. */
public final class StrideTwoAllocationProbe {
    private StrideTwoAllocationProbe() { }

    public static void main(String[] args) {
        java.lang.management.ThreadMXBean raw = ManagementFactory.getThreadMXBean();
        System.out.println("stride-two diagnostic: java=" + System.getProperty("java.version")
                + ", arch=" + System.getProperty("os.arch")
                + ", cpus=" + Runtime.getRuntime().availableProcessors()
                + ", species_bits=" + VectorSupport.F32.vectorBitSize()
                + ", flags=" + ManagementFactory.getRuntimeMXBean().getInputArguments());
        if (!(raw instanceof ThreadMXBean)
                || !((ThreadMXBean) raw).isThreadAllocatedMemorySupported()) {
            System.out.println("allocation counter unsupported");
            return;
        }
        ThreadMXBean bean = (ThreadMXBean) raw;
        if (!bean.isThreadAllocatedMemoryEnabled()) bean.setThreadAllocatedMemoryEnabled(true);
        VectorBackend backend = new VectorBackend();
        long id = Thread.currentThread().threadId();
        for (int oc : new int[]{1,4,8,12,16,25}) {
            float[] a = new float[3*32*160], w = new float[oc*3*9];
            float[] b = new float[oc], out = new float[oc*16*80];
            Arrays.fill(a, .25f); Arrays.fill(w, .5f); Arrays.fill(b, .125f);
            int calls = 0;
            // Both budgets are fixed ahead of time, regardless of the result.
            for (int target : new int[]{100,1100}) {
                while (calls < target) { run(backend,a,w,b,out,oc); calls++; }
                long[] perCall = new long[5];
                long before = bean.getThreadAllocatedBytes(id), previous = before;
                long start = System.nanoTime();
                for (int i = 0; i < 5; i++) {
                    run(backend,a,w,b,out,oc);
                    long after = bean.getThreadAllocatedBytes(id);
                    perCall[i] = after - previous;
                    previous = after;
                }
                long elapsed = System.nanoTime() - start;
                calls += 5;
                System.out.println("generic stride-two diagnostic: oc=" + oc
                        + ", prior_calls=" + target + ", measured_calls=5"
                        + ", allocated_bytes=" + (previous-before)
                        + ", bytes_per_call=" + Arrays.toString(perCall)
                        + ", elapsed_ns=" + elapsed + ", interior=" + out[8*80+40]);
            }
        }
    }

    private static void run(VectorBackend backend, float[] a, float[] w, float[] b,
                            float[] out, int oc) {
        backend.conv(a,0,w,0,b,0,out,0,1,3,32,160,oc,3,3,2,2,2,2,2,2,2,2,1,16,80);
    }
}
