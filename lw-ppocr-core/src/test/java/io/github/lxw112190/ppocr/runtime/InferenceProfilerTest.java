package io.github.lxw112190.ppocr.runtime;

import io.github.lxw112190.ppocr.model.OperatorType;
import org.junit.Assert;
import org.junit.Test;

public final class InferenceProfilerTest {
    @Test
    public void sharedProfilerCollectsWorkerThreads() throws Exception {
        InferenceProfiler.Profile snapshot;
        try (InferenceProfiler profiler = InferenceProfiler.startShared()) {
            profiler.record(OperatorType.CONV, 11L);
            Thread worker = new Thread(new Runnable() {
                @Override
                public void run() {
                    InferenceProfiler.current().record(OperatorType.CONV, 17L);
                }
            });
            worker.start();
            worker.join();
            snapshot = profiler.snapshot();
        }

        Assert.assertEquals(2L, snapshot.getInvocations(OperatorType.CONV));
        Assert.assertEquals(28L, snapshot.getElapsedNanos(OperatorType.CONV));
        Assert.assertNull(InferenceProfiler.current());
    }

    @Test
    public void localProfilerDoesNotLeakToWorkerThreads() throws Exception {
        final boolean[] visible = new boolean[1];
        try (InferenceProfiler ignored = InferenceProfiler.start()) {
            Thread worker = new Thread(new Runnable() {
                @Override
                public void run() {
                    visible[0] = InferenceProfiler.current() != null;
                }
            });
            worker.start();
            worker.join();
        }
        Assert.assertFalse(visible[0]);
    }
}
