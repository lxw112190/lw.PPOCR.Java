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

    @Test
    public void nodeProfilesMergeEquivalentResolvedVariantsAndSortByTime() {
        Object model = new Object();
        InferenceProfiler.NodeDescriptor first = new InferenceProfiler.NodeDescriptor(
                model, 7, OperatorType.CONV, "inputs=[[1,3,48,320]],outputs=[[1,8,24,160]]");
        InferenceProfiler.NodeDescriptor same = new InferenceProfiler.NodeDescriptor(
                model, 7, OperatorType.CONV, "inputs=[[1,3,48,320]],outputs=[[1,8,24,160]]");
        InferenceProfiler.NodeDescriptor second = new InferenceProfiler.NodeDescriptor(
                model, 9, OperatorType.MAT_MUL, "inputs=[[1,80,8],[8,16]],outputs=[[1,80,16]]");
        InferenceProfiler.Profile snapshot;
        try (InferenceProfiler profiler = InferenceProfiler.start()) {
            profiler.record(first, 11L);
            profiler.record(same, 17L);
            profiler.record(second, 50L);
            snapshot = profiler.snapshot();
        }

        Assert.assertEquals(2, snapshot.getNodes().size());
        Assert.assertEquals(9, snapshot.getNodes().get(0).getNodeIndex());
        Assert.assertEquals(50L, snapshot.getNodes().get(0).getElapsedNanos());
        Assert.assertEquals(7, snapshot.getNodes().get(1).getNodeIndex());
        Assert.assertEquals(2L, snapshot.getNodes().get(1).getInvocations());
        Assert.assertEquals(28L, snapshot.getNodes().get(1).getElapsedNanos());
    }
}
