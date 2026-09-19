package io.github.lxw112190.ppocr.ppocr;

import java.util.concurrent.ThreadFactory;
import java.util.concurrent.atomic.AtomicInteger;
import org.junit.Assert;
import org.junit.Test;

public class ReusableParallelExecutorTest {
    @Test
    public void reusesWorkerRunnablesAcrossBatches() {
        final AtomicInteger threadIds = new AtomicInteger();
        ThreadFactory factory = new ThreadFactory() {
            @Override
            public Thread newThread(Runnable runnable) {
                return new Thread(runnable, "test-worker-" + threadIds.incrementAndGet());
            }
        };
        final int[] counts = new int[3];
        ReusableParallelExecutor executor = new ReusableParallelExecutor(factory);
        try {
            executor.run(3, new ReusableParallelExecutor.WorkerAction() {
                @Override
                public void run(int workerIndex) {
                    counts[workerIndex]++;
                }
            });
            executor.run(2, new ReusableParallelExecutor.WorkerAction() {
                @Override
                public void run(int workerIndex) {
                    counts[workerIndex]++;
                }
            });
        } finally {
            executor.close();
        }
        Assert.assertArrayEquals(new int[] {2, 2, 1}, counts);
        Assert.assertEquals(3, threadIds.get());
    }

    @Test
    public void reusableBatchRecoversAfterWorkerFailure() {
        ThreadFactory factory = new ThreadFactory() {
            @Override
            public Thread newThread(Runnable runnable) {
                return new Thread(runnable, "failure-worker");
            }
        };
        final AtomicInteger successfulCalls = new AtomicInteger();
        ReusableParallelExecutor executor = new ReusableParallelExecutor(factory);
        try {
            try {
                executor.run(2, new ReusableParallelExecutor.WorkerAction() {
                    @Override
                    public void run(int workerIndex) {
                        if (workerIndex == 0) throw new IllegalStateException("expected");
                    }
                });
                Assert.fail("worker failure should be propagated");
            } catch (IllegalStateException expected) {
                // The reusable batch must be left in a completed state.
            }
            executor.run(2, new ReusableParallelExecutor.WorkerAction() {
                @Override
                public void run(int workerIndex) {
                    successfulCalls.incrementAndGet();
                }
            });
        } finally {
            executor.close();
        }
        Assert.assertEquals(2, successfulCalls.get());
    }
}
