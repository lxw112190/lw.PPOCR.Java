package io.github.lxw112190.ppocr.runtime;

import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.ThreadFactory;
import java.util.concurrent.atomic.AtomicInteger;

/** Reusable session jobs; callers participate and pool jobs never submit nested work. */
final class OperatorParallelExecutor {
    interface Action { void run(int shard, int shards); }
    private static final class Pool {
        static final ExecutorService INSTANCE = Executors.newFixedThreadPool(
                Math.max(1, Math.min(8, Runtime.getRuntime().availableProcessors()) - 1),
                new ThreadFactory() {
                    private final AtomicInteger ids = new AtomicInteger();
                    public Thread newThread(Runnable task) {
                        Thread t = new Thread(task, "ppocr-op-" + ids.incrementAndGet());
                        t.setDaemon(true);
                        return t;
                    }
                });
    }
    private final Runnable[] runners = new Runnable[7];
    private Action action;
    private int shards, remaining;
    private Throwable failure;

    OperatorParallelExecutor() {
        for (int i = 0; i < runners.length; i++) {
            final int shard = i + 1;
            runners[i] = new Runnable() { public void run() { execute(shard); } };
        }
    }
    void run(int count, Action next) {
        synchronized (this) { action = next; shards = count; remaining = count; failure = null; }
        int submitted = 1;
        try {
            for (; submitted < count; submitted++) Pool.INSTANCE.execute(runners[submitted - 1]);
        } catch (Throwable e) {
            synchronized (this) { failure = e; remaining -= count - submitted; }
        }
        execute(0);
        boolean interrupted = false;
        synchronized (this) {
            while (remaining != 0) {
                try { wait(); } catch (InterruptedException e) { interrupted = true; }
            }
            action = null;
            if (interrupted) Thread.currentThread().interrupt();
            if (failure instanceof Error) throw (Error) failure;
            if (failure instanceof RuntimeException) throw (RuntimeException) failure;
            if (failure != null) throw new IllegalStateException("parallel operator failed", failure);
            if (interrupted) throw new IllegalStateException("parallel operator interrupted");
        }
    }
    private void execute(int shard) {
        try { action.run(shard, shards); }
        catch (Throwable e) { synchronized (this) { if (failure == null) failure = e; } }
        finally { synchronized (this) { if (--remaining == 0) notifyAll(); } }
    }
}
