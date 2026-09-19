package io.github.lxw112190.ppocr.ppocr;

import io.github.lxw112190.ppocr.model.OcrErrorCode;
import io.github.lxw112190.ppocr.model.OcrException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.ThreadFactory;

/**
 * Reuses one runnable per worker so a batch does not allocate a FutureTask and
 * Callable for every worker. The executor is intended for one owner at a time.
 */
final class ReusableParallelExecutor implements AutoCloseable {
    interface WorkerAction {
        void run(int workerIndex);
    }

    private final ThreadFactory threadFactory;
    private ExecutorService executor;
    private int executorSize;
    private WorkerRunner[] runners = new WorkerRunner[0];
    private final Batch reusableBatch = new Batch();
    private volatile Batch batch;
    private boolean closed;

    ReusableParallelExecutor(ThreadFactory threadFactory) {
        if (threadFactory == null) {
            throw new OcrException(OcrErrorCode.INVALID_ARGUMENT,
                    "parallel executor thread factory is required");
        }
        this.threadFactory = threadFactory;
    }

    void run(int workerCount, WorkerAction action) {
        if (workerCount <= 0 || action == null) {
            throw new OcrException(OcrErrorCode.INVALID_ARGUMENT,
                    "parallel executor batch is invalid");
        }
        ensureOpen();
        ensureExecutor(workerCount);
        Batch next = reusableBatch;
        next.begin(workerCount, action);
        batch = next;
        int submitted = 0;
        try {
            for (; submitted < workerCount; submitted++) {
                executor.execute(runners[submitted]);
            }
        } catch (RuntimeException e) {
            next.record(e);
            for (int i = submitted; i < workerCount; i++) next.complete();
        }

        try {
            next.await();
            next.throwFailure();
        } finally {
            batch = null;
            next.clear();
        }
    }

    @Override
    public void close() {
        closed = true;
        batch = null;
        if (executor != null) {
            executor.shutdown();
            executor = null;
        }
        executorSize = 0;
        runners = new WorkerRunner[0];
    }

    private void ensureExecutor(int workerCount) {
        if (executor != null && executorSize >= workerCount) return;
        if (executor != null) executor.shutdown();
        executor = Executors.newFixedThreadPool(workerCount, threadFactory);
        executorSize = workerCount;
        runners = new WorkerRunner[workerCount];
        for (int i = 0; i < workerCount; i++) runners[i] = new WorkerRunner(i);
    }

    private void ensureOpen() {
        if (closed) {
            throw new OcrException(OcrErrorCode.INVALID_ARGUMENT,
                    "parallel executor is closed");
        }
    }

    private final class WorkerRunner implements Runnable {
        private final int workerIndex;

        private WorkerRunner(int workerIndex) {
            this.workerIndex = workerIndex;
        }

        @Override
        public void run() {
            Batch current = batch;
            if (current == null) return;
            try {
                current.action.run(workerIndex);
            } catch (Throwable failure) {
                current.record(failure);
            } finally {
                current.complete();
            }
        }
    }

    private static final class Batch {
        private int remaining;
        private WorkerAction action;
        private Throwable failure;

        private synchronized void begin(int workerCount, WorkerAction action) {
            this.remaining = workerCount;
            this.action = action;
            this.failure = null;
        }

        private synchronized void record(Throwable candidate) {
            if (failure == null) failure = candidate;
            else if (failure != candidate) failure.addSuppressed(candidate);
        }

        private void complete() {
            synchronized (this) {
                if (remaining > 0) remaining--;
                if (remaining == 0) notifyAll();
            }
        }

        private void await() {
            boolean interrupted = false;
            synchronized (this) {
                while (remaining != 0) {
                    try {
                        wait();
                    } catch (InterruptedException e) {
                        interrupted = true;
                    }
                }
            }
            if (interrupted) {
                Thread.currentThread().interrupt();
                throw new OcrException(OcrErrorCode.RESOURCE_LIMIT,
                        "parallel OCR execution was interrupted");
            }
        }

        private synchronized void throwFailure() {
            if (failure == null) return;
            if (failure instanceof RuntimeException) throw (RuntimeException) failure;
            throw new OcrException(OcrErrorCode.RESOURCE_LIMIT,
                    "parallel OCR execution failed", failure);
        }

        private synchronized void clear() {
            action = null;
            failure = null;
            remaining = 0;
        }
    }
}
