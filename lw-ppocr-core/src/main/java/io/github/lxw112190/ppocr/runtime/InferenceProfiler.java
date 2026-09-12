package io.github.lxw112190.ppocr.runtime;

import io.github.lxw112190.ppocr.model.OperatorType;

/** Optional thread-local operator timing for diagnostics and benchmarks. */
public final class InferenceProfiler implements AutoCloseable {
    private static final ThreadLocal<InferenceProfiler> CURRENT = new ThreadLocal<InferenceProfiler>();

    private final Thread owner = Thread.currentThread();
    private final long[] elapsedNanos = new long[OperatorType.values().length];
    private final long[] invocations = new long[OperatorType.values().length];
    private boolean closed;

    private InferenceProfiler() { }

    /** Starts profiling inference executed by the current thread. Nested scopes are rejected. */
    public static InferenceProfiler start() {
        if (CURRENT.get() != null) throw new IllegalStateException("inference profiling is already active");
        InferenceProfiler profiler = new InferenceProfiler();
        CURRENT.set(profiler);
        return profiler;
    }

    /** Returns an immutable copy of all timings collected so far. */
    public Profile snapshot() {
        ensureOwner();
        return new Profile(elapsedNanos.clone(), invocations.clone());
    }

    @Override
    public void close() {
        ensureOwner();
        if (!closed) {
            closed = true;
            CURRENT.remove();
        }
    }

    static InferenceProfiler current() {
        return CURRENT.get();
    }

    void record(OperatorType operator, long nanos) {
        int index = operator.ordinal();
        elapsedNanos[index] += Math.max(0L, nanos);
        invocations[index]++;
    }

    private void ensureOwner() {
        if (Thread.currentThread() != owner) {
            throw new IllegalStateException("inference profiler must be used by its owning thread");
        }
    }

    /** Immutable aggregate grouped by LWM operator type. */
    public static final class Profile {
        private final long[] elapsedNanos;
        private final long[] invocations;

        private Profile(long[] elapsedNanos, long[] invocations) {
            this.elapsedNanos = elapsedNanos;
            this.invocations = invocations;
        }

        public long getElapsedNanos(OperatorType operator) {
            return elapsedNanos[operator.ordinal()];
        }

        public long getInvocations(OperatorType operator) {
            return invocations[operator.ordinal()];
        }

        public long getTotalElapsedNanos() {
            long total = 0L;
            for (long value : elapsedNanos) total += value;
            return total;
        }
    }
}
