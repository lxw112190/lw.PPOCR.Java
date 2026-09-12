package io.github.lxw112190.ppocr.runtime;

import io.github.lxw112190.ppocr.model.OperatorType;
import java.util.concurrent.atomic.AtomicLongArray;
import java.util.concurrent.atomic.AtomicReference;

/** Optional thread-local or process-wide operator timing for diagnostics and benchmarks. */
public final class InferenceProfiler implements AutoCloseable {
    private static final ThreadLocal<InferenceProfiler> CURRENT = new ThreadLocal<InferenceProfiler>();
    private static final AtomicReference<InferenceProfiler> SHARED =
            new AtomicReference<InferenceProfiler>();

    private final Thread owner = Thread.currentThread();
    private final AtomicLongArray elapsedNanos = new AtomicLongArray(OperatorType.values().length);
    private final AtomicLongArray invocations = new AtomicLongArray(OperatorType.values().length);
    private final boolean shared;
    private boolean closed;

    private InferenceProfiler(boolean shared) {
        this.shared = shared;
    }

    /** Starts profiling inference executed by the current thread. Nested scopes are rejected. */
    public static InferenceProfiler start() {
        if (CURRENT.get() != null) throw new IllegalStateException("inference profiling is already active");
        InferenceProfiler profiler = new InferenceProfiler(false);
        CURRENT.set(profiler);
        return profiler;
    }

    /** Starts one process-wide scope so worker-thread inference is included in the snapshot. */
    public static InferenceProfiler startShared() {
        if (CURRENT.get() != null) throw new IllegalStateException("inference profiling is already active");
        InferenceProfiler profiler = new InferenceProfiler(true);
        if (!SHARED.compareAndSet(null, profiler)) {
            throw new IllegalStateException("shared inference profiling is already active");
        }
        CURRENT.set(profiler);
        return profiler;
    }

    /** Returns an immutable copy of all timings collected so far. */
    public Profile snapshot() {
        ensureOwner();
        long[] elapsed = new long[elapsedNanos.length()];
        long[] calls = new long[invocations.length()];
        for (int i = 0; i < elapsed.length; i++) {
            elapsed[i] = elapsedNanos.get(i);
            calls[i] = invocations.get(i);
        }
        return new Profile(elapsed, calls);
    }

    @Override
    public void close() {
        ensureOwner();
        if (!closed) {
            closed = true;
            CURRENT.remove();
            if (shared) SHARED.compareAndSet(this, null);
        }
    }

    static InferenceProfiler current() {
        InferenceProfiler profiler = CURRENT.get();
        return profiler == null ? SHARED.get() : profiler;
    }

    void record(OperatorType operator, long nanos) {
        int index = operator.ordinal();
        elapsedNanos.addAndGet(index, Math.max(0L, nanos));
        invocations.incrementAndGet(index);
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
