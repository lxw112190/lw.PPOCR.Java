package io.github.lxw112190.ppocr.runtime;

import io.github.lxw112190.ppocr.model.OperatorType;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;
import java.util.concurrent.ConcurrentHashMap;
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
    private final ConcurrentHashMap<NodeDescriptor, NodeCounters> nodes =
            new ConcurrentHashMap<NodeDescriptor, NodeCounters>();
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
        List<NodeProfile> nodeProfiles = new ArrayList<NodeProfile>(nodes.size());
        for (NodeCounters counters : nodes.values()) {
            nodeProfiles.add(new NodeProfile(counters.descriptor.nodeIndex,
                    counters.descriptor.operator, counters.descriptor.description,
                    counters.elapsedNanos.get(), counters.invocations.get()));
        }
        Collections.sort(nodeProfiles, NodeProfile.HOTTEST_FIRST);
        return new Profile(elapsed, calls, nodeProfiles);
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
        recordAggregate(operator, nanos);
    }

    void record(NodeDescriptor descriptor, long nanos) {
        recordAggregate(descriptor.operator, nanos);
        NodeCounters counters = nodes.get(descriptor);
        if (counters == null) {
            NodeCounters created = new NodeCounters(descriptor);
            NodeCounters existing = nodes.putIfAbsent(descriptor, created);
            counters = existing == null ? created : existing;
        }
        counters.elapsedNanos.addAndGet(Math.max(0L, nanos));
        counters.invocations.incrementAndGet();
    }

    private void recordAggregate(OperatorType operator, long nanos) {
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
        private final List<NodeProfile> nodes;

        private Profile(long[] elapsedNanos, long[] invocations, List<NodeProfile> nodes) {
            this.elapsedNanos = elapsedNanos;
            this.invocations = invocations;
            this.nodes = Collections.unmodifiableList(nodes);
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

        /** Returns executed graph nodes ordered by descending summed thread time. */
        public List<NodeProfile> getNodes() { return nodes; }
    }

    /** Immutable timing for one model node and resolved shape variant. */
    public static final class NodeProfile {
        private static final Comparator<NodeProfile> HOTTEST_FIRST = new Comparator<NodeProfile>() {
            @Override
            public int compare(NodeProfile left, NodeProfile right) {
                int elapsed = Long.compare(right.elapsedNanos, left.elapsedNanos);
                if (elapsed != 0) return elapsed;
                return Integer.compare(left.nodeIndex, right.nodeIndex);
            }
        };

        private final int nodeIndex;
        private final OperatorType operator;
        private final String description;
        private final long elapsedNanos;
        private final long invocations;

        private NodeProfile(int nodeIndex, OperatorType operator, String description,
                            long elapsedNanos, long invocations) {
            this.nodeIndex = nodeIndex;
            this.operator = operator;
            this.description = description;
            this.elapsedNanos = elapsedNanos;
            this.invocations = invocations;
        }

        public int getNodeIndex() { return nodeIndex; }
        public OperatorType getOperator() { return operator; }
        public String getDescription() { return description; }
        public long getElapsedNanos() { return elapsedNanos; }
        public long getInvocations() { return invocations; }
    }

    static final class NodeDescriptor {
        private final Object modelIdentity;
        private final int nodeIndex;
        private final OperatorType operator;
        private final String description;
        private final int hashCode;

        NodeDescriptor(Object modelIdentity, int nodeIndex, OperatorType operator,
                       String description) {
            this.modelIdentity = modelIdentity;
            this.nodeIndex = nodeIndex;
            this.operator = operator;
            this.description = description;
            this.hashCode = 31 * (31 * System.identityHashCode(modelIdentity) + nodeIndex)
                    + description.hashCode();
        }

        @Override
        public boolean equals(Object other) {
            if (!(other instanceof NodeDescriptor)) return false;
            NodeDescriptor descriptor = (NodeDescriptor) other;
            return modelIdentity == descriptor.modelIdentity && nodeIndex == descriptor.nodeIndex
                    && operator == descriptor.operator && description.equals(descriptor.description);
        }

        @Override
        public int hashCode() { return hashCode; }
    }

    private static final class NodeCounters {
        private final NodeDescriptor descriptor;
        private final java.util.concurrent.atomic.AtomicLong elapsedNanos =
                new java.util.concurrent.atomic.AtomicLong();
        private final java.util.concurrent.atomic.AtomicLong invocations =
                new java.util.concurrent.atomic.AtomicLong();

        private NodeCounters(NodeDescriptor descriptor) {
            this.descriptor = descriptor;
        }
    }
}
