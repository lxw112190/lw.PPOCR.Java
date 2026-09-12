package io.github.lxw112190.ppocr.model;

/** Bounds used while reading an untrusted LWM file. */
public final class RuntimeLimits {
    private final long maxModelFileSize;
    private final long maxTensorCount;
    private final long maxNodeCount;
    private final int maxRank;

    public RuntimeLimits(long maxModelFileSize, long maxTensorCount,
                         long maxNodeCount, int maxRank) {
        if (maxModelFileSize <= 0 || maxTensorCount <= 0 || maxNodeCount <= 0 || maxRank <= 0) {
            throw new IllegalArgumentException("runtime limits must be positive");
        }
        this.maxModelFileSize = maxModelFileSize;
        this.maxTensorCount = maxTensorCount;
        this.maxNodeCount = maxNodeCount;
        this.maxRank = maxRank;
    }

    public static RuntimeLimits defaults() {
        return new RuntimeLimits(1024L * 1024L * 1024L, 1_000_000L, 1_000_000L, 8);
    }

    public long getMaxModelFileSize() {
        return maxModelFileSize;
    }

    public long getMaxTensorCount() {
        return maxTensorCount;
    }

    public long getMaxNodeCount() {
        return maxNodeCount;
    }

    public int getMaxRank() {
        return maxRank;
    }
}
