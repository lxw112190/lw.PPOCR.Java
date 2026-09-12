package io.github.lxw112190.ppocr.runtime;

import java.util.Arrays;

/** Immutable byte offsets for runtime-owned tensor storage. */
public final class WorkspacePlan {
    private final long[] offsets;
    private final long[] sizes;
    private final long totalBytes;

    WorkspacePlan(long[] offsets, long[] sizes, long totalBytes) {
        this.offsets = offsets.clone();
        this.sizes = sizes.clone();
        this.totalBytes = totalBytes;
    }

    public long getTotalBytes() { return totalBytes; }
    public long getOffset(int tensorIndex) { return offsets[tensorIndex]; }
    public long getSize(int tensorIndex) { return sizes[tensorIndex]; }
    public boolean isAllocated(int tensorIndex) { return offsets[tensorIndex] >= 0; }
    public int getTensorCount() { return offsets.length; }

    @Override
    public String toString() {
        return "WorkspacePlan{totalBytes=" + totalBytes + ", offsets=" + Arrays.toString(offsets) + "}";
    }
}
