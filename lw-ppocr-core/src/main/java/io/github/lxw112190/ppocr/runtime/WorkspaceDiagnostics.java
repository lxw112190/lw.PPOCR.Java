package io.github.lxw112190.ppocr.runtime;

/** Stable memory-planning measurements for one shape-specialized execution. */
public final class WorkspaceDiagnostics {
    private final long oldWorkspaceBytes;
    private final long workspaceBytes;
    private final long liveLowerBoundBytes;

    WorkspaceDiagnostics(long oldWorkspaceBytes, long workspaceBytes,
                         long liveLowerBoundBytes) {
        this.oldWorkspaceBytes = oldWorkspaceBytes;
        this.workspaceBytes = workspaceBytes;
        this.liveLowerBoundBytes = liveLowerBoundBytes;
    }

    public long getOldWorkspaceBytes() { return oldWorkspaceBytes; }
    public long getWorkspaceBytes() { return workspaceBytes; }
    public long getLiveLowerBoundBytes() { return liveLowerBoundBytes; }

    /** Combines measurements from shape-specialized sessions without changing any session state. */
    public static WorkspaceDiagnostics aggregate(WorkspaceDiagnostics... values) {
        long oldBytes = 0L;
        long plannedBytes = 0L;
        long lowerBoundBytes = 0L;
        if (values != null) {
            for (WorkspaceDiagnostics value : values) {
                if (value == null) continue;
                oldBytes = Math.addExact(oldBytes, value.oldWorkspaceBytes);
                plannedBytes = Math.addExact(plannedBytes, value.workspaceBytes);
                lowerBoundBytes = Math.addExact(lowerBoundBytes, value.liveLowerBoundBytes);
            }
        }
        return new WorkspaceDiagnostics(oldBytes, plannedBytes, lowerBoundBytes);
    }

    public double getEfficiency() {
        return workspaceBytes == 0 ? 1.0 : (double) liveLowerBoundBytes / workspaceBytes;
    }

    @Override
    public String toString() {
        return "WorkspaceDiagnostics{oldWorkspaceBytes=" + oldWorkspaceBytes
                + ", workspaceBytes=" + workspaceBytes
                + ", liveLowerBoundBytes=" + liveLowerBoundBytes
                + ", efficiency=" + getEfficiency() + '}';
    }
}
