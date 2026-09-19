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
