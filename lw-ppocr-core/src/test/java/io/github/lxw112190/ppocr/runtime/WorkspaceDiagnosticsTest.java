package io.github.lxw112190.ppocr.runtime;

import org.junit.Assert;
import org.junit.Test;

public final class WorkspaceDiagnosticsTest {
    @Test
    public void aggregatesPlannedAndLowerBoundBytes() {
        WorkspaceDiagnostics combined = WorkspaceDiagnostics.aggregate(
                new WorkspaceDiagnostics(100L, 80L, 60L),
                new WorkspaceDiagnostics(40L, 32L, 24L));

        Assert.assertEquals(140L, combined.getOldWorkspaceBytes());
        Assert.assertEquals(112L, combined.getWorkspaceBytes());
        Assert.assertEquals(84L, combined.getLiveLowerBoundBytes());
        Assert.assertEquals(0.75, combined.getEfficiency(), 0.000001);
    }

    @Test
    public void emptyAggregateHasUnitEfficiency() {
        WorkspaceDiagnostics empty = WorkspaceDiagnostics.aggregate();

        Assert.assertEquals(0L, empty.getWorkspaceBytes());
        Assert.assertEquals(1.0, empty.getEfficiency(), 0.0);
    }
}
