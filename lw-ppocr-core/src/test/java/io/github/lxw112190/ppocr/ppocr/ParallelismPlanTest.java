package io.github.lxw112190.ppocr.ppocr;

import org.junit.Assert;
import org.junit.Test;

public final class ParallelismPlanTest {
    @Test
    public void followsAutomaticCpuBudgetMatrix() {
        assertAuto(1, 1, 1);
        assertAuto(2, 2, 1);
        assertAuto(4, 4, 1);
        assertAuto(8, 4, 2);
        assertAuto(16, 4, 4);
        assertAuto(32, 4, 4);
    }

    @Test
    public void clampsWorkersToAvailableLinesAndCpuBudget() {
        ParallelismPlan plan = ParallelismPlan.automatic(16, 2);
        Assert.assertEquals(2, plan.getLineWorkers());
        Assert.assertEquals(2, plan.getClassifierWorkers());
        Assert.assertEquals(2, plan.getRecognizerWorkers());
        Assert.assertTrue(plan.getLineWorkers() * plan.getRecognizerIntraOp()
                <= plan.getAvailableProcessors());
    }

    @Test
    public void preservesManualWorkerLimits() {
        ParallelismPlan plan = ParallelismPlan.manual(2, 3, 4, 2);
        Assert.assertEquals(3, plan.getClassifierWorkers());
        Assert.assertEquals(2, plan.getRecognizerWorkers());
        Assert.assertEquals(1, plan.getRecognizerIntraOp());
    }

    private static void assertAuto(int processors, int lineWorkers, int recIntraOp) {
        ParallelismPlan plan = ParallelismPlan.automatic(processors, 16);
        Assert.assertEquals(processors, plan.getAvailableProcessors());
        Assert.assertEquals(lineWorkers, plan.getLineWorkers());
        Assert.assertEquals(lineWorkers, plan.getClassifierWorkers());
        Assert.assertEquals(lineWorkers, plan.getRecognizerWorkers());
        Assert.assertEquals(recIntraOp, plan.getRecognizerIntraOp());
        Assert.assertTrue(plan.getLineWorkers() * plan.getRecognizerIntraOp() <= processors);
    }
}
