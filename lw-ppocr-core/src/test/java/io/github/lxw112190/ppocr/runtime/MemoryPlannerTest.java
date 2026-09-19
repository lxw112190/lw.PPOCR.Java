package io.github.lxw112190.ppocr.runtime;

import io.github.lxw112190.ppocr.model.DataType;
import io.github.lxw112190.ppocr.model.NodeInfo;
import io.github.lxw112190.ppocr.model.OperatorType;
import io.github.lxw112190.ppocr.model.TensorInfo;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import org.junit.Assert;
import org.junit.Test;

public final class MemoryPlannerTest {
    @Test
    public void reusesBlocksAfterLastConsumer() {
        List<TensorInfo> tensors = Arrays.asList(
                tensor(2), tensor(0), tensor(0), tensor(4), constant());
        List<NodeInfo> nodes = Arrays.asList(
                new NodeInfo(OperatorType.ADD, new int[] {0, 1}, new int[] {2}, 0, 0),
                new NodeInfo(OperatorType.ADD, new int[] {2, 1}, new int[] {3}, 0, 0));
        List<TensorShape> shapes = Arrays.asList(
                TensorShape.of(4), TensorShape.of(4), TensorShape.of(4),
                TensorShape.of(4), TensorShape.of(1));

        WorkspacePlan plan = MemoryPlanner.plan(tensors, nodes,
                Arrays.asList(0, 1), Collections.singletonList(3), shapes);

        Assert.assertEquals(144L, plan.getTotalBytes());
        Assert.assertEquals(plan.getOffset(0), plan.getOffset(3));
        Assert.assertFalse(plan.isAllocated(4));
        Assert.assertEquals(16L, plan.getSize(0));
    }

    @Test
    public void splitsAndCoalescesReusedBlocks() {
        List<TensorInfo> tensors = Arrays.asList(
                tensor(2), tensor(0), tensor(0), tensor(4), tensor(4));
        List<NodeInfo> nodes = Arrays.asList(
                new NodeInfo(OperatorType.ADD, new int[] {0}, new int[] {2}, 0, 0),
                new NodeInfo(OperatorType.ADD, new int[] {2}, new int[] {3, 4}, 0, 0));
        List<TensorShape> shapes = Arrays.asList(
                TensorShape.of(32), TensorShape.of(1), TensorShape.of(4),
                TensorShape.of(4), TensorShape.of(4));

        WorkspacePlan plan = MemoryPlanner.plan(tensors, nodes,
                Arrays.asList(0, 1), Arrays.asList(3, 4), shapes);

        Assert.assertEquals(196L, plan.getTotalBytes());
        Assert.assertEquals(0L, plan.getOffset(3));
        Assert.assertEquals(64L, plan.getOffset(4));
        Assert.assertEquals(208L, plan.getDiagnostics().getOldWorkspaceBytes());
        Assert.assertEquals(148L, plan.getDiagnostics().getLiveLowerBoundBytes());
    }

    @Test
    public void fusedGeluOutputIsReservedForTheWholeFusedSpan() {
        List<TensorInfo> tensors = Arrays.asList(
                tensor(TensorInfo.INPUT, 4), constant(1), runtime(4), constant(1),
                runtime(4), runtime(4), runtime(4), constant(1), runtime(4));
        List<NodeInfo> nodes = Arrays.asList(
                new NodeInfo(OperatorType.DIV, new int[] {0, 1}, new int[] {2}, 0, 0),
                new NodeInfo(OperatorType.ERF, new int[] {2}, new int[] {4}, 0, 0),
                new NodeInfo(OperatorType.ADD, new int[] {4, 3}, new int[] {5}, 0, 0),
                new NodeInfo(OperatorType.MUL, new int[] {0, 5}, new int[] {6}, 0, 0),
                new NodeInfo(OperatorType.MUL, new int[] {6, 7}, new int[] {8}, 0, 0));
        List<TensorShape> shapes = Arrays.asList(
                TensorShape.of(4), TensorShape.of(1), TensorShape.of(4), TensorShape.of(1),
                TensorShape.of(4), TensorShape.of(4), TensorShape.of(4), TensorShape.of(1),
                TensorShape.of(4));

        WorkspacePlan plan = MemoryPlanner.plan(tensors, nodes,
                Collections.singletonList(0), Collections.singletonList(8), shapes, true);

        Assert.assertNotEquals("fused output must not alias its input: " + plan,
                plan.getOffset(0), plan.getOffset(8));
        Assert.assertEquals(plan.getOffset(8), plan.getOffset(2));
        Assert.assertEquals(plan.getOffset(8), plan.getOffset(4));
        Assert.assertEquals(plan.getOffset(8), plan.getOffset(5));
        Assert.assertEquals(plan.getOffset(8), plan.getOffset(6));
    }

    @Test
    public void aliasesSameShapeElementwiseOutputToSingleUseInput() {
        List<TensorInfo> tensors = Arrays.asList(
                tensor(TensorInfo.INPUT, 4), constant(1), runtime(4));
        List<NodeInfo> nodes = Collections.singletonList(
                new NodeInfo(OperatorType.ADD, new int[] {0, 1}, new int[] {2}, 0, 0));
        List<TensorShape> shapes = Arrays.asList(
                TensorShape.of(4), TensorShape.of(1), TensorShape.of(4));

        WorkspacePlan plan = MemoryPlanner.plan(tensors, nodes,
                Collections.singletonList(0), Collections.singletonList(2), shapes,
                false, true);

        Assert.assertEquals(plan.getOffset(0), plan.getOffset(2));
        Assert.assertEquals(0L, plan.getSize(2));
        Assert.assertEquals(16L, plan.getTotalBytes());
    }

    @Test
    public void doesNotAliasInputThatHasAnotherConsumer() {
        List<TensorInfo> tensors = Arrays.asList(
                tensor(TensorInfo.INPUT, 4), constant(1), runtime(4), runtime(4));
        List<NodeInfo> nodes = Arrays.asList(
                new NodeInfo(OperatorType.ADD, new int[] {0, 1}, new int[] {2}, 0, 0),
                new NodeInfo(OperatorType.ADD, new int[] {0, 2}, new int[] {3}, 0, 0));
        List<TensorShape> shapes = Arrays.asList(
                TensorShape.of(4), TensorShape.of(1), TensorShape.of(4), TensorShape.of(4));

        WorkspacePlan plan = MemoryPlanner.plan(tensors, nodes,
                Collections.singletonList(0), Collections.singletonList(3), shapes,
                false, true);

        Assert.assertNotEquals(plan.getOffset(0), plan.getOffset(2));
    }

    @Test
    public void chainsElementwiseAliasesIntoOneStorageGroup() {
        List<TensorInfo> tensors = Arrays.asList(
                tensor(TensorInfo.INPUT, 4), constant(1), runtime(4), runtime(4));
        List<NodeInfo> nodes = Arrays.asList(
                new NodeInfo(OperatorType.ADD, new int[] {0, 1}, new int[] {2}, 0, 0),
                new NodeInfo(OperatorType.MUL, new int[] {2, 1}, new int[] {3}, 0, 0));
        List<TensorShape> shapes = Arrays.asList(
                TensorShape.of(4), TensorShape.of(1), TensorShape.of(4), TensorShape.of(4));

        WorkspacePlan plan = MemoryPlanner.plan(tensors, nodes,
                Collections.singletonList(0), Collections.singletonList(3), shapes,
                false, true);

        Assert.assertEquals(plan.getOffset(0), plan.getOffset(2));
        Assert.assertEquals(plan.getOffset(0), plan.getOffset(3));
        Assert.assertEquals(16L, plan.getTotalBytes());
    }

    @Test
    public void aliasesShapeOnlyOperatorsIntoOneStorageGroup() {
        List<TensorInfo> tensors = Arrays.asList(
                tensor(TensorInfo.INPUT, 4), runtime(4), runtime(4));
        List<NodeInfo> nodes = Arrays.asList(
                new NodeInfo(OperatorType.SQUEEZE, new int[] {0}, new int[] {1}, 0, 0),
                new NodeInfo(OperatorType.RESHAPE, new int[] {1}, new int[] {2}, 0, 0));
        List<TensorShape> shapes = Arrays.asList(
                TensorShape.of(1, 4), TensorShape.of(4), TensorShape.of(2, 2));

        WorkspacePlan plan = MemoryPlanner.plan(tensors, nodes,
                Collections.singletonList(0), Collections.singletonList(2), shapes,
                false, true);

        Assert.assertEquals(plan.getOffset(0), plan.getOffset(1));
        Assert.assertEquals(plan.getOffset(0), plan.getOffset(2));
        Assert.assertEquals(0L, plan.getSize(1));
        Assert.assertEquals(0L, plan.getSize(2));
        Assert.assertEquals(16L, plan.getTotalBytes());
    }

    @Test
    public void aliasesContiguousConcatInputsIntoOutputSlices() {
        List<TensorInfo> tensors = Arrays.asList(
                tensor(TensorInfo.INPUT, 2), tensor(TensorInfo.INPUT, 3), runtime(5));
        List<NodeInfo> nodes = Collections.singletonList(
                new NodeInfo(OperatorType.CONCAT, new int[] {0, 1}, new int[] {2}, 0, 0));
        List<TensorShape> shapes = Arrays.asList(
                TensorShape.of(2), TensorShape.of(3), TensorShape.of(5));

        WorkspacePlan plan = MemoryPlanner.plan(tensors, nodes,
                Arrays.asList(0, 1), Collections.singletonList(2), shapes,
                false, true, new int[] {0});

        Assert.assertEquals(0L, plan.getOffset(0));
        Assert.assertEquals(8L, plan.getOffset(1));
        Assert.assertEquals(0L, plan.getOffset(2));
        Assert.assertEquals(0L, plan.getSize(0));
        Assert.assertEquals(0L, plan.getSize(1));
        Assert.assertEquals(20L, plan.getTotalBytes());
    }

    @Test
    public void doesNotAliasInterleavedConcatSlices() {
        List<TensorInfo> tensors = Arrays.asList(
                shaped(TensorInfo.INPUT, 2, 2), shaped(TensorInfo.INPUT, 2, 3),
                shaped(0, 2, 5));
        List<NodeInfo> nodes = Collections.singletonList(
                new NodeInfo(OperatorType.CONCAT, new int[] {0, 1}, new int[] {2}, 0, 0));
        List<TensorShape> shapes = Arrays.asList(
                TensorShape.of(2, 2), TensorShape.of(2, 3), TensorShape.of(2, 5));

        WorkspacePlan plan = MemoryPlanner.plan(tensors, nodes,
                Arrays.asList(0, 1), Collections.singletonList(2), shapes,
                false, true, new int[] {1});

        Assert.assertEquals(16L, plan.getSize(0));
        Assert.assertEquals(24L, plan.getSize(1));
    }

    @Test(expected = IllegalArgumentException.class)
    public void rejectsInvalidShape() {
        TensorShape.of(4, 0);
    }

    private static TensorInfo tensor(int flags) {
        return tensor(flags, 4);
    }

    private static TensorInfo runtime(int elements) {
        return tensor(0, elements);
    }

    private static TensorInfo tensor(int flags, int elements) {
        return new TensorInfo(DataType.F32, new int[] {elements}, flags, 0, 0, -1, 0);
    }

    private static TensorInfo shaped(int flags, int... dimensions) {
        return new TensorInfo(DataType.F32, dimensions, flags, 0, 0, -1, 0);
    }

    private static TensorInfo constant() {
        return constant(1);
    }

    private static TensorInfo constant(int elements) {
        return new TensorInfo(DataType.F32, new int[] {elements}, TensorInfo.CONSTANT,
                128, elements * 4L, -1, 0);
    }
}
