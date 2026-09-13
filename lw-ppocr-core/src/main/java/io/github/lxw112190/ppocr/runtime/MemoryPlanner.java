package io.github.lxw112190.ppocr.runtime;

import io.github.lxw112190.ppocr.model.NodeInfo;
import io.github.lxw112190.ppocr.model.OcrErrorCode;
import io.github.lxw112190.ppocr.model.OcrException;
import io.github.lxw112190.ppocr.model.TensorInfo;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;

/** Lifetime-based best-fit planner for non-constant tensors. */
public final class MemoryPlanner {
    private static final long ALIGNMENT = 64;

    private MemoryPlanner() { }

    public static WorkspacePlan plan(List<TensorInfo> tensors, List<NodeInfo> nodes,
                                     List<Integer> graphInputs, List<Integer> graphOutputs,
                                     List<TensorShape> shapes) {
        return plan(tensors, nodes, graphInputs, graphOutputs, shapes, true);
    }

    static WorkspacePlan planPartial(List<TensorInfo> tensors, List<NodeInfo> nodes,
                                     List<Integer> graphInputs, List<Integer> graphOutputs,
                                     List<TensorShape> shapes) {
        return plan(tensors, nodes, graphInputs, graphOutputs, shapes, false);
    }

    private static WorkspacePlan plan(List<TensorInfo> tensors, List<NodeInfo> nodes,
                                      List<Integer> graphInputs, List<Integer> graphOutputs,
                                      List<TensorShape> shapes, boolean requireEveryRuntimeTensor) {
        if (tensors == null || nodes == null || graphInputs == null || graphOutputs == null || shapes == null ||
                tensors.size() != shapes.size()) {
            throw new OcrException(OcrErrorCode.INVALID_ARGUMENT, "planner inputs are inconsistent");
        }
        int tensorCount = tensors.size();
        long[] offsets = new long[tensorCount];
        long[] sizes = new long[tensorCount];
        int[] births = new int[tensorCount];
        int[] deaths = new int[tensorCount];
        java.util.Arrays.fill(offsets, -1);
        java.util.Arrays.fill(births, Integer.MAX_VALUE);
        java.util.Arrays.fill(deaths, -1);

        for (int i = 0; i < tensorCount; i++) {
            TensorInfo tensor = tensors.get(i);
            if (tensor.isConstant()) {
                continue;
            }
            long bytes = multiplyExact(shapes.get(i).getElementCount(), tensor.getDataType().getByteSize());
            sizes[i] = bytes;
        }
        for (int input : graphInputs) {
            requireIndex(input, tensorCount);
            births[input] = 0;
        }
        for (int nodeIndex = 0; nodeIndex < nodes.size(); nodeIndex++) {
            NodeInfo node = nodes.get(nodeIndex);
            for (int output : node.getOutputs()) {
                requireIndex(output, tensorCount);
                births[output] = Math.min(births[output], nodeIndex);
            }
            for (int input : node.getInputs()) {
                requireIndex(input, tensorCount);
                deaths[input] = Math.max(deaths[input], nodeIndex);
            }
        }
        for (int output : graphOutputs) {
            requireIndex(output, tensorCount);
            deaths[output] = Math.max(deaths[output], nodes.size());
        }
        for (int i = 0; i < tensorCount; i++) {
            if (requireEveryRuntimeTensor && !tensors.get(i).isConstant() &&
                    births[i] == Integer.MAX_VALUE) {
                throw new OcrException(OcrErrorCode.INVALID_MODEL, "runtime tensor has no producer or graph input: " + i);
            }
            if (!tensors.get(i).isConstant() && deaths[i] < births[i]) {
                deaths[i] = births[i];
            }
        }

        List<Integer> order = new ArrayList<Integer>();
        for (int i = 0; i < tensorCount; i++) {
            if (!tensors.get(i).isConstant() && births[i] != Integer.MAX_VALUE) {
                order.add(i);
            }
        }
        Collections.sort(order, new Comparator<Integer>() {
            @Override
            public int compare(Integer left, Integer right) {
                int result = Integer.compare(births[left], births[right]);
                return result != 0 ? result : Integer.compare(deaths[left], deaths[right]);
            }
        });

        List<Block> active = new ArrayList<Block>();
        List<Block> free = new ArrayList<Block>();
        long totalBytes = 0;
        for (int tensorIndex : order) {
            releaseFinished(active, free, births[tensorIndex]);
            long bytes = sizes[tensorIndex];
            Block selected = bestFit(free, bytes);
            if (selected == null) {
                long start = align(totalBytes);
                totalBytes = addExact(start, bytes);
                selected = new Block(start, bytes, deaths[tensorIndex], tensorIndex);
            } else {
                free.remove(selected);
                long blockEnd = addExact(selected.offset, selected.capacity);
                long remainderOffset = align(addExact(selected.offset, bytes));
                long allocationCapacity = selected.capacity;
                if (remainderOffset < blockEnd) {
                    allocationCapacity = remainderOffset - selected.offset;
                    free.add(new Block(remainderOffset, blockEnd - remainderOffset, -1, -1));
                    coalesceFree(free);
                }
                selected = new Block(selected.offset, allocationCapacity,
                        deaths[tensorIndex], tensorIndex);
            }
            offsets[tensorIndex] = selected.offset;
            active.add(selected);
        }
        return new WorkspacePlan(offsets, sizes, totalBytes);
    }

    private static void releaseFinished(List<Block> active, List<Block> free, int birth) {
        for (int i = active.size() - 1; i >= 0; i--) {
            Block block = active.get(i);
            if (block.death < birth) {
                active.remove(i);
                free.add(new Block(block.offset, block.capacity, -1, -1));
            }
        }
        coalesceFree(free);
    }

    private static void coalesceFree(List<Block> free) {
        if (free.size() < 2) return;
        Collections.sort(free, new Comparator<Block>() {
            @Override
            public int compare(Block left, Block right) {
                return Long.compare(left.offset, right.offset);
            }
        });
        int destination = 0;
        for (int source = 1; source < free.size(); source++) {
            Block previous = free.get(destination);
            Block current = free.get(source);
            if (addExact(previous.offset, previous.capacity) == current.offset) {
                free.set(destination, new Block(previous.offset,
                        addExact(previous.capacity, current.capacity), -1, -1));
            } else {
                destination++;
                free.set(destination, current);
            }
        }
        while (free.size() > destination + 1) free.remove(free.size() - 1);
    }

    private static Block bestFit(List<Block> free, long bytes) {
        Block best = null;
        for (Block block : free) {
            if (block.capacity >= bytes && (best == null || block.capacity < best.capacity)) {
                best = block;
            }
        }
        return best;
    }

    private static long align(long value) {
        long remainder = value % ALIGNMENT;
        return remainder == 0 ? value : addExact(value, ALIGNMENT - remainder);
    }

    private static long multiplyExact(long left, long right) {
        if (left < 0 || right < 0 || (right != 0 && left > Long.MAX_VALUE / right)) {
            throw new OcrException(OcrErrorCode.RESOURCE_LIMIT, "workspace size overflows");
        }
        return left * right;
    }

    private static long addExact(long left, long right) {
        if (left < 0 || right < 0 || left > Long.MAX_VALUE - right) {
            throw new OcrException(OcrErrorCode.RESOURCE_LIMIT, "workspace size overflows");
        }
        return left + right;
    }

    private static void requireIndex(int index, int count) {
        if (index < 0 || index >= count) {
            throw new OcrException(OcrErrorCode.INVALID_MODEL, "tensor index is outside the model");
        }
    }

    private static final class Block {
        final long offset;
        final long capacity;
        final int death;
        final int tensor;

        Block(long offset, long capacity, int death, int tensor) {
            this.offset = offset;
            this.capacity = capacity;
            this.death = death;
            this.tensor = tensor;
        }
    }
}
