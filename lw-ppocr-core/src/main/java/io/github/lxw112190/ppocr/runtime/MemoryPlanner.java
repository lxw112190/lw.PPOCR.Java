package io.github.lxw112190.ppocr.runtime;

import io.github.lxw112190.ppocr.model.NodeInfo;
import io.github.lxw112190.ppocr.model.OperatorType;
import io.github.lxw112190.ppocr.model.OcrErrorCode;
import io.github.lxw112190.ppocr.model.OcrException;
import io.github.lxw112190.ppocr.model.TensorInfo;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;

/** Lifetime-based workspace planner with a measurable greedy V2 placement path. */
public final class MemoryPlanner {
    private static final long ALIGNMENT = 64;

    private MemoryPlanner() { }

    public static WorkspacePlan plan(List<TensorInfo> tensors, List<NodeInfo> nodes,
                                     List<Integer> graphInputs, List<Integer> graphOutputs,
                                     List<TensorShape> shapes) {
        return plan(tensors, nodes, graphInputs, graphOutputs, shapes, false);
    }

    static WorkspacePlan plan(List<TensorInfo> tensors, List<NodeInfo> nodes,
                              List<Integer> graphInputs, List<Integer> graphOutputs,
                              List<TensorShape> shapes, boolean fuseGelu) {
        return plan(tensors, nodes, graphInputs, graphOutputs, shapes, fuseGelu, false);
    }

    static WorkspacePlan plan(List<TensorInfo> tensors, List<NodeInfo> nodes,
                              List<Integer> graphInputs, List<Integer> graphOutputs,
                              List<TensorShape> shapes, boolean fuseGelu,
                              boolean aliasElementwise) {
        PlanningData data = prepare(tensors, nodes, graphInputs, graphOutputs, shapes, true);
        Allocation old = allocateLegacy(data);
        if (fuseGelu) markGeluPhantoms(data);
        if (aliasElementwise) markElementwiseAliases(data);
        Allocation current = allocateGreedy(data);
        mapPhantomOffsets(data, current.offsets);
        mapAliasOffsets(data, current.offsets);
        return new WorkspacePlan(current.offsets, data.sizes, current.totalBytes,
                old.totalBytes, liveLowerBound(data));
    }

    static WorkspacePlan planPartial(List<TensorInfo> tensors, List<NodeInfo> nodes,
                                     List<Integer> graphInputs, List<Integer> graphOutputs,
                                     List<TensorShape> shapes) {
        return planPartial(tensors, nodes, graphInputs, graphOutputs, shapes, false);
    }

    static WorkspacePlan planPartial(List<TensorInfo> tensors, List<NodeInfo> nodes,
                                     List<Integer> graphInputs, List<Integer> graphOutputs,
                                     List<TensorShape> shapes, boolean fuseGelu) {
        return planPartial(tensors, nodes, graphInputs, graphOutputs, shapes, fuseGelu, false);
    }

    static WorkspacePlan planPartial(List<TensorInfo> tensors, List<NodeInfo> nodes,
                                     List<Integer> graphInputs, List<Integer> graphOutputs,
                                     List<TensorShape> shapes, boolean fuseGelu,
                                     boolean aliasElementwise) {
        PlanningData data = prepare(tensors, nodes, graphInputs, graphOutputs, shapes, false);
        Allocation old = allocateLegacy(data);
        if (fuseGelu) markGeluPhantoms(data);
        if (aliasElementwise) markElementwiseAliases(data);
        Allocation current = allocateGreedy(data);
        mapPhantomOffsets(data, current.offsets);
        mapAliasOffsets(data, current.offsets);
        return new WorkspacePlan(current.offsets, data.sizes, current.totalBytes,
                old.totalBytes, liveLowerBound(data));
    }

    private static PlanningData prepare(List<TensorInfo> tensors, List<NodeInfo> nodes,
                                        List<Integer> graphInputs, List<Integer> graphOutputs,
                                        List<TensorShape> shapes, boolean requireEveryRuntimeTensor) {
        if (tensors == null || nodes == null || graphInputs == null || graphOutputs == null
                || shapes == null || tensors.size() != shapes.size()) {
            throw new OcrException(OcrErrorCode.INVALID_ARGUMENT, "planner inputs are inconsistent");
        }
        int tensorCount = tensors.size();
        long[] sizes = new long[tensorCount];
        int[] births = new int[tensorCount];
        int[] deaths = new int[tensorCount];
        int[] consumerCounts = new int[tensorCount];
        int[] lastUses = new int[tensorCount];
        Arrays.fill(births, Integer.MAX_VALUE);
        Arrays.fill(deaths, -1);
        Arrays.fill(lastUses, -1);

        for (int i = 0; i < tensorCount; i++) {
            TensorInfo tensor = tensors.get(i);
            if (!tensor.isConstant()) {
                sizes[i] = multiplyExact(shapes.get(i).getElementCount(),
                        tensor.getDataType().getByteSize());
            }
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
                consumerCounts[input]++;
                lastUses[input] = nodeIndex;
            }
        }
        for (int output : graphOutputs) {
            requireIndex(output, tensorCount);
            deaths[output] = Math.max(deaths[output], nodes.size());
            lastUses[output] = nodes.size();
        }
        for (int i = 0; i < tensorCount; i++) {
            if (requireEveryRuntimeTensor && !tensors.get(i).isConstant()
                    && births[i] == Integer.MAX_VALUE) {
                throw new OcrException(OcrErrorCode.INVALID_MODEL,
                        "runtime tensor has no producer or graph input: " + i);
            }
            if (!tensors.get(i).isConstant() && deaths[i] < births[i]) {
                deaths[i] = births[i];
            }
        }
        return new PlanningData(tensors, nodes, shapes, nodes.size(), sizes, births, deaths,
                consumerCounts, lastUses);
    }

    private static Allocation allocateLegacy(PlanningData data) {
        long[] offsets = new long[data.sizes.length];
        Arrays.fill(offsets, -1);
        List<Integer> order = runtimeOrder(data, false);
        List<Block> active = new ArrayList<Block>();
        List<Block> free = new ArrayList<Block>();
        long totalBytes = 0;
        for (int tensorIndex : order) {
            releaseFinished(active, free, data.births[tensorIndex]);
            long bytes = data.sizes[tensorIndex];
            Block selected = bestFit(free, bytes);
            if (selected == null) {
                long start = align(totalBytes);
                totalBytes = addExact(start, bytes);
                selected = new Block(start, bytes, data.deaths[tensorIndex], tensorIndex);
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
                        data.deaths[tensorIndex], tensorIndex);
            }
            offsets[tensorIndex] = selected.offset;
            active.add(selected);
        }
        return new Allocation(offsets, totalBytes);
    }

    private static Allocation allocateGreedy(PlanningData data) {
        long[] offsets = new long[data.sizes.length];
        Arrays.fill(offsets, -1);
        List<Integer> order = runtimeOrder(data, true);
        List<Placement> placed = new ArrayList<Placement>();
        long totalBytes = 0;
        for (int tensorIndex : order) {
            long bytes = data.sizes[tensorIndex];
            long reserved = align(bytes);
            long offset = 0;
            for (;;) {
                long nextOffset = offset;
                boolean collision = false;
                for (Placement other : placed) {
                    if (!overlapsLifetime(data.births[tensorIndex], data.deaths[tensorIndex],
                            other.birth, other.death)) continue;
                    if (!rangesOverlap(offset, reserved, other.offset, other.reserved)) continue;
                    nextOffset = Math.max(nextOffset,
                            align(addExact(other.offset, other.reserved)));
                    collision = true;
                }
                if (!collision) break;
                if (nextOffset <= offset) {
                    throw new OcrException(OcrErrorCode.RESOURCE_LIMIT,
                            "workspace placement does not make progress");
                }
                offset = nextOffset;
            }
            offsets[tensorIndex] = offset;
            totalBytes = Math.max(totalBytes, addExact(offset, bytes));
            placed.add(new Placement(offset, reserved,
                    data.births[tensorIndex], data.deaths[tensorIndex]));
        }
        return new Allocation(offsets, totalBytes);
    }

    private static void markGeluPhantoms(PlanningData data) {
        for (int start = 0; start + 4 < data.nodes.size(); start++) {
            NodeInfo divide = data.nodes.get(start);
            NodeInfo erf = data.nodes.get(start + 1);
            NodeInfo add = data.nodes.get(start + 2);
            NodeInfo multiply = data.nodes.get(start + 3);
            NodeInfo scale = data.nodes.get(start + 4);
            if (divide.getOperator() != OperatorType.DIV
                    || erf.getOperator() != OperatorType.ERF
                    || add.getOperator() != OperatorType.ADD
                    || multiply.getOperator() != OperatorType.MUL
                    || scale.getOperator() != OperatorType.MUL) continue;

            int[] divideInputs = divide.getInputs();
            int[] divideOutputs = divide.getOutputs();
            int[] erfInputs = erf.getInputs();
            int[] erfOutputs = erf.getOutputs();
            int[] addInputs = add.getInputs();
            int[] addOutputs = add.getOutputs();
            int[] multiplyInputs = multiply.getInputs();
            int[] multiplyOutputs = multiply.getOutputs();
            int[] scaleInputs = scale.getInputs();
            int[] scaleOutputs = scale.getOutputs();
            if (divideInputs.length != 2 || divideOutputs.length != 1
                    || erfInputs.length != 1 || erfOutputs.length != 1
                    || addInputs.length != 2 || addOutputs.length != 1
                    || multiplyInputs.length != 2 || multiplyOutputs.length != 1
                    || scaleInputs.length != 2 || scaleOutputs.length != 1) continue;

            int input = divideInputs[0];
            int divideOutput = divideOutputs[0];
            int erfOutput = erfOutputs[0];
            int addOutput = addOutputs[0];
            int multiplyOutput = multiplyOutputs[0];
            int output = scaleOutputs[0];
            if (erfInputs[0] != divideOutput || addInputs[0] != erfOutput
                    || !samePair(multiplyInputs, input, addOutput)
                    || scaleInputs[0] != multiplyOutput
                    || data.consumerCounts[divideOutput] != 1
                    || data.consumerCounts[erfOutput] != 1
                    || data.consumerCounts[addOutput] != 1
                    || data.consumerCounts[multiplyOutput] != 1
                    || data.lastUses[divideOutput] > start + 4
                    || data.lastUses[erfOutput] > start + 4
                    || data.lastUses[addOutput] > start + 4
                    || data.lastUses[multiplyOutput] > start + 4
                    || !sameShape(data, input, divideOutput, erfOutput, addOutput,
                            multiplyOutput, output)
                    || !isScalarConstant(data, divideInputs[1])
                    || !isScalarConstant(data, addInputs[1])
                    || !isScalarConstant(data, scaleInputs[1])) continue;

            for (int node = start; node < start + 4; node++) {
                data.fusedNodes[node] = true;
                for (int intermediate : data.nodes.get(node).getOutputs()) {
                    if (data.lastUses[intermediate] <= start + 4) {
                        data.phantomSink[intermediate] = output;
                        data.sizes[intermediate] = 0;
                    }
                }
            }
            data.fusedNodes[start + 4] = true;
            // The fused kernel writes the final output at the first node's
            // execution time, not at the original scale node. Keep the sink
            // reserved for the whole fused span so it cannot overlap with a
            // tensor that is still read by the fused kernel.
            data.births[output] = Math.min(data.births[output], start);
            start += 4;
        }
    }

    private static void markElementwiseAliases(PlanningData data) {
        for (int nodeIndex = 0; nodeIndex < data.nodes.size(); nodeIndex++) {
            if (data.fusedNodes[nodeIndex]) continue;
            NodeInfo node = data.nodes.get(nodeIndex);
            if (!isAliasableBinary(node.getOperator())) continue;
            int[] inputs = node.getInputs();
            int[] outputs = node.getOutputs();
            if (inputs.length != 2 || outputs.length != 1) continue;
            int output = outputs[0];
            if (!isRuntimeTensor(data, output) || data.sizes[output] == 0
                    || data.aliasSink[output] >= 0) continue;
            for (int input : inputs) {
                if (!isAliasableInput(data, input, output, nodeIndex)) continue;
                int root = aliasRoot(data, input);
                data.aliasSink[output] = root;
                data.sizes[output] = 0;
                data.deaths[root] = Math.max(data.deaths[root], data.deaths[output]);
                break;
            }
        }
    }

    private static boolean isAliasableBinary(OperatorType operator) {
        return operator == OperatorType.ADD || operator == OperatorType.MUL
                || operator == OperatorType.DIV || operator == OperatorType.SUB
                || operator == OperatorType.POW;
    }

    private static boolean isAliasableInput(PlanningData data, int input, int output,
                                            int nodeIndex) {
        return input != output
                && isRuntimeTensor(data, input)
                && data.phantomSink[input] < 0
                && data.consumerCounts[input] == 1
                && data.lastUses[input] == nodeIndex
                && data.shapes.get(input).equals(data.shapes.get(output));
    }

    private static boolean isRuntimeTensor(PlanningData data, int tensor) {
        return tensor >= 0 && tensor < data.tensors.size()
                && !data.tensors.get(tensor).isConstant()
                && data.births[tensor] != Integer.MAX_VALUE;
    }

    private static int aliasRoot(PlanningData data, int tensor) {
        int root = tensor;
        while (data.aliasSink[root] >= 0) root = data.aliasSink[root];
        return root;
    }

    private static void mapPhantomOffsets(PlanningData data, long[] offsets) {
        for (int tensor = 0; tensor < data.phantomSink.length; tensor++) {
            int sink = data.phantomSink[tensor];
            if (sink < 0) continue;
            if (offsets[sink] < 0) {
                throw new OcrException(OcrErrorCode.INVALID_MODEL,
                        "GELU phantom sink has no workspace allocation: " + sink);
            }
            offsets[tensor] = offsets[sink];
        }
    }

    private static void mapAliasOffsets(PlanningData data, long[] offsets) {
        for (int tensor = 0; tensor < data.aliasSink.length; tensor++) {
            int root = data.aliasSink[tensor];
            if (root < 0) continue;
            if (offsets[root] < 0) {
                throw new OcrException(OcrErrorCode.INVALID_MODEL,
                        "elementwise alias root has no workspace allocation: " + root);
            }
            offsets[tensor] = offsets[root];
        }
    }

    private static boolean isScalarConstant(PlanningData data, int tensor) {
        return tensor >= 0 && tensor < data.tensors.size()
                && data.tensors.get(tensor).isConstant()
                && data.shapes.get(tensor).getElementCount() == 1;
    }

    private static boolean sameShape(PlanningData data, int first, int... remaining) {
        TensorShape shape = data.shapes.get(first);
        for (int tensor : remaining) {
            if (!shape.equals(data.shapes.get(tensor))) return false;
        }
        return true;
    }

    private static boolean samePair(int[] values, int first, int second) {
        return values.length == 2 && ((values[0] == first && values[1] == second)
                || (values[0] == second && values[1] == first));
    }

    private static List<Integer> runtimeOrder(final PlanningData data, final boolean bySize) {
        List<Integer> order = new ArrayList<Integer>();
        for (int i = 0; i < data.sizes.length; i++) {
            if (!data.tensors.get(i).isConstant() && data.births[i] != Integer.MAX_VALUE) {
                order.add(i);
            }
        }
        Collections.sort(order, new Comparator<Integer>() {
            @Override
            public int compare(Integer left, Integer right) {
                if (bySize) {
                    int result = Long.compare(data.sizes[right], data.sizes[left]);
                    if (result != 0) return result;
                }
                int result = Integer.compare(data.births[left], data.births[right]);
                if (result != 0) return result;
                result = Integer.compare(data.deaths[left], data.deaths[right]);
                return result != 0 ? result : Integer.compare(left, right);
            }
        });
        return order;
    }

    private static long liveLowerBound(PlanningData data) {
        long maximum = 0;
        for (int time = 0; time <= data.nodeCount; time++) {
            long live = 0;
            for (int i = 0; i < data.sizes.length; i++) {
                if (data.tensors.get(i).isConstant() || data.births[i] == Integer.MAX_VALUE) continue;
                if (data.births[i] <= time && data.deaths[i] >= time) {
                    live = addExact(live, data.sizes[i]);
                }
            }
            maximum = Math.max(maximum, live);
        }
        return maximum;
    }

    private static boolean overlapsLifetime(int leftBirth, int leftDeath,
                                             int rightBirth, int rightDeath) {
        return leftBirth <= rightDeath && rightBirth <= leftDeath;
    }

    private static boolean rangesOverlap(long leftOffset, long leftSize,
                                         long rightOffset, long rightSize) {
        return leftSize > 0 && rightSize > 0
                && leftOffset < addExact(rightOffset, rightSize)
                && rightOffset < addExact(leftOffset, leftSize);
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

    private static final class PlanningData {
        final List<TensorInfo> tensors;
        final List<NodeInfo> nodes;
        final List<TensorShape> shapes;
        final int nodeCount;
        final long[] sizes;
        final int[] births;
        final int[] deaths;
        final int[] consumerCounts;
        final int[] lastUses;
        final int[] phantomSink;
        final int[] aliasSink;
        final boolean[] fusedNodes;

        PlanningData(List<TensorInfo> tensors, List<NodeInfo> nodes,
                     List<TensorShape> shapes, int nodeCount, long[] sizes,
                     int[] births, int[] deaths, int[] consumerCounts, int[] lastUses) {
            this.tensors = tensors;
            this.nodes = nodes;
            this.shapes = shapes;
            this.nodeCount = nodeCount;
            this.sizes = sizes;
            this.births = births;
            this.deaths = deaths;
            this.consumerCounts = consumerCounts;
            this.lastUses = lastUses;
            this.phantomSink = new int[tensors.size()];
            Arrays.fill(this.phantomSink, -1);
            this.aliasSink = new int[tensors.size()];
            Arrays.fill(this.aliasSink, -1);
            this.fusedNodes = new boolean[nodes.size()];
        }
    }

    private static final class Allocation {
        final long[] offsets;
        final long totalBytes;

        Allocation(long[] offsets, long totalBytes) {
            this.offsets = offsets;
            this.totalBytes = totalBytes;
        }
    }

    private static final class Placement {
        final long offset;
        final long reserved;
        final int birth;
        final int death;

        Placement(long offset, long reserved, int birth, int death) {
            this.offset = offset;
            this.reserved = reserved;
            this.birth = birth;
            this.death = death;
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
