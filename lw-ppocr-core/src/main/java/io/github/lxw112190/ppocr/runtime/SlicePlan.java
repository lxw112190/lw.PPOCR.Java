package io.github.lxw112190.ppocr.runtime;

import io.github.lxw112190.ppocr.model.OcrErrorCode;
import io.github.lxw112190.ppocr.model.OcrException;
import io.github.lxw112190.ppocr.model.OperatorType;
import java.nio.ByteBuffer;

/** Prepared bounded Slice. Output extents come from starts AND ends, not the input tail. */
final class SlicePlan {
    private final int[] inputStrides, outputStrides, dimensions, starts, steps;
    private final int length, last;
    SlicePlan(TensorShape input, TensorShape output, ByteBuffer p) {
        TensorShape inferred = ShapeResolver.resolveNode(OperatorType.SLICE,
                new TensorShape[] {input}, new int[0], p);
        if (!inferred.equals(output)) throw new OcrException(OcrErrorCode.INVALID_MODEL, "Slice output shape mismatch");
        dimensions = output.getDimensions(); last = dimensions.length - 1;
        inputStrides = strides(input.getDimensions()); outputStrides = strides(dimensions);
        length = Math.toIntExact(output.getElementCount()); starts = new int[dimensions.length]; steps = new int[dimensions.length];
        java.util.Arrays.fill(steps, 1);
        for (int i = 0; i < Short.toUnsignedInt(p.getShort(2)); i++) {
            int axis = p.getInt(68 + i * 4); if (axis < 0) axis += dimensions.length;
            long start = p.getInt(4 + i * 4); if (start < 0) start += input.get(axis);
            starts[axis] = (int) Math.max(0, Math.min(input.get(axis), start));
            steps[axis] = p.getInt(100 + i * 4);
        }
    }
    void run(float[] input, int inputOffset, float[] output, int outputOffset) {
        int block = steps[last] == 1 ? dimensions[last] : 1;
        for (int linear = 0; linear < length; linear += block) {
            int remainder = linear, source = inputOffset;
            for (int axis = 0; axis < dimensions.length; axis++) {
                int coordinate = remainder / outputStrides[axis]; remainder %= outputStrides[axis];
                source += (starts[axis] + coordinate * steps[axis]) * inputStrides[axis];
            }
            if (block == 1) output[outputOffset + linear] = input[source];
            else System.arraycopy(input, source, output, outputOffset + linear, block);
        }
    }
    private static int[] strides(int[] dimensions) {
        int[] result = new int[dimensions.length]; int stride = 1;
        for (int i = dimensions.length - 1; i >= 0; i--) { result[i] = stride; stride = Math.multiplyExact(stride, dimensions[i]); }
        return result;
    }
}
