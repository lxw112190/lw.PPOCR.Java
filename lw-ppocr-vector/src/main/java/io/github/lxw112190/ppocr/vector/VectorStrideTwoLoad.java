package io.github.lxw112190.ppocr.vector;

import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorMask;
import jdk.incubator.vector.VectorShuffle;
import jdk.incubator.vector.VectorSpecies;

/** Portable stride-two input load: no gather, dynamic index vector or masked tail. */
final class VectorStrideTwoLoad {
    private static final VectorSpecies<Float> S = VectorSupport.F32;
    private static final int L = S.length();
    private static final VectorShuffle<Float> EVEN = VectorShuffle.fromOp(S, i -> (i * 2) % L);
    private static final VectorShuffle<Float> ODD = VectorShuffle.fromOp(S, i -> (i * 2 + 1) % L);
    private static final VectorMask<Float> UPPER = VectorMask.fromLong(S, -1L << (L / 2));

    private VectorStrideTwoLoad() { }

    static FloatVector load(float[] input, int offset) {
        FloatVector first = FloatVector.fromArray(S, input, offset);
        // Overlap one element: the last load ends at offset+2*L-2, exactly the
        // last gather lane. No unused extra element beyond a row / array end.
        FloatVector second = FloatVector.fromArray(S, input, offset + L - 1);
        return first.rearrange(EVEN).blend(second.rearrange(ODD), UPPER);
    }
}
