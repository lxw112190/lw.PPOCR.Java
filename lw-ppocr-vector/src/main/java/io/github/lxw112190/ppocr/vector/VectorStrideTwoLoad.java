package io.github.lxw112190.ppocr.vector;

import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorMask;
import jdk.incubator.vector.VectorShuffle;
import jdk.incubator.vector.VectorSpecies;

/** Cached portable stride-two permutations; kernels keep the Vector values local. */
final class VectorStrideTwoLoad {
    private static final VectorSpecies<Float> S = VectorSupport.F32;
    private static final int L = S.length();
    static final VectorShuffle<Float> EVEN = VectorShuffle.fromOp(S, i -> (i * 2) % L);
    static final VectorShuffle<Float> ODD = VectorShuffle.fromOp(S, i -> (i * 2 + 1) % L);
    static final VectorMask<Float> UPPER = VectorMask.fromLong(S, -1L << (L / 2));

    private VectorStrideTwoLoad() { }

    // Reference for bit-pattern tests only. Hot kernels must perform this sequence
    // in their own array-entry loop: a non-inlined Vector return allocates a box
    // and payload, even when this method itself is fully C2-compiled.
    static FloatVector load(float[] input, int offset) {
        FloatVector first = FloatVector.fromArray(S, input, offset);
        // Overlap one element: the last load ends at offset+2*L-2, exactly the
        // last gather lane. No unused extra element beyond a row / array end.
        FloatVector second = FloatVector.fromArray(S, input, offset + L - 1);
        return first.rearrange(EVEN).blend(second.rearrange(ODD), UPPER);
    }
}
