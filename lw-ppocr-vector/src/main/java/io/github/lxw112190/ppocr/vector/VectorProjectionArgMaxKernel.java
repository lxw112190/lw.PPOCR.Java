package io.github.lxw112190.ppocr.vector;

import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;

/** Finishes a small batch of dense projection rows without materializing softmax output. */
final class VectorProjectionArgMaxKernel {
    private static final VectorSpecies<Float> SPECIES = VectorSupport.F32;

    private VectorProjectionArgMaxKernel() { }

    static void finish(float[] rowScratch, int rows, int columns, float[] bias, int biasOffset,
                       int[] bestIndices, float[] bestLogits, float[] bestProbabilities,
                       int rowOffset) {
        int bound = SPECIES.loopBound(columns);
        for (int localRow = 0; localRow < rows; localRow++) {
            int row = rowOffset + localRow;
            int scratchBase = localRow * columns;
            int column = 0;
            for (; column < bound; column += SPECIES.length()) {
                FloatVector.fromArray(SPECIES, rowScratch, scratchBase + column)
                        .add(FloatVector.fromArray(SPECIES, bias, biasOffset + column))
                        .intoArray(rowScratch, scratchBase + column);
            }
            for (; column < columns; column++) {
                rowScratch[scratchBase + column] += bias[biasOffset + column];
            }

            int best = 0;
            float maximum = rowScratch[scratchBase];
            if (!Float.isFinite(maximum)) {
                throw new IllegalArgumentException("projection contains non-finite values");
            }
            for (column = 1; column < columns; column++) {
                float value = rowScratch[scratchBase + column];
                if (!Float.isFinite(value)) {
                    throw new IllegalArgumentException("projection contains non-finite values");
                }
                if (value > maximum) {
                    maximum = value;
                    best = column;
                }
            }

            FloatVector vectorSum = FloatVector.zero(SPECIES);
            column = 0;
            for (; column < bound; column += SPECIES.length()) {
                FloatVector values = FloatVector.fromArray(SPECIES,
                                rowScratch, scratchBase + column)
                        .sub(maximum).lanewise(VectorOperators.EXP);
                vectorSum = vectorSum.add(values);
            }
            float sum = vectorSum.reduceLanes(VectorOperators.ADD);
            for (; column < columns; column++) {
                float value = (float) Math.exp(rowScratch[scratchBase + column] - maximum);
                sum += value;
            }
            bestIndices[row] = best;
            bestLogits[row] = maximum;
            bestProbabilities[row] = 1.0f / sum;
        }
    }
}
