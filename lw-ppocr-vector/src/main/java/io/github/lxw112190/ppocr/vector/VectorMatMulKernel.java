package io.github.lxw112190.ppocr.vector;

import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorSpecies;

/** Vector microkernel for row-major matrix multiplication. */
final class VectorMatMulKernel {
    private static final VectorSpecies<Float> SPECIES = VectorSupport.F32;

    private VectorMatMulKernel() { }

    static void multiply(float[] left, int leftOffset, float[] right, int rightOffset,
                         float[] output, int outputOffset, int rows, int inner, int columns) {
        int bound = SPECIES.loopBound(columns);
        int vectorWidth = SPECIES.length();
        int pairedBound = columns - columns % (vectorWidth * 2);
        int blockWidth = vectorWidth * 4;
        int blockBound = columns - columns % blockWidth;
        int row = 0;
        for (; row + 3 < rows; row += 4) {
            int leftRow0 = leftOffset + row * inner;
            int leftRow1 = leftRow0 + inner;
            int leftRow2 = leftRow1 + inner;
            int leftRow3 = leftRow2 + inner;
            int outputRow0 = outputOffset + row * columns;
            int outputRow1 = outputRow0 + columns;
            int outputRow2 = outputRow1 + columns;
            int outputRow3 = outputRow2 + columns;
            int column = 0;
            for (; column < pairedBound; column += vectorWidth * 2) {
                FloatVector sum00 = FloatVector.zero(SPECIES);
                FloatVector sum01 = FloatVector.zero(SPECIES);
                FloatVector sum10 = FloatVector.zero(SPECIES);
                FloatVector sum11 = FloatVector.zero(SPECIES);
                FloatVector sum20 = FloatVector.zero(SPECIES);
                FloatVector sum21 = FloatVector.zero(SPECIES);
                FloatVector sum30 = FloatVector.zero(SPECIES);
                FloatVector sum31 = FloatVector.zero(SPECIES);
                int rightRow = rightOffset + column;
                for (int k = 0; k < inner; k++) {
                    FloatVector values0 = FloatVector.fromArray(SPECIES, right, rightRow);
                    FloatVector values1 = FloatVector.fromArray(
                            SPECIES, right, rightRow + vectorWidth);
                    float left0 = left[leftRow0 + k];
                    float left1 = left[leftRow1 + k];
                    float left2 = left[leftRow2 + k];
                    float left3 = left[leftRow3 + k];
                    sum00 = sum00.add(values0.mul(left0));
                    sum01 = sum01.add(values1.mul(left0));
                    sum10 = sum10.add(values0.mul(left1));
                    sum11 = sum11.add(values1.mul(left1));
                    sum20 = sum20.add(values0.mul(left2));
                    sum21 = sum21.add(values1.mul(left2));
                    sum30 = sum30.add(values0.mul(left3));
                    sum31 = sum31.add(values1.mul(left3));
                    rightRow += columns;
                }
                sum00.intoArray(output, outputRow0 + column);
                sum01.intoArray(output, outputRow0 + column + vectorWidth);
                sum10.intoArray(output, outputRow1 + column);
                sum11.intoArray(output, outputRow1 + column + vectorWidth);
                sum20.intoArray(output, outputRow2 + column);
                sum21.intoArray(output, outputRow2 + column + vectorWidth);
                sum30.intoArray(output, outputRow3 + column);
                sum31.intoArray(output, outputRow3 + column + vectorWidth);
            }
            for (; column < bound; column += vectorWidth) {
                FloatVector sum0 = FloatVector.zero(SPECIES);
                FloatVector sum1 = FloatVector.zero(SPECIES);
                FloatVector sum2 = FloatVector.zero(SPECIES);
                FloatVector sum3 = FloatVector.zero(SPECIES);
                int rightRow = rightOffset + column;
                for (int k = 0; k < inner; k++) {
                    FloatVector values = FloatVector.fromArray(SPECIES, right, rightRow);
                    sum0 = sum0.add(values.mul(left[leftRow0 + k]));
                    sum1 = sum1.add(values.mul(left[leftRow1 + k]));
                    sum2 = sum2.add(values.mul(left[leftRow2 + k]));
                    sum3 = sum3.add(values.mul(left[leftRow3 + k]));
                    rightRow += columns;
                }
                sum0.intoArray(output, outputRow0 + column);
                sum1.intoArray(output, outputRow1 + column);
                sum2.intoArray(output, outputRow2 + column);
                sum3.intoArray(output, outputRow3 + column);
            }
            for (; column < columns; column++) {
                float sum0 = 0.0f;
                float sum1 = 0.0f;
                float sum2 = 0.0f;
                float sum3 = 0.0f;
                int rightIndex = rightOffset + column;
                for (int k = 0; k < inner; k++) {
                    float value = right[rightIndex];
                    sum0 += left[leftRow0 + k] * value;
                    sum1 += left[leftRow1 + k] * value;
                    sum2 += left[leftRow2 + k] * value;
                    sum3 += left[leftRow3 + k] * value;
                    rightIndex += columns;
                }
                output[outputRow0 + column] = sum0;
                output[outputRow1 + column] = sum1;
                output[outputRow2 + column] = sum2;
                output[outputRow3 + column] = sum3;
            }
        }
        for (; row < rows; row++) {
            int outputRow = outputOffset + row * columns;
            int leftRow = leftOffset + row * inner;
            int column = 0;
            for (; column < blockBound; column += blockWidth) {
                FloatVector sum0 = FloatVector.zero(SPECIES);
                FloatVector sum1 = FloatVector.zero(SPECIES);
                FloatVector sum2 = FloatVector.zero(SPECIES);
                FloatVector sum3 = FloatVector.zero(SPECIES);
                int rightRow = rightOffset + column;
                for (int k = 0; k < inner; k++) {
                    float value = left[leftRow + k];
                    sum0 = sum0.add(FloatVector.fromArray(SPECIES, right, rightRow).mul(value));
                    sum1 = sum1.add(FloatVector.fromArray(
                            SPECIES, right, rightRow + vectorWidth).mul(value));
                    sum2 = sum2.add(FloatVector.fromArray(
                            SPECIES, right, rightRow + vectorWidth * 2).mul(value));
                    sum3 = sum3.add(FloatVector.fromArray(
                            SPECIES, right, rightRow + vectorWidth * 3).mul(value));
                    rightRow += columns;
                }
                sum0.intoArray(output, outputRow + column);
                sum1.intoArray(output, outputRow + column + vectorWidth);
                sum2.intoArray(output, outputRow + column + vectorWidth * 2);
                sum3.intoArray(output, outputRow + column + vectorWidth * 3);
            }
            for (; column < bound; column += vectorWidth) {
                FloatVector sum = FloatVector.zero(SPECIES);
                int rightRow = rightOffset + column;
                for (int k = 0; k < inner; k++) {
                    sum = sum.add(FloatVector.fromArray(SPECIES, right, rightRow)
                            .mul(left[leftRow + k]));
                    rightRow += columns;
                }
                sum.intoArray(output, outputRow + column);
            }
            for (; column < columns; column++) {
                float sum = 0.0f;
                int rightIndex = rightOffset + column;
                for (int k = 0; k < inner; k++) {
                    sum += left[leftRow + k] * right[rightIndex];
                    rightIndex += columns;
                }
                output[outputRow + column] = sum;
            }
        }
    }
}
