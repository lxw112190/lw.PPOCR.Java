package io.github.lxw112190.ppocr.runtime;

import io.github.lxw112190.ppocr.model.OcrErrorCode;
import io.github.lxw112190.ppocr.model.OcrException;

/** Matrix multiplication with right-aligned broadcast batch dimensions (rank >= 2). */
final class MatMulPlan {
    final int rows, inner, columns, batches;
    final int[] leftOffsets, rightOffsets;

    MatMulPlan(TensorShape a, TensorShape b) {
        TensorShape out = outputShape(a, b);
        int rank = out.getRank(), batchRank = rank - 2;
        rows = out.get(rank - 2); columns = out.get(rank - 1); inner = a.get(a.getRank() - 1);
        batches = Math.toIntExact(out.getElementCount() / rows / columns);
        leftOffsets = new int[batches]; rightOffsets = new int[batches];
        for (int i = 0; i < batches; i++) {
            int rest = i, as = rows * inner, bs = inner * columns;
            for (int axis = batchRank - 1; axis >= 0; axis--) {
                int coordinate = rest % out.get(axis); rest /= out.get(axis);
                int aa = axis - (rank - a.getRank()), ba = axis - (rank - b.getRank());
                if (aa >= 0) { if (a.get(aa) != 1) leftOffsets[i] += coordinate * as; as *= a.get(aa); }
                if (ba >= 0) { if (b.get(ba) != 1) rightOffsets[i] += coordinate * bs; bs *= b.get(ba); }
            }
        }
    }

    static TensorShape outputShape(TensorShape a, TensorShape b) {
        int ar = a.getRank(), br = b.getRank(), rank = Math.max(ar, br);
        if (ar < 2 || br < 2 || a.get(ar - 1) != b.get(br - 2)) throw bad("MatMul matrix dimensions mismatch");
        int[] out = new int[rank];
        for (int i = 0; i < rank - 2; i++) {
            int aa = i - (rank - ar), ba = i - (rank - br);
            int ad = aa < 0 ? 1 : a.get(aa), bd = ba < 0 ? 1 : b.get(ba);
            if (ad != bd && ad != 1 && bd != 1) throw bad("MatMul batch dimensions cannot broadcast");
            out[i] = Math.max(ad, bd);
        }
        out[rank - 2] = a.get(ar - 2); out[rank - 1] = b.get(br - 1);
        return new TensorShape(out);
    }
    private static OcrException bad(String message) { return new OcrException(OcrErrorCode.INVALID_MODEL, message); }
}
