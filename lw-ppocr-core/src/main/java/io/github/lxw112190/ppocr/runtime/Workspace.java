package io.github.lxw112190.ppocr.runtime;

import io.github.lxw112190.ppocr.model.OcrErrorCode;
import io.github.lxw112190.ppocr.model.OcrException;

/** Session-owned primitive storage allocated once from a workspace plan. */
public final class Workspace {
    private final float[] fp32;

    public Workspace(WorkspacePlan plan) {
        if (plan == null || plan.getTotalBytes() > Integer.MAX_VALUE * 4L) {
            throw new OcrException(OcrErrorCode.RESOURCE_LIMIT, "workspace exceeds Java array capacity");
        }
        long elements = (plan.getTotalBytes() + 3L) / 4L;
        if (elements > Integer.MAX_VALUE) {
            throw new OcrException(OcrErrorCode.RESOURCE_LIMIT, "workspace exceeds Java array capacity");
        }
        this.fp32 = new float[(int) elements];
    }

    public float[] fp32() {
        return fp32;
    }
}
