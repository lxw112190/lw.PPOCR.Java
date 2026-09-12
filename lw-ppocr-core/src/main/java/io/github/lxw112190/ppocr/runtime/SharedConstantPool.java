package io.github.lxw112190.ppocr.runtime;

import io.github.lxw112190.ppocr.model.DataType;
import io.github.lxw112190.ppocr.model.LwmModel;
import io.github.lxw112190.ppocr.model.TensorInfo;
import java.nio.ByteBuffer;
import java.util.List;
import java.util.Map;
import java.util.WeakHashMap;

/** Shares immutable decoded model weights between dynamic-shape sessions. */
final class SharedConstantPool {
    private static final Map<LwmModel, float[][]> CACHE = new WeakHashMap<LwmModel, float[][]>();

    private SharedConstantPool() { }

    static float[][] acquire(LwmModel model, int[] lengths) {
        synchronized (CACHE) {
            float[][] existing = CACHE.get(model);
            if (existing != null) return existing;
            List<TensorInfo> tensors = model.getTensors();
            float[][] decoded = new float[tensors.size()][];
            for (int i = 0; i < tensors.size(); i++) {
                TensorInfo tensor = tensors.get(i);
                if (!tensor.isConstant()) continue;
                if (tensor.getDataType() != DataType.F32) {
                    throw new IllegalArgumentException("only F32 execution is supported: tensor " + i);
                }
                decoded[i] = readF32(model.constantData(i), lengths[i]);
            }
            CACHE.put(model, decoded);
            return decoded;
        }
    }

    private static float[] readF32(ByteBuffer bytes, int length) {
        float[] values = new float[length];
        for (int i = 0; i < length; i++) values[i] = bytes.getFloat(i * 4);
        return values;
    }
}
