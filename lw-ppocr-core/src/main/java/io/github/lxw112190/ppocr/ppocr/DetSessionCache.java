package io.github.lxw112190.ppocr.ppocr;

import io.github.lxw112190.ppocr.kernels.KernelBackend;
import io.github.lxw112190.ppocr.model.LwmModel;
import io.github.lxw112190.ppocr.runtime.WorkspaceDiagnostics;
import java.util.LinkedHashMap;
import java.util.Map;

/** Small LRU cache for dynamic DET sessions; access is serialized by the detector. */
final class DetSessionCache implements AutoCloseable {
    private final int capacity;
    private final LinkedHashMap<DetShapeKey, DetSessionContext> entries =
            new LinkedHashMap<DetShapeKey, DetSessionContext>(4, 0.75f, true);

    DetSessionCache(int capacity) {
        if (capacity <= 0) throw new IllegalArgumentException("DET cache capacity must be positive");
        this.capacity = capacity;
    }

    DetSessionContext get(DetShapeKey key) { return entries.get(key); }

    WorkspaceDiagnostics workspaceDiagnostics() {
        WorkspaceDiagnostics[] values = new WorkspaceDiagnostics[entries.size()];
        int index = 0;
        for (DetSessionContext context : entries.values()) {
            values[index++] = context.workspaceDiagnostics();
        }
        return WorkspaceDiagnostics.aggregate(values);
    }

    long decodedConstantBytes() {
        for (DetSessionContext context : entries.values()) {
            return context.decodedConstantBytes();
        }
        return 0L;
    }

    int size() { return entries.size(); }

    DetSessionContext getOrCreate(DetShapeKey key, LwmModel model, KernelBackend backend) {
        DetSessionContext existing = entries.get(key);
        if (existing != null) return existing;
        // DetShapeKey stores height first for cache identity, while the context
        // constructor follows the public image convention (width, height).
        DetSessionContext created = new DetSessionContext(model, key.width(), key.height(), backend);
        entries.put(key, created);
        if (entries.size() > capacity) {
            Map.Entry<DetShapeKey, DetSessionContext> eldest = entries.entrySet().iterator().next();
            entries.remove(eldest.getKey());
            eldest.getValue().close();
        }
        return created;
    }

    @Override
    public void close() {
        for (DetSessionContext context : entries.values()) context.close();
        entries.clear();
    }
}
