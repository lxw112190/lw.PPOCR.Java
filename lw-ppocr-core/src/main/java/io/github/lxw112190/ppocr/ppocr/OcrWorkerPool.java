package io.github.lxw112190.ppocr.ppocr;

import io.github.lxw112190.ppocr.image.BgrImage;
import io.github.lxw112190.ppocr.model.OcrErrorCode;
import io.github.lxw112190.ppocr.model.OcrException;
import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Collections;
import java.util.IdentityHashMap;
import java.util.List;
import java.util.Set;

/** Thread-safe pool of independent OCR sessions. The pool owns its workers. */
public final class OcrWorkerPool implements AutoCloseable {
    private final Object lock = new Object();
    private final ArrayDeque<PaddleOcr> available;
    private final List<PaddleOcr> workers;
    private int inFlight;
    private boolean closed;

    /** Takes ownership of every worker; the same worker instance cannot appear twice. */
    public OcrWorkerPool(List<PaddleOcr> workers) {
        if (workers == null || workers.isEmpty()) {
            throw new OcrException(OcrErrorCode.INVALID_ARGUMENT, "at least one OCR worker is required");
        }
        Set<PaddleOcr> identities = Collections.newSetFromMap(
                new IdentityHashMap<PaddleOcr, Boolean>());
        List<PaddleOcr> copy = new ArrayList<PaddleOcr>(workers.size());
        for (PaddleOcr worker : workers) {
            if (worker == null || !identities.add(worker)) {
                throw new OcrException(OcrErrorCode.INVALID_ARGUMENT,
                        "OCR workers must be non-null and distinct");
            }
            copy.add(worker);
        }
        this.workers = Collections.unmodifiableList(copy);
        this.available = new ArrayDeque<PaddleOcr>(copy);
    }

    public int size() { return workers.size(); }

    public OcrResult recognize(BgrImage source) {
        if (source == null) {
            throw new OcrException(OcrErrorCode.INVALID_ARGUMENT, "source image is required");
        }
        PaddleOcr worker = acquire();
        try {
            return worker.recognize(source);
        } finally {
            release(worker);
        }
    }

    @Override
    public void close() {
        boolean interrupted = false;
        synchronized (lock) {
            if (closed) return;
            closed = true;
            while (inFlight != 0) {
                try {
                    lock.wait();
                } catch (InterruptedException e) {
                    interrupted = true;
                }
            }
        }
        RuntimeException failure = null;
        for (PaddleOcr worker : workers) {
            try {
                worker.close();
            } catch (RuntimeException e) {
                if (failure == null) failure = e;
                else failure.addSuppressed(e);
            }
        }
        if (interrupted) Thread.currentThread().interrupt();
        if (failure != null) throw failure;
    }

    private PaddleOcr acquire() {
        boolean interrupted = false;
        synchronized (lock) {
            while (available.isEmpty() && !closed) {
                try {
                    lock.wait();
                } catch (InterruptedException e) {
                    interrupted = true;
                    break;
                }
            }
            if (interrupted) {
                Thread.currentThread().interrupt();
                throw new OcrException(OcrErrorCode.RESOURCE_LIMIT,
                        "OCR worker acquisition was interrupted");
            }
            if (closed) {
                throw new OcrException(OcrErrorCode.INVALID_ARGUMENT, "OCR worker pool is closed");
            }
            inFlight++;
            return available.removeFirst();
        }
    }

    private void release(PaddleOcr worker) {
        synchronized (lock) {
            available.addLast(worker);
            inFlight--;
            lock.notifyAll();
        }
    }
}
