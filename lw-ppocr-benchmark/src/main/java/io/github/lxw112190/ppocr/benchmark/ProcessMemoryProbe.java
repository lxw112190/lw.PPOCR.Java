package io.github.lxw112190.ppocr.benchmark;

import java.io.File;
import java.io.IOException;
import java.io.RandomAccessFile;
import java.nio.charset.StandardCharsets;

/** Low-allocation Linux process RSS probe for comparison benchmarks. */
final class ProcessMemoryProbe implements AutoCloseable {
    private static final String PROC_STATUS = "/proc/self/status";
    private static final byte[] VM_RSS = "VmRSS:".getBytes(StandardCharsets.US_ASCII);
    private static final byte[] VM_HWM = "VmHWM:".getBytes(StandardCharsets.US_ASCII);
    private static final int BUFFER_SIZE = 16 * 1024;

    private final RandomAccessFile status;
    private final byte[] buffer;
    private long rssBytes = -1L;
    private long highWaterBytes = -1L;

    private ProcessMemoryProbe(RandomAccessFile status) {
        this.status = status;
        this.buffer = status == null ? null : new byte[BUFFER_SIZE];
    }

    static ProcessMemoryProbe open(boolean enabled) {
        if (!enabled) return new ProcessMemoryProbe(null);
        File file = new File(PROC_STATUS);
        if (!file.isFile()) return new ProcessMemoryProbe(null);
        try {
            return new ProcessMemoryProbe(new RandomAccessFile(file, "r"));
        } catch (IOException ignored) {
            return new ProcessMemoryProbe(null);
        }
    }

    boolean isSupported() {
        return status != null;
    }

    String source() {
        return isSupported() ? "linux-procfs" : "unsupported";
    }

    void refresh() {
        if (status == null) {
            rssBytes = -1L;
            highWaterBytes = -1L;
            return;
        }
        try {
            status.seek(0L);
            int length = status.read(buffer, 0, buffer.length);
            if (length <= 0) {
                rssBytes = -1L;
                highWaterBytes = -1L;
                return;
            }
            rssBytes = findKilobytes(buffer, length, VM_RSS);
            highWaterBytes = findKilobytes(buffer, length, VM_HWM);
        } catch (IOException ignored) {
            rssBytes = -1L;
            highWaterBytes = -1L;
        }
    }

    long rssBytes() {
        return rssBytes;
    }

    long highWaterBytes() {
        return highWaterBytes;
    }

    @Override
    public void close() {
        if (status == null) return;
        try {
            status.close();
        } catch (IOException ignored) {
            // Diagnostics must never fail OCR cleanup.
        }
    }

    private static long findKilobytes(byte[] data, int length, byte[] key) {
        for (int i = 0; i + key.length <= length; i++) {
            if (i != 0 && data[i - 1] != '\n') continue;
            if (!matches(data, i, length, key)) continue;
            int position = i + key.length;
            while (position < length && (data[position] == ' ' || data[position] == '\t')) {
                position++;
            }
            long value = 0L;
            boolean found = false;
            while (position < length) {
                byte current = data[position];
                if (current < '0' || current > '9') break;
                found = true;
                value = value * 10L + current - '0';
                position++;
            }
            return found ? value * 1024L : -1L;
        }
        return -1L;
    }

    private static boolean matches(byte[] data, int offset, int length, byte[] key) {
        if (offset + key.length > length) return false;
        for (int i = 0; i < key.length; i++) {
            if (data[offset + i] != key[i]) return false;
        }
        return true;
    }
}
