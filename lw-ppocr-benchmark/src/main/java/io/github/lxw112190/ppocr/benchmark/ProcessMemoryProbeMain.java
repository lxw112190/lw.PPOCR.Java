package io.github.lxw112190.ppocr.benchmark;

/** Small self-check for the Linux process RSS probe used by comparison CI. */
public final class ProcessMemoryProbeMain {
    private ProcessMemoryProbeMain() { }

    public static void main(String[] args) {
        try (ProcessMemoryProbe probe = ProcessMemoryProbe.open(true)) {
            probe.refresh();
            System.out.println("{\"supported\":" + probe.isSupported()
                    + ",\"source\":\"" + probe.source() + "\""
                    + ",\"rss_bytes\":" + probe.rssBytes()
                    + ",\"hwm_bytes\":" + probe.highWaterBytes() + "}");
        }
    }
}
