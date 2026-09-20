package io.github.lxw112190.ppocr.benchmark;

import io.github.lxw112190.ppocr.kernels.KernelBackend;
import io.github.lxw112190.ppocr.kernels.ScalarBackend;
import java.lang.management.ManagementFactory;
import java.util.Arrays;
import java.util.Locale;

/** Matrix benchmark for the convolution shapes that dominate Tiny REC. */
public final class RecHotKernelPerformanceMain {
    private static volatile float sink;

    private RecHotKernelPerformanceMain() { }

    public static void main(String[] args) throws Exception {
        String backendName = args.length > 0 ? args[0] : "vector";
        int warmupRounds = args.length > 1 ? positive(args[1], "warmup") : 30;
        int iterations = args.length > 2 ? positive(args[2], "iterations") : 30;
        String filter = args.length > 3 ? args[3] : "all";
        KernelBackend backend = createBackend(backendName);
        PreparedShape[] cases = prepareShapes(filter);

        /*
         * Warm every selected shape before measuring allocations. This keeps
         * early shapes from absorbing Vector API C2 compilation and escape
         * analysis that belongs to the shared kernel.
         */
        for (int round = 0; round < warmupRounds; round++) {
            for (int i = 0; i < cases.length; i++) {
                PreparedShape testCase = cases[i];
                run(backend, testCase.shape, testCase.input, testCase.weights,
                        testCase.bias, testCase.output);
                consume(testCase.output, round + i);
            }
        }

        for (PreparedShape testCase : cases) {
            measureShape(backendName, backend, testCase, warmupRounds, iterations);
        }
    }

    private static void measureShape(String backendName, KernelBackend backend,
                                     PreparedShape testCase, int warmup, int iterations) {
        Shape shape = testCase.shape;
        long[] samples = new long[iterations];
        long allocatedBefore = ALLOCATION.snapshot();
        for (int i = 0; i < iterations; i++) {
            long start = System.nanoTime();
            run(backend, shape, testCase.input, testCase.weights, testCase.bias,
                    testCase.output);
            samples[i] = System.nanoTime() - start;
            consume(testCase.output, i + warmup);
        }
        long allocatedAfter = ALLOCATION.snapshot();
        Arrays.sort(samples);
        long allocatedPerOp = allocatedBefore < 0 || allocatedAfter < allocatedBefore
                ? -1 : (allocatedAfter - allocatedBefore) / iterations;
        String vectorBits = "vector".equals(backendName)
                ? System.getProperty("lwppocr.vectorBits", "preferred") : "n/a";
        String pointwiseBlock = "vector".equals(backendName)
                ? System.getProperty("lwppocr.vectorPointwiseBlock", "default") : "n/a";
        System.out.printf(Locale.ROOT,
                "{\"benchmark\":\"rec-hot-kernel\",\"backend\":\"%s\","
                        + "\"vector_bits\":\"%s\",\"pointwise_block\":\"%s\","
                        + "\"operator\":\"%s\",\"input_channels\":%d,"
                        + "\"output_channels\":%d,\"height\":%d,\"width\":%d,"
                        + "\"output_height\":%d,\"output_width\":%d,"
                        + "\"kernel\":\"%dx%d\",\"stride\":%d,"
                        + "\"warmup\":%d,\"iterations\":%d,"
                        + "\"mean_ms\":%.3f,\"median_ms\":%.3f,\"p95_ms\":%.3f,"
                        + "\"allocated_bytes_per_op\":%d,\"checksum\":\"%s\"}%n",
                backendName, vectorBits, pointwiseBlock, shape.operator,
                shape.channels, shape.outputChannels, shape.height, shape.width,
                shape.outputHeight, shape.outputWidth, shape.kernelHeight,
                shape.kernelWidth, shape.stride, warmup, iterations,
                mean(samples) / 1_000_000.0,
                samples[samples.length / 2] / 1_000_000.0,
                samples[(int) Math.min(samples.length - 1,
                        Math.ceil(samples.length * 0.95) - 1)] / 1_000_000.0,
                allocatedPerOp, checksum(testCase.output));
    }

    private static void run(KernelBackend backend, Shape shape, float[] input,
                            float[] weights, float[] bias, float[] output) {
        backend.conv(input, 0, weights, 0, bias, 0, output, 0,
                1, shape.channels, shape.height, shape.width, shape.outputChannels,
                shape.kernelHeight, shape.kernelWidth, shape.stride, shape.stride,
                1, 1, shape.pad, shape.pad, shape.pad, shape.pad, 1,
                shape.outputHeight, shape.outputWidth);
    }

    private static KernelBackend createBackend(String name) throws Exception {
        if ("scalar".equals(name)) return new ScalarBackend();
        if (!"vector".equals(name)) {
            throw new IllegalArgumentException("backend must be scalar or vector");
        }
        return (KernelBackend) Class.forName("io.github.lxw112190.ppocr.vector.VectorBackend")
                .getDeclaredConstructor().newInstance();
    }

    private static int positive(String value, String name) {
        int parsed = Integer.parseInt(value);
        if (parsed <= 0) throw new IllegalArgumentException(name + " must be positive");
        return parsed;
    }

    private static float[] fixture(int length, int period, float scale) {
        float[] values = new float[length];
        int center = period / 2;
        for (int i = 0; i < length; i++) values[i] = (i % period - center) * scale;
        return values;
    }

    private static void consume(float[] values, int iteration) {
        sink += values[(iteration * 7919) % values.length];
    }

    private static double mean(long[] values) {
        long total = 0L;
        for (long value : values) total += value;
        return (double) total / values.length;
    }

    private static String checksum(float[] values) {
        long hash = -3750763034362895579L;
        for (float value : values) {
            hash ^= Float.floatToIntBits(value) & 0xffffffffL;
            hash *= 1099511628211L;
        }
        return Long.toUnsignedString(hash);
    }

    private static final Shape[] SHAPES = createShapes();

    private static Shape[] createShapes() {
        Shape[] shapes = new Shape[16];
        int index = 0;
        for (int width : new int[] {80, 120, 160, 240}) {
            shapes[index++] = new Shape("pointwise", 160, 320, 3, width,
                    1, 1, 1, 0);
            shapes[index++] = new Shape("pointwise", 320, 160, 3, width,
                    1, 1, 1, 0);
            shapes[index++] = new Shape("pointwise", 96, 192, 6, width,
                    1, 1, 1, 0);
        }
        for (int width : new int[] {160, 240, 320, 480}) {
            shapes[index++] = new Shape("stride-two", 24, 48, 24, width,
                    3, 3, 2, 1);
        }
        return shapes;
    }

    private static PreparedShape[] prepareShapes(String filter) {
        int count = 0;
        for (Shape shape : SHAPES) {
            if (matchesFilter(shape, filter)) count++;
        }
        PreparedShape[] result = new PreparedShape[count];
        int index = 0;
        for (Shape shape : SHAPES) {
            if (matchesFilter(shape, filter)) result[index++] = new PreparedShape(shape);
        }
        return result;
    }

    private static boolean matchesFilter(Shape shape, String filter) {
        if ("all".equals(filter)) return true;
        if ("pointwise".equals(filter)) return "pointwise".equals(shape.operator);
        if ("stride-two".equals(filter)) return "stride-two".equals(shape.operator);
        throw new IllegalArgumentException("filter must be all, pointwise or stride-two");
    }

    private static final class PreparedShape {
        final Shape shape;
        final float[] input;
        final float[] weights;
        final float[] bias;
        final float[] output;

        PreparedShape(Shape shape) {
            this.shape = shape;
            this.input = fixture(shape.channels * shape.height * shape.width,
                    101, 0.001953125f);
            this.weights = fixture(shape.outputChannels * shape.channels * shape.kernelArea,
                    67, 0.0009765625f);
            this.bias = fixture(shape.outputChannels, 31, 0.00390625f);
            this.output = new float[shape.outputChannels * shape.outputHeight * shape.outputWidth];
        }
    }

    private static final class Shape {
        final String operator;
        final int channels;
        final int outputChannels;
        final int height;
        final int width;
        final int kernelHeight;
        final int kernelWidth;
        final int stride;
        final int pad;
        final int outputHeight;
        final int outputWidth;
        final int kernelArea;

        Shape(String operator, int channels, int outputChannels, int height, int width,
              int kernelHeight, int kernelWidth, int stride, int pad) {
            this.operator = operator;
            this.channels = channels;
            this.outputChannels = outputChannels;
            this.height = height;
            this.width = width;
            this.kernelHeight = kernelHeight;
            this.kernelWidth = kernelWidth;
            this.stride = stride;
            this.pad = pad;
            this.outputHeight = (height + 2 * pad - kernelHeight) / stride + 1;
            this.outputWidth = (width + 2 * pad - kernelWidth) / stride + 1;
            this.kernelArea = kernelHeight * kernelWidth;
        }
    }

    private static final AllocationProbe ALLOCATION = AllocationProbe.create();

    private static final class AllocationProbe {
        private final com.sun.management.ThreadMXBean bean;
        private final long threadId;

        private AllocationProbe(com.sun.management.ThreadMXBean bean) {
            this.bean = bean;
            this.threadId = Thread.currentThread().getId();
        }

        static AllocationProbe create() {
            java.lang.management.ThreadMXBean base = ManagementFactory.getThreadMXBean();
            if (!(base instanceof com.sun.management.ThreadMXBean)) {
                return new AllocationProbe(null);
            }
            com.sun.management.ThreadMXBean bean = (com.sun.management.ThreadMXBean) base;
            try {
                if (bean.isThreadAllocatedMemorySupported()
                        && !bean.isThreadAllocatedMemoryEnabled()) {
                    bean.setThreadAllocatedMemoryEnabled(true);
                }
                return bean.isThreadAllocatedMemorySupported()
                        && bean.isThreadAllocatedMemoryEnabled()
                        ? new AllocationProbe(bean) : new AllocationProbe(null);
            } catch (RuntimeException ex) {
                return new AllocationProbe(null);
            }
        }

        long snapshot() {
            return bean == null ? -1 : bean.getThreadAllocatedBytes(threadId);
        }
    }
}
