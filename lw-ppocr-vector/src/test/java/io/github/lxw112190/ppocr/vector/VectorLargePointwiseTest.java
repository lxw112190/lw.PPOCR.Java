package io.github.lxw112190.ppocr.vector;

import io.github.lxw112190.ppocr.kernels.PreparedConvBackend;
import io.github.lxw112190.ppocr.kernels.ScalarBackend;
import java.util.Arrays;
import java.util.Random;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import org.junit.Test;
import static org.junit.Assert.*;

public final class VectorLargePointwiseTest {
    @Test public void spatialPanelsTailsAndRowShardsAreBitExact() {
        for (int channels : new int[] {256, 257, 321}) {
            for (int outputs : new int[] {256, 259}) for (int width : new int[] {13, 17}) {
                compare(channels, outputs, 5, width, false);
                compare(channels, outputs, 5, width, true);
            }
        }
    }

    @Test public void largeReductionAndShortSpatialMapAreBitExact() {
        compare(1024, 512, 1, 65, true);
        compare(1536, 768, 1, 65, false);
    }

    @Test public void nonfiniteValuesAndNegativeZeroMatchScalar() {
        int[] p = parameters(256, 259, 1, 65);
        float[] input = new float[256 * 65], weights = new float[259 * 256];
        Arrays.fill(input, -0.0f);
        Arrays.fill(weights, 1.0f);
        weights[0] = Float.POSITIVE_INFINITY;
        weights[256] = Float.NaN;
        input[65 * 4 + 7] = Float.NEGATIVE_INFINITY;
        float[] bias = new float[259]; Arrays.fill(bias, -0.0f);
        float[] actual = new float[259 * 65], expected = new float[actual.length];
        new ScalarBackend().conv(input, 0, weights, 0, bias, 0, expected, 0,
                1, 256, 1, 65, 259, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 65);
        PreparedConvBackend.Kernel plan = new VectorBackend().prepareConv(weights, 0, p);
        assertNotNull(plan);
        plan.runRows(input, 0, bias, 0, actual, 0, new float[plan.scratchFloats()], 0, plan.outputRows());
        assertBits(expected, actual);
    }

    @Test public void unsupportedGeometryAndTinyChannelsKeepExistingPaths() {
        int[] p = parameters(256, 256, 5, 17);
        float[] weights = new float[256 * 256];
        for (int index : new int[] {0, 5, 6, 7, 8, 11, 12, 13, 14, 15, 16, 17}) {
            int[] invalid = p.clone();
            invalid[index] += 1;
            assertNull(VectorLargePointwise.prepare(weights, 0, invalid));
        }
        assertNull(VectorLargePointwise.prepare(weights, 0, parameters(128, 256, 5, 17)));
        assertNull(VectorLargePointwise.prepare(weights, 0, parameters(256, 128, 5, 17)));
        assertNull(VectorLargePointwise.prepare(weights, 0, parameters(256, 256, 1, 17)));
    }

    @Test public void abSwitchRestoresOldDispatchWithoutWeightCopies() {
        String old = System.getProperty("lwppocr.disableLargePointwise");
        try {
            VectorBackend backend = new VectorBackend();
            float[] weights = new float[256 * 256];
            int[] p = parameters(256, 256, 5, 17);
            System.clearProperty("lwppocr.disableLargePointwise");
            assertTrue(backend.prepareConv(weights, 0, p) instanceof VectorLargePointwise);
            assertEquals(0, backend.preparedConvWeightBytes());
            System.setProperty("lwppocr.disableLargePointwise", "true");
            assertNull(backend.prepareConv(weights, 0, p));
        } finally {
            if (old == null) System.clearProperty("lwppocr.disableLargePointwise");
            else System.setProperty("lwppocr.disableLargePointwise", old);
        }
    }

    @Test public void concurrentShardsReusePrivateScratchAndKeepOutputGuards() throws Exception {
        final int channels = 321, outputs = 259, plane = 258;
        final float[] input = random(channels * plane + 7), weights = random(outputs * channels + 9);
        final float[] bias = random(outputs + 5), actual = new float[outputs * plane + 22];
        float[] expected = new float[actual.length];
        Arrays.fill(actual, -31); Arrays.fill(expected, -31);
        new ScalarBackend().conv(input, 7, weights, 9, bias, 5, expected, 11,
                1, channels, 2, 129, outputs, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 2, 129);
        final PreparedConvBackend.Kernel plan = new VectorBackend().prepareConv(weights, 9,
                parameters(channels, outputs, 2, 129));
        final float[][] scratch = new float[4][plan.scratchFloats()];
        ExecutorService workers = Executors.newFixedThreadPool(4);
        try {
            for (int repeat = 0; repeat < 3; repeat++) {
                List<Future<Void>> tasks = new ArrayList<Future<Void>>();
                for (int shard = 0; shard < 4; shard++) {
                    final int worker = shard;
                    tasks.add(workers.submit(new Callable<Void>() {
                        public Void call() {
                            plan.runRows(input, 7, bias, 5, actual, 11, scratch[worker],
                                    plan.outputRows() * worker / 4, plan.outputRows() * (worker + 1) / 4);
                            return null;
                        }
                    }));
                }
                for (Future<Void> task : tasks) task.get();
                assertBits(expected, actual);
            }
        } finally { workers.shutdownNow(); }
    }

    @Test public void oversizedPanelsFallBackBeforeAllocatingScratch() {
        assertNotNull(VectorLargePointwise.prepare(new float[0], 0, parameters(8192, 256, 1, 64)));
        assertNull(VectorLargePointwise.prepare(new float[0], 0, parameters(8193, 256, 1, 64)));
    }

    private static void compare(int channels, int outputs, int height, int width, boolean withBias) {
        VectorBackend backend = new VectorBackend();
        int plane = height * width;
        float[] input = random(channels * plane + 7), weights = random(outputs * channels + 9);
        float[] bias = withBias ? random(outputs + 5) : null;
        float[] actual = new float[outputs * plane + 22], expected = new float[actual.length];
        Arrays.fill(actual, -31); Arrays.fill(expected, -31);
        new ScalarBackend().conv(input, 7, weights, 9, bias, 5, expected, 11,
                1, channels, height, width, outputs, 1, 1, 1, 1, 1, 1,
                0, 0, 0, 0, 1, height, width);
        PreparedConvBackend.Kernel plan = backend.prepareConv(weights, 9, parameters(channels, outputs, height, width));
        assertTrue(plan instanceof VectorLargePointwise);
        assertEquals(channels * 32, plan.scratchFloats());
        float[] scratch = new float[plan.scratchFloats()];
        for (int repeat = 0; repeat < 2; repeat++) {
            int split = plan.outputRows() / 2;
            plan.runRows(input, 7, bias, 5, actual, 11, scratch, split, plan.outputRows());
            plan.runRows(input, 7, bias, 5, actual, 11, scratch, 0, split);
            assertBits(expected, actual);
        }
        assertEquals(0, backend.preparedConvWeightBytes());
    }

    private static int[] parameters(int channels, int outputs, int height, int width) {
        return new int[] {1, channels, height, width, outputs, 1, 1, 1, 1, 1, 1,
                0, 0, 0, 0, 1, height, width};
    }
    private static float[] random(int length) {
        Random random = new Random(2307 + length);
        float[] values = new float[length];
        for (int i = 0; i < length; i++) values[i] = (random.nextFloat() - 0.5f) * (i % 13 + 1);
        return values;
    }
    private static void assertBits(float[] expected, float[] actual) {
        for (int i = 0; i < expected.length; i++) {
            assertEquals("float bits at " + i, Float.floatToIntBits(expected[i]), Float.floatToIntBits(actual[i]));
        }
    }
}
