package io.github.lxw112190.ppocr.ppocr;

import org.junit.Assert;
import org.junit.Test;

public class PaddleOcrOptionsTest {
    @Test
    public void defaultsMatchCCompatiblePipeline() {
        PaddleOcrOptions options = PaddleOcrOptions.defaults();
        Assert.assertEquals(0.3f, options.getDetectionBitmapThreshold(), 0.0f);
        Assert.assertEquals(0.6f, options.getDetectionBoxThreshold(), 0.0f);
        Assert.assertEquals(1.6f, options.getDetectionUnclipRatio(), 0.0f);
        Assert.assertFalse(options.isDetectionDilation());
        Assert.assertEquals(1000, options.getMaxDetectionCandidates());
        Assert.assertEquals(0.9f, options.getClassifierThreshold(), 0.0f);
        Assert.assertEquals(ReadingOrder.HORIZONTAL_LTR, options.getReadingOrder());
        Assert.assertEquals(1, options.getClassificationParallelism());
        Assert.assertEquals(1, options.getRecognitionParallelism());
        Assert.assertEquals(ParallelismPolicy.MANUAL, options.getParallelismPolicy());
        Assert.assertEquals(960, options.getDetectionMaximumSideLength());
    }

    @Test
    public void builderCreatesIndependentImmutableOptions() {
        PaddleOcrOptions.Builder builder = PaddleOcrOptions.builder()
                .setDetectionBitmapThreshold(0.2f)
                .setDetectionBoxThreshold(0.7f)
                .setDetectionUnclipRatio(2.0f)
                .setDetectionDilation(true)
                .setMaxDetectionCandidates(42)
                .setClassifierThreshold(0.8f)
                .setReadingOrder(ReadingOrder.VERTICAL_RTL)
                .setClassificationParallelism(3)
                .setRecognitionParallelism(4)
                .setDetectionMaximumSideLength(320);
        PaddleOcrOptions first = builder.build();
        PaddleOcrOptions second = builder.setMaxDetectionCandidates(84).build();
        Assert.assertEquals(42, first.getMaxDetectionCandidates());
        Assert.assertEquals(84, second.getMaxDetectionCandidates());
        Assert.assertTrue(first.isDetectionDilation());
        Assert.assertEquals(ReadingOrder.VERTICAL_RTL, first.getReadingOrder());
        Assert.assertEquals(3, first.getClassificationParallelism());
        Assert.assertEquals(4, first.getRecognitionParallelism());
        Assert.assertEquals(320, first.getDetectionMaximumSideLength());
        Assert.assertEquals(ParallelismPolicy.MANUAL, first.getParallelismPolicy());
    }

    @Test
    public void supportsAutomaticAndUnifiedParallelism() {
        PaddleOcrOptions automatic = PaddleOcrOptions.builder()
                .setParallelismMode(ParallelismPolicy.AUTO).build();
        Assert.assertEquals(ParallelismPolicy.AUTO, automatic.getParallelismPolicy());
        Assert.assertEquals(ParallelismPolicy.AUTO,
                PaddleOcrOptions.builder().setParallelism(0).build().getParallelismPolicy());
        PaddleOcrOptions manual = PaddleOcrOptions.builder().setParallelism(3).build();
        Assert.assertEquals(ParallelismPolicy.MANUAL, manual.getParallelismPolicy());
        Assert.assertEquals(3, manual.getClassificationParallelism());
        Assert.assertEquals(3, manual.getRecognitionParallelism());
    }

    @Test(expected = io.github.lxw112190.ppocr.model.OcrException.class)
    public void rejectsNullParallelismPolicy() {
        PaddleOcrOptions.builder().setParallelismMode(null).build();
    }

    @Test(expected = io.github.lxw112190.ppocr.model.OcrException.class)
    public void rejectsNegativeUnifiedParallelism() {
        PaddleOcrOptions.builder().setParallelism(-1).build();
    }

    @Test(expected = io.github.lxw112190.ppocr.model.OcrException.class)
    public void rejectsInvalidThreshold() {
        PaddleOcrOptions.builder().setDetectionBoxThreshold(1.1f).build();
    }

    @Test(expected = io.github.lxw112190.ppocr.model.OcrException.class)
    public void rejectsExcessiveUnclipRatio() {
        PaddleOcrOptions.builder().setDetectionUnclipRatio(10.1f).build();
    }

    @Test(expected = io.github.lxw112190.ppocr.model.OcrException.class)
    public void rejectsInvalidRecognitionParallelism() {
        PaddleOcrOptions.builder().setRecognitionParallelism(0).build();
    }

    @Test(expected = io.github.lxw112190.ppocr.model.OcrException.class)
    public void rejectsInvalidClassificationParallelism() {
        PaddleOcrOptions.builder().setClassificationParallelism(65).build();
    }

    @Test(expected = io.github.lxw112190.ppocr.model.OcrException.class)
    public void rejectsInvalidDetectionMaximumSideLength() {
        PaddleOcrOptions.builder().setDetectionMaximumSideLength(31).build();
    }
}
