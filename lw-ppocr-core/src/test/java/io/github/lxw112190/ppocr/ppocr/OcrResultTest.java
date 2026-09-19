package io.github.lxw112190.ppocr.ppocr;

import java.util.Arrays;
import java.util.ArrayList;
import java.util.List;
import org.junit.Assert;
import org.junit.Test;

public final class OcrResultTest {
    @Test
    public void sortsLinesWithoutLosingRecognitionMetadata() {
        DetectionBox right = box(20, 0);
        DetectionBox left = box(0, 0);
        OcrLineResult rightLine = new OcrLineResult(right, "右", 0.8f, null, false);
        OcrLineResult leftLine = new OcrLineResult(left, "左", 0.9f, null, false);
        OcrResult result = new OcrResult(Arrays.asList(rightLine, leftLine)).sorted(ReadingOrder.HORIZONTAL_LTR);
        Assert.assertEquals("左\n右", result.getText());
        Assert.assertSame(leftLine, result.getLines().get(0));
        Assert.assertEquals(0.8f, result.getLines().get(1).getRecognitionScore(), 0.0f);
    }

    @Test
    public void keepsLineListImmutable() {
        OcrResult result = new OcrResult(Arrays.asList(
                new OcrLineResult(box(0, 0), "text", 1.0f, null, false)));
        try {
            result.getLines().clear();
            Assert.fail("expected immutable OCR result");
        } catch (UnsupportedOperationException expected) {
            // expected
        }
    }

    @Test
    public void keepsSortedLineListImmutable() {
        OcrResult result = new OcrResult(Arrays.asList(
                new OcrLineResult(box(5, 0), "right", 1.0f, null, false),
                new OcrLineResult(box(0, 0), "left", 1.0f, null, false)))
                .sorted(ReadingOrder.HORIZONTAL_LTR);
        try {
            result.getLines().clear();
            Assert.fail("expected immutable sorted OCR result");
        } catch (UnsupportedOperationException expected) {
            // expected
        }
    }

    @Test
    public void internalHorizontalSortMutatesOnlyTheStagingOrder() {
        OcrLineResult right = new OcrLineResult(box(20, 0), "右", 1.0f, null, false);
        OcrLineResult left = new OcrLineResult(box(0, 0), "左", 1.0f, null, false);
        List<OcrLineResult> staging = new ArrayList<OcrLineResult>();
        staging.add(right);
        staging.add(left);

        ReadingOrder.sortLinesInPlace(staging, ReadingOrder.HORIZONTAL_LTR);

        Assert.assertSame(left, staging.get(0));
        Assert.assertSame(right, staging.get(1));
    }

    @Test
    public void rejectsInvalidDetectionGeometry() {
        try {
            new DetectionBox(new float[] {0, 0, Float.NaN, 0, 1, 1, 0, 1}, 0.9f);
            Assert.fail("expected invalid detection geometry");
        } catch (RuntimeException expected) {
            // expected
        }
    }

    private static DetectionBox box(float x, float y) {
        return new DetectionBox(new float[] {x, y, x + 5, y, x + 5, y + 5, x, y + 5}, 0.9f);
    }
}
