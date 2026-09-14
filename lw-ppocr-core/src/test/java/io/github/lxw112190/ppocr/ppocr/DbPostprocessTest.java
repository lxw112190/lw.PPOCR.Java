package io.github.lxw112190.ppocr.ppocr;

import io.github.lxw112190.ppocr.model.OcrErrorCode;
import io.github.lxw112190.ppocr.model.OcrException;
import java.util.List;
import org.junit.Assert;
import org.junit.Test;

public final class DbPostprocessTest {
    @Test
    public void extractsEightConnectedComponentAndRestoresCoordinates() {
        float[] probabilities = rectangle(10, 10, 1, 1, 8, 8, 0.9f);
        List<DetectionBox> boxes = DbPostprocess.decode(probabilities, 10, 10,
                0.5f, 0.7f, 1.5f, 1.5f, 4);
        Assert.assertEquals(1, boxes.size());
        Assert.assertArrayEquals(new float[] {0.0f, 0.0f, 6.0f, 0.0f,
                        6.0f, 6.0f, 0.0f, 6.0f},
                boxes.get(0).getPoints(), 0.00001f);
        Assert.assertEquals(0.9f, boxes.get(0).getScore(), 0.00001f);
    }

    @Test
    public void rejectsNonFiniteMapsAndCandidateOverflow() {
        try {
            DbPostprocess.decode(new float[] {Float.NaN}, 1, 1,
                    0.5f, 0.5f, 1.0f, 1.0f, 1);
            Assert.fail("expected invalid probability map");
        } catch (OcrException e) {
            Assert.assertEquals(OcrErrorCode.INVALID_ARGUMENT, e.getCode());
        }
        Assert.assertEquals(0, DbPostprocess.decode(new float[] {0.9f, 0.0f, 0.9f},
                3, 1, 0.5f, 0.5f, 1.0f, 1.0f, 1).size());
        try {
            DbPostprocess.decode(rectangle(20, 10, 1, 1, 8, 8, 0.9f,
                            11, 1, 18, 8),
                    20, 10, 0.5f, 0.5f, 1.0f, 1.0f, 1);
            Assert.fail("expected candidate limit failure");
        } catch (OcrException e) {
            Assert.assertEquals(OcrErrorCode.RESOURCE_LIMIT, e.getCode());
        }
    }

    @Test
    public void optionalTwoByTwoDilationJoinsForwardNeighbors() {
        List<DetectionBox> boxes = DbPostprocess.decode(
                rectangle(16, 10, 1, 1, 5, 8, 0.9f, 7, 1, 11, 8),
                16, 10, 0.5f, 0.5f, 1.0f, 1.0f, 4, 1.0f, true);
        Assert.assertEquals(1, boxes.size());
        Assert.assertTrue(boxes.get(0).getPoints()[2] - boxes.get(0).getPoints()[0] > 4.0f);
    }

    @Test
    public void expandsComponentByUnclipRatioAndClampsToMap() {
        List<DetectionBox> boxes = DbPostprocess.decode(
                rectangle(12, 12, 2, 2, 9, 9, 0.9f),
                12, 12, 0.5f, 0.5f, 1.0f, 1.0f, 2, 2.0f, false);
        Assert.assertEquals(1, boxes.size());
                Assert.assertArrayEquals(new float[] {0.0f, 0.0f, 11.0f, 0.0f,
                        11.0f, 11.0f, 0.0f, 11.0f},
                boxes.get(0).getPoints(), 0.00001f);
    }

    @Test
    public void clampsRestoredCoordinatesToSourceBounds() {
        DbPostprocess.Decoder decoder = DbPostprocess.createDecoder(12, 12);
        List<DetectionBox> boxes = decoder.decodeToSource(
                rectangle(12, 12, 2, 2, 9, 9, 0.9f),
                0.5f, 0.5f, 1.5f, 1.5f, 2, 2.0f, false, 8, 8);
        Assert.assertEquals(1, boxes.size());
        for (float point : boxes.get(0).getPoints()) {
            Assert.assertTrue(point >= 0.0f && point <= 7.0f);
        }
    }

    @Test
    public void fitsRotatedRectangleToSlantedComponent() {
        float[] probabilities = new float[24 * 24];
        for (int y = 2; y <= 12; y++) {
            for (int x = y; x <= y + 7; x++) probabilities[y * 24 + x] = 0.9f;
        }

        List<DetectionBox> boxes = DbPostprocess.decode(probabilities, 24, 24,
                0.5f, 0.5f, 1.0f, 1.0f, 4, 1.0f, false);

        Assert.assertEquals(1, boxes.size());
        float[] points = boxes.get(0).getPoints();
        Assert.assertNotEquals(points[1], points[3], 0.00001f);
        Assert.assertNotEquals(points[3], points[5], 0.00001f);
    }

    @Test
    public void reusableDecoderClearsPreviousVisitState() {
        DbPostprocess.Decoder decoder = DbPostprocess.createDecoder(12, 12);
        Assert.assertEquals(1, decoder.decode(rectangle(12, 12, 1, 1, 8, 8, 0.9f),
                0.5f, 0.5f, 1.0f, 1.0f, 4, 1.0f, false).size());
        Assert.assertEquals(0, decoder.decode(new float[12 * 12], 0.5f, 0.5f,
                1.0f, 1.0f, 4, 1.0f, false).size());
    }

    @Test
    public void filtersCandidatesAtEachPaddleMinimumSideStage() {
        Assert.assertEquals(0, DbPostprocess.decode(
                rectangle(12, 12, 2, 2, 9, 4, 0.9f),
                12, 12, 0.5f, 0.5f, 1.0f, 1.0f, 4, 1.0f, false).size());

        Assert.assertEquals(0, DbPostprocess.decode(
                rectangle(12, 12, 2, 2, 9, 5, 0.9f),
                12, 12, 0.5f, 0.5f, 1.0f, 1.0f, 4, 0.1f, false).size());

        DbPostprocess.Decoder decoder = DbPostprocess.createDecoder(12, 12);
        Assert.assertEquals(0, decoder.decodeToSource(
                rectangle(12, 12, 1, 1, 8, 8, 0.9f),
                0.5f, 0.5f, 1.0f, 1.0f, 4, 1.0f, false, 4, 4).size());
    }

    private static float[] rectangle(int width, int height, int left, int top,
                                      int right, int bottom, float value, int... extra) {
        float[] probabilities = new float[width * height];
        fill(probabilities, width, left, top, right, bottom, value);
        for (int index = 0; index < extra.length; index += 4) {
            fill(probabilities, width, extra[index], extra[index + 1], extra[index + 2],
                    extra[index + 3], value);
        }
        return probabilities;
    }

    private static void fill(float[] probabilities, int width, int left, int top,
                              int right, int bottom, float value) {
        for (int y = top; y <= bottom; y++) {
            for (int x = left; x <= right; x++) probabilities[y * width + x] = value;
        }
    }
}
