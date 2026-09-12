package io.github.lxw112190.ppocr.ppocr;

import io.github.lxw112190.ppocr.model.OcrErrorCode;
import io.github.lxw112190.ppocr.model.OcrException;
import java.util.List;
import org.junit.Assert;
import org.junit.Test;

public final class DbPostprocessTest {
    @Test
    public void extractsEightConnectedComponentAndRestoresCoordinates() {
        float[] probabilities = {
                0.0f, 0.0f, 0.0f, 0.0f,
                0.0f, 0.9f, 0.8f, 0.0f,
                0.0f, 0.7f, 0.9f, 0.0f
        };
        List<DetectionBox> boxes = DbPostprocess.decode(probabilities, 4, 3,
                0.5f, 0.7f, 2.0f, 3.0f, 4);
        Assert.assertEquals(1, boxes.size());
        Assert.assertArrayEquals(new float[] {0.375f, 0.25f, 1.125f, 0.25f,
                        1.125f, 0.6666667f, 0.375f, 0.6666667f},
                boxes.get(0).getPoints(), 0.00001f);
        Assert.assertEquals(0.825f, boxes.get(0).getScore(), 0.00001f);
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
        try {
            DbPostprocess.decode(new float[] {0.9f, 0.0f, 0.9f}, 3, 1,
                    0.5f, 0.5f, 1.0f, 1.0f, 1);
            Assert.fail("expected candidate limit failure");
        } catch (OcrException e) {
            Assert.assertEquals(OcrErrorCode.RESOURCE_LIMIT, e.getCode());
        }
    }

    @Test
    public void optionalTwoByTwoDilationJoinsForwardNeighbors() {
        List<DetectionBox> boxes = DbPostprocess.decode(new float[] {0.9f, 0.0f, 0.9f},
                3, 1, 0.5f, 0.5f, 1.0f, 1.0f, 1, true);
        Assert.assertEquals(1, boxes.size());
        Assert.assertArrayEquals(new float[] {0, 0, 2, 0, 2, 0, 0, 0},
                boxes.get(0).getPoints(), 0.0f);
    }

    @Test
    public void expandsComponentByUnclipRatioAndClampsToMap() {
        List<DetectionBox> boxes = DbPostprocess.decode(new float[] {
                        0.0f, 0.0f, 0.0f,
                        0.0f, 0.9f, 0.9f,
                        0.0f, 0.9f, 0.9f
                }, 3, 3, 0.5f, 0.5f, 1.0f, 1.0f, 2, 2.0f, false);
        Assert.assertEquals(1, boxes.size());
                Assert.assertArrayEquals(new float[] {0.5f, 0.5f, 2.0f, 0.5f,
                        2.0f, 2.0f, 0.5f, 2.0f},
                boxes.get(0).getPoints(), 0.00001f);
    }

    @Test
    public void clampsRestoredCoordinatesToSourceBounds() {
        DbPostprocess.Decoder decoder = DbPostprocess.createDecoder(3, 3);
        List<DetectionBox> boxes = decoder.decodeToSource(new float[] {
                        0.0f, 0.0f, 0.0f,
                        0.0f, 0.9f, 0.9f,
                        0.0f, 0.9f, 0.9f
                }, 0.5f, 0.5f, 1.5f, 1.5f, 2, 2.0f, false, 2, 2);
        Assert.assertEquals(1, boxes.size());
        for (float point : boxes.get(0).getPoints()) {
            Assert.assertTrue(point >= 0.0f && point <= 1.0f);
        }
    }

    @Test
    public void fitsRotatedRectangleToSlantedComponent() {
        float[] probabilities = new float[8 * 6];
        int[] foreground = {1 * 8 + 2, 1 * 8 + 3, 2 * 8 + 3, 2 * 8 + 4,
                3 * 8 + 4, 3 * 8 + 5, 4 * 8 + 5, 4 * 8 + 6};
        for (int index : foreground) probabilities[index] = 0.9f;

        List<DetectionBox> boxes = DbPostprocess.decode(probabilities, 8, 6,
                0.5f, 0.5f, 1.0f, 1.0f, 4, 1.0f, false);

        Assert.assertEquals(1, boxes.size());
        float[] points = boxes.get(0).getPoints();
        Assert.assertNotEquals(points[1], points[3], 0.00001f);
        Assert.assertNotEquals(points[3], points[5], 0.00001f);
    }

    @Test
    public void reusableDecoderClearsPreviousVisitState() {
        DbPostprocess.Decoder decoder = DbPostprocess.createDecoder(3, 3);
        Assert.assertEquals(1, decoder.decode(new float[] {
                0.9f, 0.9f, 0.0f,
                0.9f, 0.9f, 0.0f,
                0.0f, 0.0f, 0.0f
        }, 0.5f, 0.5f, 1.0f, 1.0f, 4, 1.0f, false).size());
        Assert.assertEquals(0, decoder.decode(new float[9], 0.5f, 0.5f,
                1.0f, 1.0f, 4, 1.0f, false).size());
    }
}
