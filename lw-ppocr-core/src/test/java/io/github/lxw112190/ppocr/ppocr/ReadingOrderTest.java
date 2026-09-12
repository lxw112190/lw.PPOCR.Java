package io.github.lxw112190.ppocr.ppocr;

import java.util.Arrays;
import java.util.List;
import org.junit.Assert;
import org.junit.Test;

public final class ReadingOrderTest {
    @Test
    public void sortsHorizontalLinesLeftToRight() {
        DetectionBox firstLineRight = box(20, 0, 5, 5);
        DetectionBox firstLineLeft = box(1, 2, 5, 5);
        DetectionBox secondLine = box(0, 30, 5, 5);
        List<DetectionBox> input = Arrays.asList(firstLineRight, firstLineLeft, secondLine);
        List<DetectionBox> sorted = ReadingOrder.sort(input, ReadingOrder.HORIZONTAL_LTR);
        Assert.assertSame(firstLineLeft, sorted.get(0));
        Assert.assertSame(firstLineRight, sorted.get(1));
        Assert.assertSame(secondLine, sorted.get(2));
        Assert.assertSame(firstLineRight, input.get(0));
    }

    @Test
    public void sortsVerticalColumnsInBothDirections() {
        DetectionBox leftTop = box(0, 20, 8, 8);
        DetectionBox rightTop = box(30, 10, 8, 8);
        DetectionBox leftBottom = box(0, 40, 8, 8);
        DetectionBox rightBottom = box(30, 30, 8, 8);
        List<DetectionBox> input = Arrays.asList(leftBottom, rightBottom, rightTop, leftTop);
        List<DetectionBox> leftToRight = ReadingOrder.sort(input, ReadingOrder.VERTICAL_LTR);
        Assert.assertSame(leftTop, leftToRight.get(0));
        Assert.assertSame(leftBottom, leftToRight.get(1));
        Assert.assertSame(rightTop, leftToRight.get(2));
        Assert.assertSame(rightBottom, leftToRight.get(3));
        List<DetectionBox> rightToLeft = ReadingOrder.sort(input, ReadingOrder.VERTICAL_RTL);
        Assert.assertSame(rightTop, rightToLeft.get(0));
        Assert.assertSame(rightBottom, rightToLeft.get(1));
        Assert.assertSame(leftTop, rightToLeft.get(2));
        Assert.assertSame(leftBottom, rightToLeft.get(3));
    }

    private static DetectionBox box(float x, float y, float width, float height) {
        return new DetectionBox(new float[] {x, y, x + width, y, x + width, y + height, x, y + height}, 0.9f);
    }
}
