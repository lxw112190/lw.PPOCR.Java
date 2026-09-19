package io.github.lxw112190.ppocr.ppocr;

import io.github.lxw112190.ppocr.image.BgrImage;
import org.junit.Assert;
import org.junit.Test;

public final class RecWidthPolicyTest {
    @Test
    public void selectsSharedAdaptiveBuckets() {
        BgrImage narrow = image(100, 48);
        BgrImage medium = image(300, 48);
        BgrImage wide = image(700, 48);
        Assert.assertEquals(192, RecWidthPolicy.chooseTargetWidth(narrow, 960));
        Assert.assertEquals(320, RecWidthPolicy.chooseTargetWidth(medium, 960));
        Assert.assertEquals(960, RecWidthPolicy.chooseTargetWidth(wide, 960));
        Assert.assertEquals(320, RecWidthPolicy.chooseTargetWidth(wide, 320));
    }

    @Test
    public void exposesStableSlotsForSharedBuckets() {
        Assert.assertEquals(5, RecWidthPolicy.bucketCount());
        Assert.assertEquals(0, RecWidthPolicy.bucketIndex(192));
        Assert.assertEquals(1, RecWidthPolicy.bucketIndex(320));
        Assert.assertEquals(2, RecWidthPolicy.bucketIndex(480));
        Assert.assertEquals(3, RecWidthPolicy.bucketIndex(640));
        Assert.assertEquals(4, RecWidthPolicy.bucketIndex(960));
        Assert.assertEquals(-1, RecWidthPolicy.bucketIndex(321));
    }

    @Test
    public void choosesBucketSlotsBeforeConvertingToWidths() {
        BgrImage narrow = image(100, 48);
        BgrImage wide = image(700, 48);
        Assert.assertEquals(0, RecWidthPolicy.chooseBucketIndex(narrow, 960));
        Assert.assertEquals(4, RecWidthPolicy.chooseBucketIndex(wide, 960));
        Assert.assertEquals(4, RecWidthPolicy.chooseBucketIndex(wide, 960));
        Assert.assertEquals(320, RecWidthPolicy.widthForIndex(1, 320));
        Assert.assertEquals(321, RecWidthPolicy.widthForIndex(-1, 321));
    }

    private static BgrImage image(int width, int height) {
        return new BgrImage(new byte[width * height * 3], width, height, width * 3);
    }
}
