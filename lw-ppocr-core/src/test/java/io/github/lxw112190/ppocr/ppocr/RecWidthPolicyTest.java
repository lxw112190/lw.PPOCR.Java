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

    private static BgrImage image(int width, int height) {
        return new BgrImage(new byte[width * height * 3], width, height, width * 3);
    }
}
