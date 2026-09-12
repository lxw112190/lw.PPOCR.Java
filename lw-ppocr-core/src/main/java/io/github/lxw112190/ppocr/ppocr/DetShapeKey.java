package io.github.lxw112190.ppocr.ppocr;

import java.util.Objects;

/** Value key for a dynamic DET session shape. */
final class DetShapeKey {
    private final int height;
    private final int width;

    DetShapeKey(int height, int width) {
        this.height = height;
        this.width = width;
    }

    int height() { return height; }
    int width() { return width; }

    @Override
    public boolean equals(Object other) {
        if (!(other instanceof DetShapeKey)) return false;
        DetShapeKey key = (DetShapeKey) other;
        return height == key.height && width == key.width;
    }

    @Override
    public int hashCode() { return Objects.hash(height, width); }
}
