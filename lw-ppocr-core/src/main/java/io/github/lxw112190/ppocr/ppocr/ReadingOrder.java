package io.github.lxw112190.ppocr.ppocr;

import io.github.lxw112190.ppocr.model.OcrErrorCode;
import io.github.lxw112190.ppocr.model.OcrException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;

/** Deterministic reading-order policies for detection boxes. */
public final class ReadingOrder {
    public static final int HORIZONTAL_LTR = 0;
    public static final int VERTICAL_RTL = 1;
    public static final int VERTICAL_LTR = 2;

    private ReadingOrder() { }

    public static List<DetectionBox> sort(List<DetectionBox> boxes, int order) {
        if (boxes == null || order < HORIZONTAL_LTR || order > VERTICAL_LTR) {
            throw new OcrException(OcrErrorCode.INVALID_ARGUMENT, "boxes and reading order are invalid");
        }
        List<Item> items = new ArrayList<Item>(boxes.size());
        for (int index = 0; index < boxes.size(); index++) {
            DetectionBox box = boxes.get(index);
            if (box == null) throw new OcrException(OcrErrorCode.INVALID_ARGUMENT, "boxes cannot contain null");
            items.add(new Item(box, index));
        }
        if (order == HORIZONTAL_LTR) {
            Collections.sort(items, new Comparator<Item>() {
                @Override public int compare(Item left, Item right) {
                    boolean sameLine = Math.abs(left.minY - right.minY) < 10.0f;
                    int result = Float.compare(sameLine ? left.minX : left.minY,
                            sameLine ? right.minX : right.minY);
                    return result != 0 ? result : Integer.compare(left.originalIndex, right.originalIndex);
                }
            });
        } else {
            assignColumns(items);
            final boolean reverse = order == VERTICAL_RTL;
            Collections.sort(items, new Comparator<Item>() {
                @Override public int compare(Item left, Item right) {
                    int result = Integer.compare(left.column, right.column);
                    if (reverse) result = -result;
                    if (result != 0) return result;
                    result = Float.compare(left.minY, right.minY);
                    if (result != 0) return result;
                    result = Float.compare(left.centerY, right.centerY);
                    return result != 0 ? result : Integer.compare(left.originalIndex, right.originalIndex);
                }
            });
        }
        List<DetectionBox> result = new ArrayList<DetectionBox>(items.size());
        for (Item item : items) result.add(item.box);
        return Collections.unmodifiableList(result);
    }

    private static void assignColumns(List<Item> items) {
        List<Column> columns = new ArrayList<Column>();
        Collections.sort(items, new Comparator<Item>() {
            @Override public int compare(Item left, Item right) {
                int result = Float.compare(left.centerX, right.centerX);
                return result != 0 ? result : Integer.compare(left.originalIndex, right.originalIndex);
            }
        });
        for (Item item : items) {
            Column best = null;
            float bestOverlap = 0.0f;
            for (Column column : columns) {
                float overlap = intervalOverlap(item, column);
                if (overlap >= 0.45f && overlap > bestOverlap) {
                    best = column;
                    bestOverlap = overlap;
                }
            }
            if (best == null) {
                best = new Column(columns.size(), item.minX, item.maxX, item.centerX, item.width);
                columns.add(best);
            }
            item.column = best.index;
            best.minX = Math.min(best.minX, item.minX);
            best.maxX = Math.max(best.maxX, item.maxX);
            best.centerX = (best.centerX * best.count + item.centerX) / (best.count + 1);
            best.width = Math.max(best.width, item.width);
            best.count++;
        }
    }

    private static float intervalOverlap(Item item, Column column) {
        float overlap = Math.min(item.maxX, column.maxX) - Math.max(item.minX, column.minX);
        float denominator = Math.max(item.width, column.width);
        return overlap > 0.0f && denominator > 0.0f ? overlap / denominator : 0.0f;
    }

    private static final class Item {
        private final DetectionBox box;
        private final int originalIndex;
        private final float minX;
        private final float maxX;
        private final float minY;
        private final float centerX;
        private final float centerY;
        private final float width;
        private int column;

        private Item(DetectionBox box, int originalIndex) {
            float x0 = box.x0();
            float y0 = box.y0();
            float x1 = box.x1();
            float y1 = box.y1();
            float x2 = box.x2();
            float y2 = box.y2();
            float x3 = box.x3();
            float y3 = box.y3();
            if (!Float.isFinite(x0) || !Float.isFinite(y0) || !Float.isFinite(x1) ||
                    !Float.isFinite(y1) || !Float.isFinite(x2) || !Float.isFinite(y2) ||
                    !Float.isFinite(x3) || !Float.isFinite(y3)) {
                throw new OcrException(OcrErrorCode.INVALID_ARGUMENT,
                        "detection box contains non-finite coordinates");
            }
            float lowX = Math.min(Math.min(x0, x1), Math.min(x2, x3));
            float highX = Math.max(Math.max(x0, x1), Math.max(x2, x3));
            float lowY = Math.min(Math.min(y0, y1), Math.min(y2, y3));
            float sumX = x0 + x1 + x2 + x3;
            float sumY = y0 + y1 + y2 + y3;
            this.box = box;
            this.originalIndex = originalIndex;
            this.minX = lowX;
            this.maxX = highX;
            this.minY = lowY;
            this.centerX = sumX / 4.0f;
            this.centerY = sumY / 4.0f;
            this.width = Math.max(1.0f, highX - lowX);
        }
    }

    private static final class Column {
        private final int index;
        private float minX;
        private float maxX;
        private float centerX;
        private float width;
        private int count;

        private Column(int index, float minX, float maxX, float centerX, float width) {
            this.index = index;
            this.minX = minX;
            this.maxX = maxX;
            this.centerX = centerX;
            this.width = width;
        }
    }
}
