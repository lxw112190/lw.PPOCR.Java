package io.github.lxw112190.ppocr.ppocr;

import io.github.lxw112190.ppocr.model.OcrErrorCode;
import io.github.lxw112190.ppocr.model.OcrException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/** Bounded first-stage DB postprocess for probability maps. */
public final class DbPostprocess {
    private DbPostprocess() { }

    /**
     * Extracts 8-connected foreground components and returns axis-aligned
     * quadrilaterals restored to source-image coordinates. This deliberately
     * keeps the first Java milestone deterministic and allocation-bounded.
     */
    public static List<DetectionBox> decode(float[] probabilities, int width, int height,
                                            float bitmapThreshold, float boxThreshold,
                                            float widthRatio, float heightRatio,
                                            int maxCandidates) {
        return decode(probabilities, width, height, bitmapThreshold, boxThreshold,
                widthRatio, heightRatio, maxCandidates, false);
    }

    public static List<DetectionBox> decode(float[] probabilities, int width, int height,
                                            float bitmapThreshold, float boxThreshold,
                                            float widthRatio, float heightRatio,
                                            int maxCandidates, boolean useDilation) {
        validate(probabilities, width, height, bitmapThreshold, boxThreshold,
                widthRatio, heightRatio, maxCandidates);
        boolean[] bitmap = new boolean[probabilities.length];
        for (int i = 0; i < probabilities.length; i++) bitmap[i] = probabilities[i] > bitmapThreshold;
        if (useDilation) dilate2x2(bitmap, width, height);
        boolean[] visited = new boolean[probabilities.length];
        int[] queue = new int[probabilities.length];
        List<DetectionBox> boxes = new ArrayList<DetectionBox>();
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int start = y * width + x;
                if (visited[start] || !bitmap[start]) continue;
                visited[start] = true;
                int head = 0;
                int tail = 0;
                queue[tail++] = start;
                int minX = x;
                int maxX = x;
                int minY = y;
                int maxY = y;
                float scoreSum = 0.0f;
                int count = 0;
                while (head < tail) {
                    int current = queue[head++];
                    int currentY = current / width;
                    int currentX = current - currentY * width;
                    scoreSum += probabilities[current];
                    count++;
                    minX = Math.min(minX, currentX);
                    maxX = Math.max(maxX, currentX);
                    minY = Math.min(minY, currentY);
                    maxY = Math.max(maxY, currentY);
                    for (int deltaY = -1; deltaY <= 1; deltaY++) {
                        for (int deltaX = -1; deltaX <= 1; deltaX++) {
                            if (deltaX == 0 && deltaY == 0) continue;
                            int neighborX = currentX + deltaX;
                            int neighborY = currentY + deltaY;
                            if (neighborX < 0 || neighborX >= width || neighborY < 0 || neighborY >= height) continue;
                            int neighbor = neighborY * width + neighborX;
                            if (!visited[neighbor] && bitmap[neighbor]) {
                                visited[neighbor] = true;
                                queue[tail++] = neighbor;
                            }
                        }
                    }
                }
                float score = scoreSum / count;
                if (score >= boxThreshold) {
                    if (boxes.size() == maxCandidates) {
                        throw new OcrException(OcrErrorCode.RESOURCE_LIMIT, "DB candidate limit exceeded");
                    }
                    boxes.add(new DetectionBox(new float[] {
                            minX / widthRatio, minY / heightRatio,
                            maxX / widthRatio, minY / heightRatio,
                            maxX / widthRatio, maxY / heightRatio,
                            minX / widthRatio, maxY / heightRatio
                    }, score));
                }
            }
        }
        return Collections.unmodifiableList(boxes);
    }

    private static void dilate2x2(boolean[] bitmap, int width, int height) {
        boolean[] dilated = new boolean[bitmap.length];
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int index = y * width + x;
                if (!bitmap[index]) continue;
                dilated[index] = true;
                if (x + 1 < width) dilated[index + 1] = true;
                if (y + 1 < height) dilated[index + width] = true;
                if (x + 1 < width && y + 1 < height) dilated[index + width + 1] = true;
            }
        }
        System.arraycopy(dilated, 0, bitmap, 0, bitmap.length);
    }

    private static void validate(float[] probabilities, int width, int height,
                                 float bitmapThreshold, float boxThreshold,
                                 float widthRatio, float heightRatio, int maxCandidates) {
        if (probabilities == null || width <= 0 || height <= 0 ||
                (long) width * height != probabilities.length ||
                !Float.isFinite(bitmapThreshold) || !Float.isFinite(boxThreshold) ||
                bitmapThreshold < 0.0f || bitmapThreshold > 1.0f ||
                boxThreshold < 0.0f || boxThreshold > 1.0f ||
                !Float.isFinite(widthRatio) || !Float.isFinite(heightRatio) ||
                widthRatio <= 0.0f || heightRatio <= 0.0f || maxCandidates <= 0) {
            throw new OcrException(OcrErrorCode.INVALID_ARGUMENT, "DB probability map or options are invalid");
        }
        for (float probability : probabilities) {
            if (!Float.isFinite(probability)) {
                throw new OcrException(OcrErrorCode.INVALID_ARGUMENT, "DB probability map contains non-finite values");
            }
        }
    }
}
