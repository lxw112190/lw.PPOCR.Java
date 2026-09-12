package io.github.lxw112190.ppocr.ppocr;

import io.github.lxw112190.ppocr.model.OcrErrorCode;
import io.github.lxw112190.ppocr.model.OcrException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

/** Bounded first-stage DB postprocess for probability maps. */
public final class DbPostprocess {
    private DbPostprocess() { }

    /** Creates a reusable, single-threaded decoder for one probability-map shape. */
    public static Decoder createDecoder(int width, int height) {
        return new Decoder(width, height);
    }

    /**
     * Extracts 8-connected foreground components, fits their minimum rotated
     * rectangles, and restores quadrilaterals to source-image coordinates.
     * The implementation remains deterministic and allocation-bounded.
     */
    public static List<DetectionBox> decode(float[] probabilities, int width, int height,
                                            float bitmapThreshold, float boxThreshold,
                                            float widthRatio, float heightRatio,
                                            int maxCandidates) {
        return decode(probabilities, width, height, bitmapThreshold, boxThreshold,
                widthRatio, heightRatio, maxCandidates, 1.0f, false);
    }

    public static List<DetectionBox> decode(float[] probabilities, int width, int height,
                                            float bitmapThreshold, float boxThreshold,
                                            float widthRatio, float heightRatio,
                                            int maxCandidates, boolean useDilation) {
        return decode(probabilities, width, height, bitmapThreshold, boxThreshold,
                widthRatio, heightRatio, maxCandidates, 1.0f, useDilation);
    }

    public static List<DetectionBox> decode(float[] probabilities, int width, int height,
                                            float bitmapThreshold, float boxThreshold,
                                            float widthRatio, float heightRatio,
                                            int maxCandidates, float unclipRatio,
                                            boolean useDilation) {
        validate(probabilities, width, height, bitmapThreshold, boxThreshold,
                widthRatio, heightRatio, maxCandidates, unclipRatio);
        return decodeInternal(probabilities, width, height, bitmapThreshold, boxThreshold,
                widthRatio, heightRatio, maxCandidates, unclipRatio, useDilation,
                new Scratch(width, height));
    }

    private static List<DetectionBox> decodeInternal(float[] probabilities, int width, int height,
                                                     float bitmapThreshold, float boxThreshold,
                                                     float widthRatio, float heightRatio,
                                                     int maxCandidates, float unclipRatio,
                                                     boolean useDilation, Scratch scratch) {
        boolean[] bitmap = scratch.bitmap;
        for (int i = 0; i < probabilities.length; i++) bitmap[i] = probabilities[i] > bitmapThreshold;
        if (useDilation) dilate2x2(bitmap, width, height);
        boolean[] visited = scratch.visited;
        Arrays.fill(visited, false);
        int[] queue = scratch.queue;
        long[] componentPoints = scratch.componentPoints;
        long[] hull = scratch.hull;
        double[] corners = scratch.corners;
        double[] sortedCorners = scratch.sortedCorners;
        List<DetectionBox> boxes = new ArrayList<DetectionBox>();
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int start = y * width + x;
                if (visited[start] || !bitmap[start]) continue;
                visited[start] = true;
                int head = 0;
                int tail = 0;
                queue[tail++] = start;
                while (head < tail) {
                    int current = queue[head++];
                    int currentY = current / width;
                    int currentX = current - currentY * width;
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
                int pointCount = boundaryPoints(bitmap, width, height, queue, tail, componentPoints);
                int hullCount = convexHull(componentPoints, pointCount, hull);
                Rectangle rectangle = minimumRectangle(hull, hullCount);
                if (rectangle == null) rectangle = axisAlignedRectangle(componentPoints, pointCount);
                float score = rectangleScore(probabilities, width, height, rectangle, corners);
                if (score >= boxThreshold) {
                    if (boxes.size() == maxCandidates) {
                        throw new OcrException(OcrErrorCode.RESOURCE_LIMIT, "DB candidate limit exceeded");
                    }
                    float expansion = expansion(rectangle, unclipRatio);
                    rectanglePoints(rectangle, expansion, corners);
                    orderClockwise(corners, sortedCorners);
                    float[] restored = new float[8];
                    for (int point = 0; point < 4; point++) {
                        restored[point * 2] = (float) (clamp(corners[point * 2], width - 1.0)
                                / widthRatio);
                        restored[point * 2 + 1] = (float) (clamp(corners[point * 2 + 1], height - 1.0)
                                / heightRatio);
                    }
                    boxes.add(new DetectionBox(restored, score));
                }
            }
        }
        return Collections.unmodifiableList(boxes);
    }

    private static int boundaryPoints(boolean[] bitmap, int width, int height, int[] component,
                                      int count, long[] output) {
        int pointCount = 0;
        for (int i = 0; i < count; i++) {
            int position = component[i];
            int y = position / width;
            int x = position - y * width;
            boolean boundary = x == 0 || y == 0 || x + 1 == width || y + 1 == height;
            if (!boundary && bitmap[position - 1] == false) boundary = true;
            if (!boundary && x + 1 < width && !bitmap[position + 1]) boundary = true;
            if (!boundary && y > 0 && !bitmap[position - width]) boundary = true;
            if (!boundary && y + 1 < height && !bitmap[position + width]) boundary = true;
            if (boundary) output[pointCount++] = encodePoint(x, y);
        }
        return pointCount;
    }

    private static int convexHull(long[] points, int pointCount, long[] hull) {
        Arrays.sort(points, 0, pointCount);
        int uniqueCount = 0;
        for (int i = 0; i < pointCount; i++) {
            if (uniqueCount == 0 || points[i] != points[uniqueCount - 1]) {
                points[uniqueCount++] = points[i];
            }
        }
        if (uniqueCount < 3) return 0;
        int count = 0;
        for (int i = 0; i < uniqueCount; i++) {
            while (count >= 2 && cross(hull[count - 2], hull[count - 1], points[i]) <= 0.0) count--;
            hull[count++] = points[i];
        }
        int lowerCount = count;
        for (int i = uniqueCount - 1; i > 0; i--) {
            long point = points[i - 1];
            while (count > lowerCount && cross(hull[count - 2], hull[count - 1], point) <= 0.0) count--;
            hull[count++] = point;
        }
        return count > 1 ? count - 1 : 0;
    }

    private static Rectangle minimumRectangle(long[] hull, int hullCount) {
        if (hullCount < 3) return null;
        Rectangle best = null;
        double bestArea = Double.POSITIVE_INFINITY;
        for (int edge = 0; edge < hullCount; edge++) {
            long current = hull[edge];
            long next = hull[(edge + 1) % hullCount];
            double dx = pointX(next) - pointX(current);
            double dy = pointY(next) - pointY(current);
            double length = Math.hypot(dx, dy);
            if (length <= 0.0) continue;
            double ux = dx / length;
            double uy = dy / length;
            double vx = -uy;
            double vy = ux;
            double minU = Double.POSITIVE_INFINITY;
            double maxU = Double.NEGATIVE_INFINITY;
            double minV = Double.POSITIVE_INFINITY;
            double maxV = Double.NEGATIVE_INFINITY;
            for (int point = 0; point < hullCount; point++) {
                double x = pointX(hull[point]);
                double y = pointY(hull[point]);
                double projectionU = x * ux + y * uy;
                double projectionV = x * vx + y * vy;
                minU = Math.min(minU, projectionU);
                maxU = Math.max(maxU, projectionU);
                minV = Math.min(minV, projectionV);
                maxV = Math.max(maxV, projectionV);
            }
            double area = (maxU - minU) * (maxV - minV);
            if (area > 0.0 && area < bestArea) {
                bestArea = area;
                best = new Rectangle(ux, uy, vx, vy, minU, maxU, minV, maxV);
            }
        }
        return best;
    }

    private static Rectangle axisAlignedRectangle(long[] points, int pointCount) {
        double minX = pointX(points[0]);
        double maxX = minX;
        double minY = pointY(points[0]);
        double maxY = minY;
        for (int i = 1; i < pointCount; i++) {
            minX = Math.min(minX, pointX(points[i]));
            maxX = Math.max(maxX, pointX(points[i]));
            minY = Math.min(minY, pointY(points[i]));
            maxY = Math.max(maxY, pointY(points[i]));
        }
        return new Rectangle(1.0, 0.0, 0.0, 1.0, minX, maxX, minY, maxY);
    }

    private static float rectangleScore(float[] probabilities, int width, int height,
                                        Rectangle rectangle, double[] corners) {
        rectanglePoints(rectangle, 0.0f, corners);
        double minX = corners[0];
        double maxX = corners[0];
        double minY = corners[1];
        double maxY = corners[1];
        for (int point = 1; point < 4; point++) {
            minX = Math.min(minX, corners[point * 2]);
            maxX = Math.max(maxX, corners[point * 2]);
            minY = Math.min(minY, corners[point * 2 + 1]);
            maxY = Math.max(maxY, corners[point * 2 + 1]);
        }
        int left = Math.max(0, (int) Math.floor(minX));
        int right = Math.min(width - 1, (int) Math.ceil(maxX));
        int top = Math.max(0, (int) Math.floor(minY));
        int bottom = Math.min(height - 1, (int) Math.ceil(maxY));
        double sum = 0.0;
        int count = 0;
        for (int y = top; y <= bottom; y++) {
            for (int x = left; x <= right; x++) {
                double projectionU = x * rectangle.ux + y * rectangle.uy;
                double projectionV = x * rectangle.vx + y * rectangle.vy;
                if (projectionU >= rectangle.minU && projectionU <= rectangle.maxU &&
                        projectionV >= rectangle.minV && projectionV <= rectangle.maxV) {
                    sum += probabilities[y * width + x];
                    count++;
                }
            }
        }
        return count == 0 ? 0.0f : (float) (sum / count);
    }

    private static float expansion(Rectangle rectangle, float unclipRatio) {
        if (unclipRatio <= 1.0f) return 0.0f;
        double rectangleWidth = rectangle.maxU - rectangle.minU;
        double rectangleHeight = rectangle.maxV - rectangle.minV;
        double perimeter = 2.0 * (rectangleWidth + rectangleHeight);
        return perimeter <= 0.0 ? 0.0f : (float) (rectangleWidth * rectangleHeight
                * (unclipRatio - 1.0f) / perimeter);
    }

    private static void rectanglePoints(Rectangle rectangle, float expansion, double[] points) {
        fromProjection(rectangle, rectangle.minU - expansion, rectangle.minV - expansion, points, 0);
        fromProjection(rectangle, rectangle.maxU + expansion, rectangle.minV - expansion, points, 2);
        fromProjection(rectangle, rectangle.maxU + expansion, rectangle.maxV + expansion, points, 4);
        fromProjection(rectangle, rectangle.minU - expansion, rectangle.maxV + expansion, points, 6);
    }

    private static void fromProjection(Rectangle rectangle, double projectionU, double projectionV,
                                       double[] points, int offset) {
        points[offset] = projectionU * rectangle.ux + projectionV * rectangle.vx;
        points[offset + 1] = projectionU * rectangle.uy + projectionV * rectangle.vy;
    }

    private static double clamp(double value, double maximum) {
        if (value < 0.0) return 0.0;
        return value > maximum ? maximum : value;
    }

    /** Reusable decoder; callers must not invoke it concurrently. */
    public static final class Decoder {
        private final int width;
        private final int height;
        private final Scratch scratch;

        private Decoder(int width, int height) {
            this.width = width;
            this.height = height;
            this.scratch = new Scratch(width, height);
        }

        public List<DetectionBox> decode(float[] probabilities, float bitmapThreshold,
                                         float boxThreshold, float widthRatio,
                                         float heightRatio, int maxCandidates,
                                         float unclipRatio, boolean useDilation) {
            validate(probabilities, width, height, bitmapThreshold, boxThreshold,
                    widthRatio, heightRatio, maxCandidates, unclipRatio);
            return decodeInternal(probabilities, width, height, bitmapThreshold, boxThreshold,
                    widthRatio, heightRatio, maxCandidates, unclipRatio, useDilation, scratch);
        }
    }

    private static final class Scratch {
        private final boolean[] bitmap;
        private final boolean[] visited;
        private final int[] queue;
        private final long[] componentPoints;
        private final long[] hull;
        private final double[] corners;
        private final double[] sortedCorners;

        private Scratch(int width, int height) {
            if (width <= 0 || height <= 0) {
                throw new OcrException(OcrErrorCode.INVALID_ARGUMENT, "DB map dimensions are invalid");
            }
            long pixelCount = (long) width * height;
            if (pixelCount > Integer.MAX_VALUE || pixelCount > Integer.MAX_VALUE / 2) {
                throw new OcrException(OcrErrorCode.RESOURCE_LIMIT, "DB component geometry is too large");
            }
            int count = (int) pixelCount;
            this.bitmap = new boolean[count];
            this.visited = new boolean[count];
            this.queue = new int[count];
            this.componentPoints = new long[count];
            this.hull = new long[count * 2];
            this.corners = new double[8];
            this.sortedCorners = new double[8];
        }
    }

    private static void orderClockwise(double[] points, double[] sorted) {
        System.arraycopy(points, 0, sorted, 0, points.length);
        for (int i = 1; i < 4; i++) {
            double x = sorted[i * 2];
            double y = sorted[i * 2 + 1];
            int j = i - 1;
            while (j >= 0 && (x < sorted[j * 2] ||
                    (x == sorted[j * 2] && y < sorted[j * 2 + 1]))) {
                sorted[(j + 1) * 2] = sorted[j * 2];
                sorted[(j + 1) * 2 + 1] = sorted[j * 2 + 1];
                j--;
            }
            sorted[(j + 1) * 2] = x;
            sorted[(j + 1) * 2 + 1] = y;
        }
        int leftTop = sorted[1] < sorted[3] ? 0 : 2;
        int leftBottom = leftTop == 0 ? 2 : 0;
        int rightTop = sorted[5] < sorted[7] ? 4 : 6;
        int rightBottom = rightTop == 4 ? 6 : 4;
        points[0] = sorted[leftTop];
        points[1] = sorted[leftTop + 1];
        points[2] = sorted[rightTop];
        points[3] = sorted[rightTop + 1];
        points[4] = sorted[rightBottom];
        points[5] = sorted[rightBottom + 1];
        points[6] = sorted[leftBottom];
        points[7] = sorted[leftBottom + 1];
    }

    private static long encodePoint(int x, int y) {
        return ((long) x << 32) | (y & 0xffffffffL);
    }

    private static double pointX(long point) { return (double) (point >> 32); }
    private static double pointY(long point) { return (double) (point & 0xffffffffL); }

    private static double cross(long origin, long first, long second) {
        return (pointX(first) - pointX(origin)) * (pointY(second) - pointY(origin)) -
                (pointY(first) - pointY(origin)) * (pointX(second) - pointX(origin));
    }

    private static final class Rectangle {
        private final double ux;
        private final double uy;
        private final double vx;
        private final double vy;
        private final double minU;
        private final double maxU;
        private final double minV;
        private final double maxV;

        private Rectangle(double ux, double uy, double vx, double vy,
                          double minU, double maxU, double minV, double maxV) {
            this.ux = ux;
            this.uy = uy;
            this.vx = vx;
            this.vy = vy;
            this.minU = minU;
            this.maxU = maxU;
            this.minV = minV;
            this.maxV = maxV;
        }
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
                                 float widthRatio, float heightRatio, int maxCandidates,
                                 float unclipRatio) {
        if (probabilities == null || width <= 0 || height <= 0 ||
                (long) width * height != probabilities.length ||
                !Float.isFinite(bitmapThreshold) || !Float.isFinite(boxThreshold) ||
                bitmapThreshold < 0.0f || bitmapThreshold > 1.0f ||
                boxThreshold < 0.0f || boxThreshold > 1.0f ||
                !Float.isFinite(widthRatio) || !Float.isFinite(heightRatio) ||
                widthRatio <= 0.0f || heightRatio <= 0.0f || maxCandidates <= 0 ||
                !Float.isFinite(unclipRatio) || unclipRatio <= 0.0f || unclipRatio > 10.0f) {
            throw new OcrException(OcrErrorCode.INVALID_ARGUMENT, "DB probability map or options are invalid");
        }
        for (float probability : probabilities) {
            if (!Float.isFinite(probability)) {
                throw new OcrException(OcrErrorCode.INVALID_ARGUMENT, "DB probability map contains non-finite values");
            }
        }
    }
}
