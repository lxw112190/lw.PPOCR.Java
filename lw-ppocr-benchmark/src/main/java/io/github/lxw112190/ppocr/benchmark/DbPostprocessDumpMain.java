package io.github.lxw112190.ppocr.benchmark;

import io.github.lxw112190.ppocr.ppocr.DbPostprocess;
import io.github.lxw112190.ppocr.ppocr.DetectionBox;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.List;
import java.util.Locale;

/** Applies Java DB postprocess to an externally produced FP32 probability map. */
public final class DbPostprocessDumpMain {
    private DbPostprocessDumpMain() { }

    public static void main(String[] args) throws Exception {
        if (args.length < 7 || args.length > 10) {
            throw new IllegalArgumentException("usage: output.f32 map-width map-height source-width "
                    + "source-height width-ratio height-ratio [bitmap] [box] [unclip]");
        }
        Path input = Paths.get(args[0]);
        int width = positive(args[1], "map width");
        int height = positive(args[2], "map height");
        int sourceWidth = positive(args[3], "source width");
        int sourceHeight = positive(args[4], "source height");
        float widthRatio = finitePositive(args[5], "width ratio");
        float heightRatio = finitePositive(args[6], "height ratio");
        float bitmap = args.length > 7 ? finite(args[7], "bitmap threshold") : 0.3f;
        float box = args.length > 8 ? finite(args[8], "box threshold") : 0.6f;
        float unclip = args.length > 9 ? finite(args[9], "unclip ratio") : 1.6f;
        float[] probabilities = readFloats(input);
        if ((long) width * height != probabilities.length) {
            throw new IllegalArgumentException("probability map size does not match dimensions");
        }
        List<DetectionBox> boxes = DbPostprocess.createDecoder(width, height).decodeToSource(
                probabilities, bitmap, box, widthRatio, heightRatio, 1000, unclip, false,
                sourceWidth, sourceHeight);
        StringBuilder json = new StringBuilder("{\"boxes\":[");
        for (int i = 0; i < boxes.size(); i++) {
            if (i != 0) json.append(',');
            DetectionBox detection = boxes.get(i);
            json.append("{\"score\":").append(number(detection.getScore())).append(
                    ",\"points\":[");
            float[] points = detection.getPoints();
            for (int p = 0; p < points.length; p++) {
                if (p != 0) json.append(',');
                json.append(number(points[p]));
            }
            json.append("]}");
        }
        System.out.println(json.append("]}").toString());
    }

    private static float[] readFloats(Path path) throws IOException {
        byte[] bytes = Files.readAllBytes(path);
        if ((bytes.length & 3) != 0) throw new IOException("FP32 file has incomplete element");
        ByteBuffer view = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN);
        float[] values = new float[bytes.length / 4];
        for (int i = 0; i < values.length; i++) values[i] = view.getFloat();
        return values;
    }

    private static int positive(String value, String name) {
        int parsed = Integer.parseInt(value);
        if (parsed <= 0) throw new IllegalArgumentException(name + " must be positive");
        return parsed;
    }

    private static float finitePositive(String value, String name) {
        float parsed = finite(value, name);
        if (parsed <= 0.0f) throw new IllegalArgumentException(name + " must be positive");
        return parsed;
    }

    private static float finite(String value, String name) {
        float parsed = Float.parseFloat(value);
        if (!Float.isFinite(parsed)) throw new IllegalArgumentException(name + " must be finite");
        return parsed;
    }

    private static String number(float value) {
        return String.format(Locale.ROOT, "%.9g", value);
    }
}
