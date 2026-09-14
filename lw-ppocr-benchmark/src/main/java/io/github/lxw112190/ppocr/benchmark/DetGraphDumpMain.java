package io.github.lxw112190.ppocr.benchmark;

import io.github.lxw112190.ppocr.image.BgrImage;
import io.github.lxw112190.ppocr.imageio.ImageIoLoader;
import io.github.lxw112190.ppocr.kernels.KernelBackend;
import io.github.lxw112190.ppocr.kernels.ScalarBackend;
import io.github.lxw112190.ppocr.model.LwmLoader;
import io.github.lxw112190.ppocr.model.LwmModel;
import io.github.lxw112190.ppocr.ppocr.DetInputShape;
import io.github.lxw112190.ppocr.ppocr.DetInputShapePolicy;
import io.github.lxw112190.ppocr.ppocr.DetPreprocess;
import io.github.lxw112190.ppocr.ppocr.DetPreprocessResult;
import io.github.lxw112190.ppocr.runtime.InferenceSession;
import io.github.lxw112190.ppocr.runtime.TensorShape;
import java.io.BufferedOutputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Collections;
import java.util.Locale;

/** Dumps one Java DET input and graph output for direct comparison with a reference runtime. */
public final class DetGraphDumpMain {
    private DetGraphDumpMain() { }

    public static void main(String[] args) throws Exception {
        if (args.length < 2 || args.length > 4) {
            throw new IllegalArgumentException("usage: image detector [maximum-side] [output-prefix]");
        }
        Path imagePath = Paths.get(args[0]).toAbsolutePath().normalize();
        Path modelPath = Paths.get(args[1]).toAbsolutePath().normalize();
        int maximumSide = args.length > 2 ? positive(args[2], "maximum side") : 960;
        Path prefix = args.length > 3
                ? Paths.get(args[3]).toAbsolutePath().normalize()
                : Paths.get("build-local-data", "det-graph-dump").toAbsolutePath().normalize();
        Path parent = prefix.getParent();
        if (parent != null) Files.createDirectories(parent);

        BgrImage image = ImageIoLoader.load(imagePath);
        DetInputShape shape = DetInputShapePolicy.choose(image, maximumSide);
        DetPreprocessResult preprocess = DetPreprocess.resizeNormalize(
                image, shape.getInputWidth(), shape.getInputHeight());
        KernelBackend backend = new ScalarBackend();
        try (LwmModel model = LwmLoader.load(modelPath)) {
            int outputIndex = model.getGraphOutputs().get(0);
            try (InferenceSession session = new InferenceSession(model,
                    Collections.singletonList(new TensorShape(1, 3,
                            shape.getInputHeight(), shape.getInputWidth())), backend)) {
                TensorShape outputShape = session.execution().shapes().get(outputIndex);
                float[] output = new float[(int) outputShape.getElementCount()];
                session.run(preprocess.getChw(), output);
                int inputIndex = model.getGraphInputs().get(0);
                int outputIndexForPlan = model.getGraphOutputs().get(0);
                System.out.println("workspace bytes=" + session.execution().workspacePlan().getTotalBytes()
                        + " input_offset=" + session.execution().offset(inputIndex)
                        + " output_offset=" + session.execution().offset(outputIndexForPlan));
                writeBytes(prefix.resolveSibling(prefix.getFileName() + ".bgr"), image.pixels(),
                        image.stride() * image.height());
                writeFloats(prefix.resolveSibling(prefix.getFileName() + ".input.f32"),
                        preprocess.getChw());
                writeFloats(prefix.resolveSibling(prefix.getFileName() + ".output.f32"), output);
                writeText(prefix.resolveSibling(prefix.getFileName() + ".json"),
                        manifest(imagePath, modelPath, image, shape, outputShape,
                                preprocess.getChw(), output));
                System.out.println(manifest(imagePath, modelPath, image, shape, outputShape,
                        preprocess.getChw(), output));
            }
        }
    }

    private static String manifest(Path imagePath, Path modelPath, BgrImage image,
                                   DetInputShape shape, TensorShape outputShape,
                                   float[] input, float[] output) {
        return String.format(Locale.ROOT,
                "{\"image\":\"%s\",\"model\":\"%s\","
                        + "\"source\":[%d,%d,%d],\"input_shape\":[1,3,%d,%d],"
                        + "\"output_shape\":\"%s\",\"input\":%s,\"output\":%s}%n",
                jsonEscape(imagePath.toString()), jsonEscape(modelPath.toString()),
                image.width(), image.height(), image.stride(), shape.getInputHeight(),
                shape.getInputWidth(), jsonEscape(outputShape.toString()),
                stats(input), stats(output));
    }

    private static String stats(float[] values) {
        float min = Float.POSITIVE_INFINITY;
        float max = Float.NEGATIVE_INFINITY;
        double sum = 0.0;
        int finite = 0;
        long checksum = 0xcbf29ce484222325L;
        for (float value : values) {
            int bits = Float.floatToIntBits(value);
            checksum ^= bits & 0xffL;
            checksum *= 0x100000001b3L;
            checksum ^= (bits >>> 8) & 0xffL;
            checksum *= 0x100000001b3L;
            checksum ^= (bits >>> 16) & 0xffL;
            checksum *= 0x100000001b3L;
            checksum ^= (bits >>> 24) & 0xffL;
            checksum *= 0x100000001b3L;
            if (Float.isFinite(value)) {
                finite++;
                min = Math.min(min, value);
                max = Math.max(max, value);
                sum += value;
            }
        }
        return String.format(Locale.ROOT,
                "{\"length\":%d,\"finite\":%d,\"min\":%.9g,\"max\":%.9g,"
                        + "\"mean\":%.9g,\"fnv1a_float32\":\"%016x\"}",
                values.length, finite, min, max,
                values.length == 0 ? 0.0 : sum / values.length, checksum);
    }

    private static void writeFloats(Path path, float[] values) throws IOException {
        try (OutputStream output = new BufferedOutputStream(Files.newOutputStream(path))) {
            for (float value : values) {
                int bits = Float.floatToIntBits(value);
                output.write(bits & 0xff);
                output.write((bits >>> 8) & 0xff);
                output.write((bits >>> 16) & 0xff);
                output.write((bits >>> 24) & 0xff);
            }
        }
    }

    private static void writeBytes(Path path, byte[] values, int length) throws IOException {
        try (OutputStream output = new BufferedOutputStream(Files.newOutputStream(path))) {
            output.write(values, 0, length);
        }
    }

    private static void writeText(Path path, String value) throws IOException {
        Files.write(path, value.getBytes(java.nio.charset.StandardCharsets.UTF_8));
    }

    private static int positive(String value, String name) {
        int parsed = Integer.parseInt(value);
        if (parsed <= 0) throw new IllegalArgumentException(name + " must be positive");
        return parsed;
    }

    private static String jsonEscape(String value) {
        return value.replace("\\", "\\\\").replace("\"", "\\\"");
    }
}
