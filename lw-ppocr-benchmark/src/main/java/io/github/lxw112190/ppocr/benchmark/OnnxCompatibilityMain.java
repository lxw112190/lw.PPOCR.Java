package io.github.lxw112190.ppocr.benchmark;

import io.github.lxw112190.ppocr.model.ModelLoader;
import io.github.lxw112190.ppocr.model.LwmModel;
import io.github.lxw112190.ppocr.runtime.ShapeResolver;
import io.github.lxw112190.ppocr.runtime.TensorShape;
import io.github.lxw112190.ppocr.runtime.InferenceSession;
import io.github.lxw112190.ppocr.kernels.KernelBackend;
import io.github.lxw112190.ppocr.kernels.ScalarBackend;
import io.github.lxw112190.ppocr.ppocr.PaddleOcr;
import io.github.lxw112190.ppocr.ppocr.PaddleOcrOptions;
import io.github.lxw112190.ppocr.ppocr.OcrResult;
import io.github.lxw112190.ppocr.ppocr.OcrLineResult;
import io.github.lxw112190.ppocr.ppocr.PerspectiveCrop;
import io.github.lxw112190.ppocr.ppocr.RecWidthPolicy;
import io.github.lxw112190.ppocr.ppocr.RecPreprocess;
import io.github.lxw112190.ppocr.ppocr.ClsPreprocess;
import io.github.lxw112190.ppocr.image.BgrImage;
import io.github.lxw112190.ppocr.image.BgrTransforms;
import io.github.lxw112190.ppocr.imageio.ImageIoLoader;
import io.github.lxw112190.ppocr.imageio.PaddleOcrImageIo;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.ArrayList;
import java.util.Base64;
import java.util.Locale;

/** Real-model validation; models are supplied explicitly, never from local C build paths. */
public final class OnnxCompatibilityMain {
    private OnnxCompatibilityMain() { }
    public static void main(String[] args) throws Exception {
        if (args.length < 1 || args.length > 4) throw new IllegalArgumentException("models-root [fixtures-root [scalar|vector [record-stage-inputs]]]");
        boolean record = args.length == 4 && args[3].equals("record-stage-inputs");
        if (args.length == 4 && !record) throw new IllegalArgumentException("invalid validation mode");
        Path root = Paths.get(args[0]);
        KernelBackend backend = args.length >= 3 && args[2].equals("vector")
                ? (KernelBackend) Class.forName("io.github.lxw112190.ppocr.vector.VectorBackend").getDeclaredConstructor().newInstance()
                : new ScalarBackend();
        for (String variant : new String[] {"tiny", "small", "medium"}) {
            for (String stage : new String[] {"det", "rec"}) {
                Path path = root.resolve("ppocrv6-" + variant).resolve(stage + ".onnx");
                try (LwmModel model = ModelLoader.load(path)) {
                    int[] widths = stage.equals("rec") ? new int[] {17,192,320,480,640,960} : new int[] {32,320,960};
                    for (int width : widths) {
                        TensorShape input = stage.equals("rec") ? new TensorShape(1,3,48,width) : new TensorShape(1,3,width,width);
                        List<TensorShape> shapes = ShapeResolver.resolve(model, Collections.singletonList(input));
                        TensorShape output = shapes.get(model.getGraphOutputs().get(0));
                        if (stage.equals("rec") && output.get(output.getRank() - 1) != (variant.equals("tiny") ? 6906 : 18710)) throw new IllegalStateException("REC classes mismatch");
                        System.out.println("{\"onnx\":\"" + variant + "/" + stage + "\",\"width\":" + width +
                                ",\"nodes\":" + model.getNodes().size() + ",\"output_shape\":" + output + "}");
                    }
                }
            }
        }
        try (LwmModel cls = ModelLoader.load(root.resolve("ppocrv6-tiny/cls.onnx"))) { ShapeResolver.resolveStatic(cls); }
        if (args.length >= 2) {
            Path fixtures = Paths.get(args[1]);
            List<String> cases = Files.readAllLines(fixtures.resolve("cases.tsv"), StandardCharsets.UTF_8);
            if (cases.size() != 16) throw new IllegalStateException("incomplete ONNX fixtures");
            for (String row : cases) compare(root, fixtures, row.split("\t"), backend);
            for (String variant : new String[] {"small", "medium"}) {
                Path directory = root.resolve("ppocrv6-" + variant);
                PaddleOcrOptions options = PaddleOcrOptions.builder().setDetectionMaximumSideLength(960).build();
                try (PaddleOcr ocr = PaddleOcr.load(directory.resolve("det.onnx"), root.resolve("ppocrv6-tiny/cls.onnx"),
                        directory.resolve("rec.onnx"), root.resolve("ppocrv6-shared/PP-OCRv6_small_rec_dict.txt"), options, backend)) {
                    OcrResult result = PaddleOcrImageIo.recognize(ocr, fixtures.resolve("sample.jpg"));
                    if (result.getLines().size() != 16) throw new IllegalStateException("sample must contain 16 detected lines");
                    if (record) exportInputs(fixtures, variant, result);
                    String expected = new String(Files.readAllBytes(fixtures.resolve(variant + "-pipeline-full-ocr.txt")), StandardCharsets.UTF_8).replace("\r\n", "\n").trim();
                    String cExpected = new String(Files.readAllBytes(fixtures.resolve(variant + "-full-ocr.txt")), StandardCharsets.UTF_8).replace("\r\n", "\n").trim();
                    if (!result.getText().equals(expected)) {
                        throw new IllegalStateException(variant + " full OCR text mismatch:\n" + result.getText());
                    }
                    System.out.println("{\"onnx_full_ocr\":\"" + variant + "\",\"lines\":" + result.getLines().size() +
                            ",\"pipeline_golden_parity\":true,\"c_reference_text_parity\":" + result.getText().equals(cExpected) +
                            ",\"stage_inputs_exported\":" + record + "}");
                }
            }
        }
    }
    private static void exportInputs(Path fixtures, String variant, OcrResult result) throws Exception {
        BgrImage image = ImageIoLoader.load(fixtures.resolve("sample.jpg"));
        List<String> rows = new ArrayList<String>(); int index = 0;
        for (OcrLineResult line : result.getLines()) {
            BgrImage crop = PerspectiveCrop.crop(image, line.getBox());
            String stem = variant + "-line-" + index++;
            writeFloats(fixtures.resolve(stem + "-cls.f32"), ClsPreprocess.resizeNormalize(crop).getChw());
            if (line.isRotated()) BgrTransforms.rotate180InPlace(crop);
            int width = RecWidthPolicy.chooseTargetWidth(crop, 960);
            writeFloats(fixtures.resolve(stem + "-rec.f32"), RecPreprocess.resizeNormalize(crop, width).getChw());
            rows.add(stem + "\t" + width + "\t" + Base64.getEncoder().encodeToString(line.getText().getBytes(StandardCharsets.UTF_8)) +
                    "\t" + line.getRecognitionScore() + "\t" + line.getClassification().getLabel() +
                    "\t" + line.getClassification().getScore() + "\t" + line.isRotated());
        }
        Files.write(fixtures.resolve(variant + "-stage-inputs.tsv"), rows, StandardCharsets.UTF_8);
        Files.write(fixtures.resolve(variant + "-java-full-ocr.txt"), result.getText().getBytes(StandardCharsets.UTF_8));
    }
    private static void writeFloats(Path path, float[] values) throws Exception {
        ByteBuffer b = ByteBuffer.allocate(values.length * 4).order(ByteOrder.LITTLE_ENDIAN);
        for (float value : values) b.putFloat(value); Files.write(path, b.array());
    }
    private static void compare(Path root, Path fixtures, String[] row, KernelBackend backend) throws Exception {
        if (row.length != 6) throw new IllegalStateException("invalid fixture descriptor");
        int height = Integer.parseInt(row[1]), width = Integer.parseInt(row[2]);
        int[] expectedShape = Arrays.stream(row[5].split(",")).mapToInt(Integer::parseInt).toArray();
        try (LwmModel model = ModelLoader.load(root.resolve(row[0]));
             InferenceSession session = new InferenceSession(model, Collections.singletonList(new TensorShape(1,3,height,width)), backend)) {
            int output = model.getGraphOutputs().get(0);
            if (!session.execution().shapes().get(output).equalsDimensions(expectedShape)) throw new IllegalStateException("shape mismatch: " + row[0]);
            float[] input = floats(fixtures.resolve(row[3])), expected = floats(fixtures.resolve(row[4]));
            float[] actual = new float[expected.length]; session.run(input, actual);
            double max = 0, mean = 0;
            for (int i = 0; i < actual.length; i++) {
                double error = Math.abs((double) actual[i] - expected[i]);
                if (!Double.isFinite(error)) throw new IllegalStateException("non-finite output");
                max = Math.max(max, error); mean += error;
            }
            mean /= actual.length;
            if (max > 1e-3 || mean > 1e-4) throw new IllegalStateException(row[0] + " max=" + max + " mean=" + mean);
            int classes = expectedShape[expectedShape.length - 1];
            if (row[0].endsWith("rec.onnx") || row[0].endsWith("cls.onnx")) {
                for (int offset = 0; offset < expected.length; offset += classes) {
                    int a = 0, b = 0;
                    for (int i = 1; i < classes; i++) { if (actual[offset+i] > actual[offset+a]) a = i; if (expected[offset+i] > expected[offset+b]) b = i; }
                    if (a != b) throw new IllegalStateException("argmax parity failed: " + row[0]);
                }
            }
            System.out.printf(Locale.ROOT, "{\"onnx_parity\":\"%s\",\"width\":%d,\"max_abs_error\":%.9g,\"mean_abs_error\":%.9g}%n", row[0], width, max, mean);
        }
    }
    private static float[] floats(Path path) throws Exception {
        byte[] data = Files.readAllBytes(path);
        if (data.length % 4 != 0) throw new IllegalStateException("invalid FP32 fixture");
        ByteBuffer bytes = ByteBuffer.wrap(data).order(ByteOrder.LITTLE_ENDIAN); float[] result = new float[data.length / 4];
        for (int i = 0; i < result.length; i++) result[i] = bytes.getFloat(); return result;
    }
}
