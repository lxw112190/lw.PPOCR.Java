package io.github.lxw112190.ppocr.benchmark;

import io.github.lxw112190.ppocr.model.LwmLoader;
import io.github.lxw112190.ppocr.model.LwmModel;
import io.github.lxw112190.ppocr.model.NodeInfo;
import io.github.lxw112190.ppocr.model.OperatorType;
import io.github.lxw112190.ppocr.model.TensorInfo;
import io.github.lxw112190.ppocr.runtime.PreparedExecution;
import io.github.lxw112190.ppocr.runtime.TensorShape;
import java.io.IOException;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.util.Collections;
import java.util.EnumMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;

/** Emits a stable CI summary of real Tiny model topology and resolved workloads. */
public final class ModelCompatibilityMain {
    private static final String ROOT = "/golden/";

    private ModelCompatibilityMain() { }

    public static void main(String[] args) throws Exception {
        StringBuilder json = new StringBuilder("{\"schema\":1,\"models\":[");
        append(json, "det", ROOT + "det/det.lwm", new TensorShape(1, 3, 320, 320));
        json.append(',');
        append(json, "cls", ROOT + "cls/cls.lwm", new TensorShape(1, 3, 80, 160));
        json.append(',');
        append(json, "rec-192", ROOT + "rec/rec.lwm", new TensorShape(1, 3, 48, 192));
        json.append(',');
        append(json, "rec-320", ROOT + "rec/rec.lwm", new TensorShape(1, 3, 48, 320));
        json.append(',');
        append(json, "rec-480", ROOT + "rec/rec.lwm", new TensorShape(1, 3, 48, 480));
        json.append(',');
        append(json, "rec-640", ROOT + "rec/rec.lwm", new TensorShape(1, 3, 48, 640));
        json.append(',');
        append(json, "rec-960", ROOT + "rec/rec.lwm", new TensorShape(1, 3, 48, 960));
        System.out.println(json.append("]}").toString());
    }

    private static void append(StringBuilder json, String name, String resource,
                               TensorShape inputShape) throws IOException {
        try (LwmModel model = load(resource)) {
            PreparedExecution execution = new PreparedExecution(model,
                    Collections.singletonList(inputShape));
            Map<OperatorType, Integer> operators = new EnumMap<OperatorType, Integer>(OperatorType.class);
            int pointwise = 0;
            int depthwise = 0;
            int general = 0;
            long pointwiseMacs = 0L;
            long depthwiseMacs = 0L;
            long generalMacs = 0L;
            List<NodeInfo> nodes = model.getNodes();
            for (int nodeIndex = 0; nodeIndex < nodes.size(); nodeIndex++) {
                NodeInfo node = nodes.get(nodeIndex);
                Integer count = operators.get(node.getOperator());
                operators.put(node.getOperator(), count == null ? 1 : count + 1);
                if (node.getOperator() != OperatorType.CONV) continue;
                int[] inputs = node.getInputs();
                int[] outputs = node.getOutputs();
                TensorShape input = execution.shapes().get(inputs[0]);
                TensorShape weights = execution.shapes().get(inputs[1]);
                TensorShape output = execution.shapes().get(outputs[0]);
                ByteBuffer parameters = model.parameterData(nodeIndex);
                int groups = parameters.getInt(4);
                int kernelHeight = parameters.getInt(8);
                int kernelWidth = parameters.getInt(12);
                long macs = Math.multiplyExact(output.getElementCount(),
                        (long) weights.get(1) * kernelHeight * kernelWidth);
                if (kernelHeight == 1 && kernelWidth == 1) {
                    pointwise++;
                    pointwiseMacs += macs;
                } else if (groups == input.get(1) && output.get(1) == input.get(1) && weights.get(1) == 1) {
                    depthwise++;
                    depthwiseMacs += macs;
                } else {
                    general++;
                    generalMacs += macs;
                }
            }
            long constantBytes = 0L;
            for (TensorInfo tensor : model.getTensors()) if (tensor.isConstant()) constantBytes += tensor.getDataSize();
            TensorShape output = execution.shapes().get(model.getGraphOutputs().get(0));
            json.append("{\"name\":\"").append(name).append("\",\"input\":")
                    .append(shapeJson(inputShape)).append(",\"output\":").append(shapeJson(output))
                    .append(",\"tensors\":").append(model.getTensors().size())
                    .append(",\"nodes\":").append(nodes.size())
                    .append(",\"constant_bytes\":").append(constantBytes)
                    .append(",\"workspace_bytes\":").append(execution.workspacePlan().getTotalBytes())
                    .append(",\"conv\":{\"pointwise\":").append(pointwise)
                    .append(",\"depthwise\":").append(depthwise)
                    .append(",\"general\":").append(general)
                    .append(",\"pointwise_macs\":").append(pointwiseMacs)
                    .append(",\"depthwise_macs\":").append(depthwiseMacs)
                    .append(",\"general_macs\":").append(generalMacs)
                    .append("},\"operators\":{");
            boolean first = true;
            for (OperatorType operator : OperatorType.values()) {
                Integer count = operators.get(operator);
                if (count == null) continue;
                if (!first) json.append(',');
                first = false;
                json.append('"').append(operator.name().toLowerCase(Locale.ROOT))
                        .append("\":").append(count);
            }
            json.append("}}");
        }
    }

    private static String shapeJson(TensorShape shape) {
        StringBuilder json = new StringBuilder("[");
        for (int axis = 0; axis < shape.getRank(); axis++) {
            if (axis != 0) json.append(',');
            json.append(shape.get(axis));
        }
        return json.append(']').toString();
    }

    private static LwmModel load(String name) throws IOException {
        try (InputStream input = ModelCompatibilityMain.class.getResourceAsStream(name)) {
            if (input == null) throw new IOException("missing compatibility resource: " + name);
            return LwmLoader.load(input);
        }
    }
}
