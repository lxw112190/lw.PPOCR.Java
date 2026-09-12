package io.github.lxw112190.ppocr.runtime;

import io.github.lxw112190.ppocr.model.LwmModel;
import io.github.lxw112190.ppocr.model.OcrErrorCode;
import io.github.lxw112190.ppocr.model.OcrException;
import io.github.lxw112190.ppocr.model.TensorInfo;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/**
 * Resolves model metadata into concrete shapes for a prepared execution.
 *
 * This first implementation resolves fully static graphs and graph inputs with
 * an explicitly supplied shape. Operator-specific propagation is intentionally
 * kept for the graph preparation milestone, not hidden in the loader.
 */
public final class ShapeResolver {
    private ShapeResolver() { }

    public static List<TensorShape> resolveStatic(LwmModel model) {
        if (model == null) {
            throw new OcrException(OcrErrorCode.INVALID_ARGUMENT, "model is required");
        }
        return resolve(model, Collections.<TensorShape>emptyList());
    }

    public static List<TensorShape> resolve(LwmModel model, List<TensorShape> inputShapes) {
        if (model == null || inputShapes == null) {
            throw new OcrException(OcrErrorCode.INVALID_ARGUMENT, "model and input shapes are required");
        }
        if (inputShapes.size() != model.getGraphInputs().size() && !inputShapes.isEmpty()) {
            throw new OcrException(OcrErrorCode.INVALID_ARGUMENT, "input shape count does not match model");
        }
        List<TensorShape> resolved = new ArrayList<TensorShape>(model.getTensors().size());
        for (int i = 0; i < model.getTensors().size(); i++) {
            TensorInfo tensor = model.getTensors().get(i);
            int[] dimensions = tensor.getDimensions();
            boolean dynamic = false;
            for (int dimension : dimensions) {
                dynamic |= dimension == -1;
            }
            int inputPosition = model.getGraphInputs().indexOf(i);
            if (inputPosition >= 0 && !inputShapes.isEmpty()) {
                TensorShape input = inputShapes.get(inputPosition);
                if (input.getRank() != dimensions.length) {
                    throw invalid("input rank does not match model tensor " + i);
                }
                int[] bound = input.getDimensions();
                for (int axis = 0; axis < dimensions.length; axis++) {
                    if (dimensions[axis] != -1 && dimensions[axis] != bound[axis]) {
                        throw invalid("input dimension does not match model tensor " + i);
                    }
                }
                resolved.add(input);
            } else if (dynamic) {
                throw invalid("tensor " + i + " still has unresolved dynamic dimensions");
            } else {
                resolved.add(new TensorShape(dimensions));
            }
        }
        return Collections.unmodifiableList(resolved);
    }

    private static OcrException invalid(String message) {
        return new OcrException(OcrErrorCode.INVALID_MODEL, message);
    }
}
