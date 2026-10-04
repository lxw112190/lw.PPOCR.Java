package io.github.lxw112190.ppocr.model;

import io.github.lxw112190.ppocr.runtime.ShapeResolver;
import io.github.lxw112190.ppocr.runtime.TensorShape;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import static io.github.lxw112190.ppocr.model.OnnxProto.*;

/**
 * Pure Java importer for batch-one PP-OCR ONNX graphs (standard opsets 7..14).
 * Shape controls are folded; numeric operators reuse the validated LWM IR.
 * This is not an arbitrary ONNX executor. No native or protobuf dependency.
 */
public final class OnnxLoader {
    private OnnxLoader() { }
    public static LwmModel load(Path path) { return load(path, RuntimeLimits.defaults()); }
    public static LwmModel load(Path path, RuntimeLimits limits) {
        if (path == null || limits == null) throw argument();
        try {
            budget(Files.size(path) <= Math.min(MAX_BYTES, limits.getMaxModelFileSize()), "file budget");
            try (InputStream in = Files.newInputStream(path)) { return load(in, limits); }
        } catch (IOException e) { throw new OcrException(OcrErrorCode.IO_ERROR, "unable to read ONNX: " + path, e); }
    }
    public static LwmModel load(InputStream input) { return load(input, RuntimeLimits.defaults()); }
    /** Leaves caller-owned streams open. Normalized storage does not retain the source bytes. */
    public static LwmModel load(InputStream input, RuntimeLimits limits) {
        if (input == null || limits == null) throw argument();
        try {
            ByteArrayOutputStream out = new ByteArrayOutputStream(); byte[] buffer = new byte[8192]; int n;
            long max = Math.min(MAX_BYTES, limits.getMaxModelFileSize());
            while ((n = input.read(buffer)) >= 0) {
                if (n == 0) { int b = input.read(); if (b < 0) break; budget(out.size() < max, "file budget"); out.write(b); }
                else { budget((long) out.size() + n <= max, "file budget"); out.write(buffer, 0, n); }
            }
            return new Importer(OnnxProto.parse(out.toByteArray()), limits).load();
        } catch (IOException e) { throw new OcrException(OcrErrorCode.IO_ERROR, "unable to read ONNX stream", e); }
        catch (ArithmeticException e) { throw bad("shape / storage overflow"); }
        catch (IllegalArgumentException e) { throw new OcrException(OcrErrorCode.INVALID_MODEL, "ONNX: invalid shape", e); }
    }
    private static OcrException argument() { return new OcrException(OcrErrorCode.INVALID_ARGUMENT, "ONNX source and limits are required"); }

    private static final class Datum {
        int[] shape; long[] controls; Tensor constant;
        Datum(int[] shape, long[] controls, Tensor constant) { this.shape = shape; this.controls = controls; this.constant = constant; }
    }
    private static final class Lowered {
        OperatorType type; String output; List<String> inputs; byte[] params; int[] shape;
    }
    private static final class Importer {
        final OnnxProto proto; final RuntimeLimits limits;
        final Map<String, Datum> values = new LinkedHashMap<String, Datum>();
        final Map<String, String> aliases = new LinkedHashMap<String, String>();
        final List<Lowered> nodes = new ArrayList<Lowered>();
        final Map<String, int[]> first = new LinkedHashMap<String, int[]>();
        final Map<String, int[]> declared = new LinkedHashMap<String, int[]>();
        Value input; String output;
        Importer(OnnxProto proto, RuntimeLimits limits) { this.proto = proto; this.limits = limits; }

        LwmModel load() {
            budget(proto.nodes.size() <= limits.getMaxNodeCount(), "configured node budget");
            budget(proto.constants.size() <= limits.getMaxTensorCount(), "configured initializer budget");
            for (Value v : proto.inputs) {
                if (proto.constants.containsKey(v.name)) {
                    Tensor t = proto.constants.get(v.name);
                    require(v.type == t.type && Arrays.equals(v.shape, t.shape), "initializer input declaration mismatch");
                } else { require(input == null, "one non-initializer graph input is required"); input = v; }
            }
            require(input != null && input.type == 1 && input.shape.length == 4 &&
                    (input.shape[0] == 1 || input.shape[0] == -1) && input.shape[1] == 3,
                    "requires batch-one FP32 NCHW RGB input");
            budget(input.shape.length <= limits.getMaxRank(), "configured input rank budget");
            declared.put(input.name, input.shape.clone()); declared.get(input.name)[0] = 1;
            pass(0); pass(1);
            output = root(proto.outputs.get(0).name); Datum result = get(output);
            require(result.constant == null, "constant graph outputs are unsupported");
            Value expected = proto.outputs.get(0);
            require(expected.type == 1 && result.controls == null && expected.shape.length == result.shape.length, "numeric output type / rank mismatch");
            for (int i = 0; i < expected.shape.length; i++) {
                require(expected.shape[i] == -1 || (expected.shape[i] == result.shape[i] && declared.get(output)[i] == expected.shape[i]), "output dimension mismatch");
            }
            return serialize();
        }
        void pass(int pass) {
            values.clear(); aliases.clear(); nodes.clear();
            for (Tensor t : proto.constants.values()) {
                values.put(t.name, new Datum(t.shape, t.type == 1 ? null : integers(t), t));
            }
            int[] shape = declared.get(input.name).clone();
            for (int i = 0; i < shape.length; i++) if (shape[i] == -1) shape[i] = pass == 0 ? 320 : 640;
            require(!values.containsKey(input.name), "input initializer conflict");
            values.put(input.name, new Datum(shape, null, null));
            for (int index = 0; index < proto.nodes.size(); index++) {
                Node n = proto.nodes.get(index);
                try { lower(n, pass); }
                catch (OcrException e) { throw new OcrException(e.getCode(), e.getMessage() + " at node " + index + " (" + n.op + " -> " + n.output + ")", e); }
            }
        }
        void lower(Node n, int pass) {
            budget(values.size() + aliases.size() < Math.min(MAX_VALUES, limits.getMaxTensorCount()), "value budget");
            require(!values.containsKey(n.output) && !aliases.containsKey(n.output), "duplicate value producer");
            if (n.op.equals("Constant")) {
                arity(n, 0, 0); allowed(n, "value"); Attribute a = attr(n, "value", 4);
                require(a != null, "Constant tensor is required"); Tensor t = a.tensor;
                values.put(n.output, new Datum(t.shape, t.type == 1 ? null : integers(t), t)); return;
            }
            List<Datum> in = new ArrayList<Datum>();
            for (String name : n.inputs) {
                // Only Resize ROI may be omitted in the supported subset.
                if (name.isEmpty()) { in.add(null); } else in.add(get(name));
            }
            if (n.op.equals("Identity")) {
                arity(n, 1, 1); allowed(n, ""); require(in.get(0) != null, "Identity input");
                aliases.put(n.output, root(n.inputs.get(0))); return;
            }
            if (n.op.equals("Shape")) {
                arity(n, 1, 1); allowed(n, ""); Datum d = in.get(0); require(d != null, "Shape input");
                long[] controls = new long[d.shape.length]; for (int i = 0; i < controls.length; i++) controls[i] = d.shape[i];
                values.put(n.output, new Datum(new int[] {controls.length}, controls, null)); return;
            }
            OperatorType type = operator(n.op);
            for (int i = 0; i < in.size(); i++) require(in.get(i) != null ||
                    (type == OperatorType.RESIZE && i == 1), "omitted input unsupported");
            ByteBuffer p = parameters(n, type, in);
            if (type == OperatorType.SOFTMAX) {
                int axis = integer(n, "axis", proto.opset < 13 ? 1 : -1);
                int rank = in.get(0).shape.length;
                if (axis < 0) axis += rank;
                require(axis >= 0 && axis < rank, "Softmax axis");
                // Pre-opset-13 flattens all dimensions from axis onward.
                // Reviewed PP-OCR graphs use the last axis in every Softmax.
                require(proto.opset >= 13 || axis == rank - 1, "pre-opset-13 Softmax supports last axis only");
            }
            int count = in.size();
            if (type == OperatorType.RESHAPE || type == OperatorType.RESIZE || type == OperatorType.SLICE ||
                    type == OperatorType.SQUEEZE || type == OperatorType.UNSQUEEZE) count = 1;
            if (in.get(0).controls != null) {
                values.put(n.output, metadata(n, type, in, p)); return;
            }
            TensorShape[] shapes = new TensorShape[count];
            for (int i = 0; i < count; i++) {
                require(in.get(i) != null && in.get(i).controls == null, "metadata / omitted input reaches numeric operator");
                require(in.get(i).constant == null || in.get(i).constant.type == 1, "non-FP32 numeric input");
                shapes[i] = new TensorShape(in.get(i).shape);
            }
            int[] reshape = type == OperatorType.RESHAPE ? reshape(in.get(0), controls(in.get(1))) : new int[0];
            int[] actual = ShapeResolver.resolveNode(type, shapes, reshape, p).getDimensions();
            budget(actual.length <= Math.min(MAX_RANK, limits.getMaxRank()), "rank budget");
            if (pass == 0) { first.put(n.output, actual); declared.put(n.output, actual.clone()); }
            else {
                int[] previous = first.get(n.output), dynamic = declared.get(n.output);
                require(previous != null && previous.length == actual.length, "varying rank");
                int varying = 0;
                for (int i = 0; i < actual.length; i++) { if (previous[i] != actual[i]) { dynamic[i] = -1; varying++; } }
                require(type != OperatorType.RESHAPE || varying <= 1, "dynamic Reshape has multiple varying axes");
            }
            Datum d = new Datum(actual, null, null); values.put(n.output, d);
            Lowered node = new Lowered(); node.type = type; node.output = n.output; node.shape = declared.get(n.output);
            node.params = p.array(); node.inputs = new ArrayList<String>();
            for (int i = 0; i < count; i++) node.inputs.add(root(n.inputs.get(i)));
            nodes.add(node);
        }
        Datum metadata(Node n, OperatorType type, List<Datum> in, ByteBuffer p) {
            Datum d = in.get(0); long[] c = d.controls;
            if (type == OperatorType.CONCAT) {
                require(integer(n, "axis", 0) == 0, "metadata Concat axis");
                List<Long> result = new ArrayList<Long>();
                for (Datum v : in) { require(v.shape.length == 1, "metadata Concat rank"); for (long x : controls(v)) result.add(x); }
                budget(result.size() <= MAX_RANK, "metadata vector budget");
                long[] out = new long[result.size()]; for (int i = 0; i < out.length; i++) out[i] = result.get(i);
                return new Datum(new int[] {out.length}, out, null);
            }
            if (type == OperatorType.SLICE) {
                require(d.shape.length == 1 && p.getShort(2) == 1 && p.getInt(68) == 0, "metadata Slice must be a vector");
                int start = bound(p.getInt(4), c.length), end = bound(p.getInt(36), c.length), step = p.getInt(100);
                require(step > 0, "metadata Slice positive step"); int len = Math.max(0, (end - start + step - 1) / step);
                long[] out = new long[len]; for (int i = 0; i < len; i++) out[i] = c[start + i * step];
                return new Datum(new int[] {len}, out, null);
            }
            if (type == OperatorType.SQUEEZE || type == OperatorType.UNSQUEEZE) {
                int[] out = ShapeResolver.resolveNode(type, new TensorShape[] {new TensorShape(d.shape)}, new int[0], p).getDimensions();
                return new Datum(out, c.clone(), null);
            }
            throw bad("unsupported shape-control operator: " + n.op);
        }
        String root(String name) { while (aliases.containsKey(name)) name = aliases.get(name); return name; }
        Datum get(String name) { Datum d = values.get(root(name)); require(d != null, "missing / non-topological input: " + name); return d; }

        LwmModel serialize() {
            Map<String, Integer> ids = new LinkedHashMap<String, Integer>();
            ids.put(input.name, 0); if (!ids.containsKey(output)) ids.put(output, ids.size());
            for (Lowered n : nodes) { for (String name : n.inputs) if (!ids.containsKey(name)) ids.put(name, ids.size()); if (!ids.containsKey(n.output)) ids.put(n.output, ids.size()); }
            // Identity-only graph can have the same input and output.
            if (output.equals(input.name)) ids.put(input.name, 0);
            long parameters = 0, weights = 0;
            for (Lowered n : nodes) parameters += align(n.params.length);
            for (String name : ids.keySet()) if (get(name).constant != null) weights += align(get(name).constant.data.length);
            int tensorOffset = 176, nodeOffset = tensorOffset + ids.size() * 80, paramOffset = nodeOffset + nodes.size() * 72;
            long weightOffset = paramOffset + parameters, size = weightOffset + weights;
            budget(size <= Math.min(MAX_BYTES, limits.getMaxModelFileSize()), "normalized model budget");
            ByteBuffer b = ByteBuffer.allocate((int) size).order(ByteOrder.LITTLE_ENDIAN);
            b.put(new byte[] {'L', 'W', 'M', '0'}); b.putShort(6, (short) 1); b.putInt(8, 160); b.putInt(12, 1);
            b.putInt(16, ids.size()); b.putInt(20, nodes.size()); b.putInt(24, 1); b.putInt(28, 1);
            b.putLong(32, 160); b.putLong(40, 168); b.putLong(48, tensorOffset); b.putLong(56, nodeOffset);
            b.putLong(64, paramOffset); b.putLong(72, parameters); b.putLong(80, weightOffset); b.putLong(96, weightOffset);
            b.putLong(104, weights); b.putLong(112, size); b.putInt(160, ids.get(input.name)); b.putInt(168, ids.get(output));
            int w = (int) weightOffset;
            for (Map.Entry<String, Integer> entry : ids.entrySet()) {
                Datum d = get(entry.getKey()); int off = tensorOffset + entry.getValue() * 80;
                boolean constant = d.constant != null; require(!constant || d.constant.type == 1, "only numeric FP32 constants are emitted");
                int[] dims = constant ? d.shape : declared.get(entry.getKey()); require(dims != null, "missing numeric declaration");
                b.putInt(off, 1); b.putInt(off + 4, dims.length);
                for (int i = 0; i < dims.length; i++) b.putInt(off + 8 + i * 4, dims[i]);
                int flags = (constant ? 1 : 0) | (entry.getKey().equals(input.name) ? 2 : 0) | (entry.getKey().equals(output) ? 4 : 0);
                b.putInt(off + 40, flags); b.putLong(off + 64, -1L);
                if (constant) { b.putLong(off + 48, w); b.putLong(off + 56, d.constant.data.length); b.position(w); b.put(d.constant.data); w += align(d.constant.data.length); }
            }
            int p = paramOffset;
            for (int i = 0; i < nodes.size(); i++) {
                Lowered n = nodes.get(i); int off = nodeOffset + i * 72;
                b.putShort(off, (short) n.type.getCode()); b.putShort(off + 2, (short) n.inputs.size()); b.putShort(off + 4, (short) 1);
                for (int j = 0; j < n.inputs.size(); j++) b.putInt(off + 8 + j * 4, ids.get(n.inputs.get(j)));
                b.putInt(off + 40, ids.get(n.output));
                if (n.params.length != 0) { b.putLong(off + 56, p); b.putInt(off + 64, n.params.length); b.position(p); b.put(n.params); p += align(n.params.length); }
            }
            long hash = 0xcbf29ce484222325L; for (byte value : b.array()) { hash ^= value & 255; hash *= 0x100000001b3L; }
            b.putLong(128, hash); return LwmLoader.parse(b, limits);
        }
    }

    private static ByteBuffer parameters(Node n, OperatorType type, List<Datum> in) {
        ByteBuffer p = ByteBuffer.allocate(type.expectedParameterSize()).order(ByteOrder.LITTLE_ENDIAN);
        if (p.capacity() > 0) p.putShort(0, (short) 1);
        switch (type) {
            case CONV: case CONV_TRANSPOSE: case AVERAGE_POOL: case MAX_POOL:
                boolean conv = type == OperatorType.CONV || type == OperatorType.CONV_TRANSPOSE;
                allowed(n, conv ? "auto_pad,kernel_shape,strides,dilations,pads,group,output_padding,output_shape" : "auto_pad,kernel_shape,strides,pads,dilations,ceil_mode,count_include_pad,storage_order");
                arity(n, conv ? 2 : 1, conv ? 3 : 1);
                int[] kernel = list(n, "kernel_shape", null);
                if (kernel == null && conv) { int[] w = in.get(1).shape; require(w.length == 4, "Conv weight rank"); kernel = new int[] {w[2], w[3]}; }
                require(kernel != null && kernel.length == 2, "spatial kernel shape"); put(p, 8, kernel, 2);
                put(p, 16, list(n, "strides", new int[] {1,1}), 2);
                int padOffset = conv ? 32 : 24;
                put(p, padOffset, list(n, "pads", new int[] {0,0,0,0}), 4);
                if (conv) { p.putShort(2, (short) 2); p.putInt(4, integer(n, "group", 1)); put(p, 24, list(n, "dilations", new int[] {1,1}), 2); }
                else {
                    p.putInt(4, 2); p.putInt(40, integer(n, "ceil_mode", 0)); p.putInt(44, integer(n, "count_include_pad", 0));
                    require(Arrays.equals(list(n, "dilations", new int[] {1,1}), new int[] {1,1}) && integer(n, "storage_order", 0) == 0, "pool dilation / storage order");
                }
                require(!n.attrs.containsKey("output_shape") && (!n.attrs.containsKey("output_padding") ||
                        (type == OperatorType.CONV_TRANSPOSE && Arrays.equals(list(n, "output_padding", null), new int[] {0,0}))), "ConvTranspose output shape / padding");
                String auto = string(n, "auto_pad", "NOTSET");
                if (!auto.equals("NOTSET")) {
                    require(auto.equals("SAME_UPPER") && type != OperatorType.CONV_TRANSPOSE && p.getInt(16) == 1 && p.getInt(20) == 1, "dynamic auto_pad");
                    require(!n.attrs.containsKey("pads"), "auto_pad conflicts with pads");
                    for (int i = 0; i < 2; i++) { int total = Math.multiplyExact(kernel[i] - 1, conv ? p.getInt(24 + i * 4) : 1); require(total >= 0, "padding"); p.putInt(padOffset + i * 4, total / 2); p.putInt(padOffset + 8 + i * 4, total - total / 2); }
                }
                break;
            case HARD_SIGMOID:
                arity(n, 1, 1); allowed(n, "alpha,beta"); p.putFloat(4, floating(n, "alpha", .2f)); p.putFloat(8, floating(n, "beta", .5f)); break;
            case BATCH_NORMALIZATION:
                arity(n, 5, 5); allowed(n, "epsilon,momentum,training_mode,spatial");
                require(integer(n, "training_mode", 0) == 0 && integer(n, "spatial", 1) == 1, "only inference spatial BN");
                p.putFloat(4, floating(n, "epsilon", 1e-5f)); p.putFloat(8, floating(n, "momentum", .9f)); break;
            case REDUCE_MEAN:
                arity(n, 1, 1); allowed(n, n.op.equals("GlobalAveragePool") ? "" : "axes,keepdims,noop_with_empty_axes");
                int[] reduce = n.op.equals("GlobalAveragePool") ? new int[] {2,3} : list(n, "axes", null);
                if (reduce == null || reduce.length == 0) { reduce = new int[in.get(0).shape.length]; for (int i = 0; i < reduce.length; i++) reduce[i] = i; }
                axes(p, 12, reduce); p.putInt(4, integer(n, "keepdims", 1)); p.putInt(8, integer(n, "noop_with_empty_axes", 0));
                require(p.getInt(8) == 0, "ReduceMean noop mode unsupported"); break;
            case SQUEEZE: case UNSQUEEZE: case TRANSPOSE:
                arity(n, 1, type == OperatorType.TRANSPOSE ? 1 : 2); allowed(n, type == OperatorType.TRANSPOSE ? "perm" : "axes");
                int[] layout = list(n, type == OperatorType.TRANSPOSE ? "perm" : "axes", null);
                if (in.size() == 2) { require(layout == null, "duplicate axes input / attribute"); layout = ints(controls(in.get(1))); }
                if (layout == null) {
                    if (type == OperatorType.TRANSPOSE) { layout = new int[in.get(0).shape.length]; for (int i = 0; i < layout.length; i++) layout[i] = layout.length - 1 - i; }
                    else { require(type == OperatorType.SQUEEZE, "Unsqueeze axes required"); layout = new int[0]; }
                }
                axes(p, 4, layout); break;
            case SOFTMAX: case CONCAT:
                arity(n, 1, type == OperatorType.CONCAT ? 8 : 1); allowed(n, "axis"); p.putInt(4, integer(n, "axis", type == OperatorType.CONCAT ? 0 : -1)); break;
            case RESHAPE:
                arity(n, 2, 2); allowed(n, ""); controls(in.get(1)); break;
            case RESIZE:
                arity(n, 3, 3); allowed(n, "mode,coordinate_transformation_mode,nearest_mode");
                require((in.get(1) == null || (in.get(1).constant != null && in.get(1).constant.data.length == 0)) &&
                        in.get(2) != null && in.get(2).constant != null && in.get(2).constant.type == 1 &&
                        Arrays.equals(in.get(2).shape, new int[] {4}), "Resize constant scales / empty ROI required");
                require(string(n, "mode", "nearest").equals("nearest") && string(n, "coordinate_transformation_mode", "half_pixel").equals("asymmetric") &&
                        string(n, "nearest_mode", "round_prefer_floor").equals("floor"), "only asymmetric floor nearest Resize");
                p.putShort(2, (short) 4); ByteBuffer scales = ByteBuffer.wrap(in.get(2).constant.data).order(ByteOrder.LITTLE_ENDIAN);
                for (int i = 0; i < 4; i++) { float scale = scales.getFloat(i * 4); require(Float.isFinite(scale) && scale > 0 && (i >= 2 || scale == 1), "Resize scales"); p.putFloat(4 + i * 4, scale); } break;
            case SLICE:
                arity(n, 1, 5); allowed(n, "starts,ends,axes,steps");
                int[] starts, ends, sliceAxes, steps;
                if (in.size() == 1) { starts = list(n, "starts", null); ends = list(n, "ends", null); sliceAxes = list(n, "axes", null); steps = list(n, "steps", null); }
                else {
                    require(in.size() >= 3 && n.attrs.isEmpty(), "Slice input controls / attributes conflict");
                    starts = clamped(controls(in.get(1))); ends = clamped(controls(in.get(2)));
                    sliceAxes = in.size() > 3 ? ints(controls(in.get(3))) : null; steps = in.size() > 4 ? ints(controls(in.get(4))) : null;
                }
                require(starts != null && ends != null && starts.length > 0 && starts.length == ends.length, "Slice controls");
                int count = starts.length;
                if (sliceAxes == null) { sliceAxes = new int[count]; for (int i = 0; i < count; i++) sliceAxes[i] = i; }
                if (steps == null) { steps = new int[count]; Arrays.fill(steps, 1); }
                p.putShort(2, (short) count); put(p, 4, starts, count); put(p, 36, ends, count); put(p, 68, sliceAxes, count); put(p, 100, steps, count);
                for (int step : steps) require(step > 0, "Slice positive steps only"); break;
            case ADD: case SUB: case MUL: case DIV: case POW: case MAT_MUL: arity(n, 2, 2); allowed(n, ""); break;
            default: arity(n, 1, 1); allowed(n, "");
        }
        return p;
    }
    private static OperatorType operator(String name) {
        String[] names = {"Conv","Add","Mul","Div","Erf","HardSigmoid","BatchNormalization","ReduceMean","Relu","AveragePool","Squeeze","Transpose","Unsqueeze","MatMul","Softmax","Reshape","Concat","ConvTranspose","MaxPool","Resize","Sigmoid","Sub","Sqrt","Pow","Slice"};
        if (name.equals("GlobalAveragePool")) return OperatorType.REDUCE_MEAN;
        for (int i = 0; i < names.length; i++) if (name.equals(names[i])) return OperatorType.fromCode(i + 1);
        throw new OcrException(OcrErrorCode.UNSUPPORTED_OPERATOR, "ONNX: unsupported operator: " + name);
    }
    private static void arity(Node n, int min, int max) { require(n.inputs.size() >= min && n.inputs.size() <= max, "operator arity"); }
    private static void allowed(Node n, String names) { String joined = "," + names + ","; for (String key : n.attrs.keySet()) require(joined.contains("," + key + ","), "unsupported attribute: " + key); }
    private static Attribute attr(Node n, String name, int type) { Attribute a = n.attrs.get(name); require(a == null || a.type == type, "attribute type: " + name); return a; }
    private static int integer(Node n, String name, int fallback) { Attribute a = attr(n, name, 2); return a == null ? fallback : exact(a.i); }
    private static float floating(Node n, String name, float fallback) { Attribute a = attr(n, name, 1); float f = a == null ? fallback : a.f; require(Float.isFinite(f), "non-finite attribute"); return f; }
    private static String string(Node n, String name, String fallback) { Attribute a = attr(n, name, 3); return a == null ? fallback : a.s; }
    private static int[] list(Node n, String name, int[] fallback) { Attribute a = attr(n, name, 7); if (a == null) return fallback; int[] result = new int[a.ints.size()]; for (int i = 0; i < result.length; i++) result[i] = exact(a.ints.get(i)); return result; }
    private static int exact(long n) { require(n >= Integer.MIN_VALUE && n <= Integer.MAX_VALUE, "integer overflow"); return (int) n; }
    private static int[] ints(long[] controls) { int[] r = new int[controls.length]; for (int i = 0; i < r.length; i++) r[i] = exact(controls[i]); return r; }
    private static int[] clamped(long[] controls) { int[] r = new int[controls.length]; for (int i = 0; i < r.length; i++) r[i] = (int) Math.max(Integer.MIN_VALUE, Math.min(Integer.MAX_VALUE, controls[i])); return r; }
    private static long[] controls(Datum d) { require(d != null && d.controls != null, "constant shape controls required"); return d.controls; }
    private static long[] integers(Tensor t) {
        require(t.type == 6 || t.type == 7, "integer control tensor"); int size = t.type == 7 ? 8 : 4;
        budget(t.data.length / size <= MAX_RANK, "control vector budget");
        long[] c = new long[t.data.length / size]; ByteBuffer b = ByteBuffer.wrap(t.data).order(ByteOrder.LITTLE_ENDIAN);
        for (int i = 0; i < c.length; i++) c[i] = size == 8 ? b.getLong() : b.getInt(); return c;
    }
    private static void axes(ByteBuffer p, int offset, int[] axes) { budget(axes.length <= MAX_RANK, "axes budget"); p.putShort(2, (short) axes.length); put(p, offset, axes, axes.length); }
    private static void put(ByteBuffer p, int offset, int[] values, int count) { require(values.length == count && count <= MAX_RANK, "attribute vector length"); for (int i = 0; i < count; i++) p.putInt(offset + i * 4, values[i]); }
    private static int[] reshape(Datum d, long[] controls) {
        require(controls.length > 0 && controls.length <= MAX_RANK, "Reshape control rank"); int[] shape = ints(controls);
        for (int i = 0; i < shape.length; i++) if (shape[i] == 0) { require(i < d.shape.length, "Reshape zero axis"); shape[i] = d.shape[i]; }
        return shape;
    }
    private static int bound(int value, int size) { long n = value < 0 ? (long) value + size : value; return (int) Math.max(0, Math.min(size, n)); }
    private static int align(int n) { return (n + 7) & ~7; }
}
