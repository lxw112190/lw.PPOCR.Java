package io.github.lxw112190.ppocr.model;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.CharacterCodingException;
import java.nio.charset.CodingErrorAction;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/** Bounded protobuf decoder for the reviewed PP-OCR ONNX subset. */
final class OnnxProto {
    static final int MAX_BYTES = 256 * 1024 * 1024, MAX_NODES = 4096, MAX_VALUES = 16384, MAX_RANK = 8;
    final Map<String, Tensor> constants = new LinkedHashMap<String, Tensor>();
    final List<Node> nodes = new ArrayList<Node>();
    final List<Value> inputs = new ArrayList<Value>(), outputs = new ArrayList<Value>();
    int opset;
    private long names, constantBytes;
    static OcrException bad(String message) { return new OcrException(OcrErrorCode.INVALID_MODEL, "ONNX: " + message); }
    static void require(boolean condition, String message) { if (!condition) throw bad(message); }
    static void budget(boolean condition, String message) { if (!condition) throw new OcrException(OcrErrorCode.RESOURCE_LIMIT, "ONNX: " + message); }

    static OnnxProto parse(byte[] bytes) {
        OnnxProto m = new OnnxProto(); Reader r = new Reader(bytes, 0, bytes.length);
        boolean graph = false; long ir = 0;
        while (r.hasNext()) {
            int tag = r.tag();
            switch (tag >>> 3) {
                case 1: r.wire(tag, 0); ir = r.varint(); break;
                case 7: r.wire(tag, 2); require(!graph, "duplicate graph"); graph = true; m.graph(r.child()); break;
                case 8:
                    r.wire(tag, 2); Reader o = r.child(); String domain = ""; long version = 0;
                    while (o.hasNext()) { int t = o.tag(); if (t == 10) domain = m.name(o); else if (t == 16) version = o.varint(); else o.skip(t); }
                    require(domain.isEmpty() || domain.equals("ai.onnx"), "custom opset domain: " + domain);
                    require(m.opset == 0, "duplicate standard opset");
                    require(version >= 7 && version <= 14, "supported standard opsets are 7 through 14"); m.opset = (int) version; break;
                case 25: throw bad("local functions are unsupported");
                case 20: throw bad("training graphs are unsupported");
                default: r.skip(tag);
            }
        }
        require(graph && ir >= 3 && ir <= 10 && m.opset != 0, "missing graph / supported IR / opset");
        require(m.outputs.size() == 1, "one graph output is required"); return m;
    }
    private void graph(Reader r) {
        while (r.hasNext()) {
            int tag = r.tag();
            switch (tag >>> 3) {
                case 1: r.wire(tag, 2); budget(nodes.size() < MAX_NODES, "node budget"); nodes.add(node(r.child())); break;
                case 5:
                    r.wire(tag, 2); Tensor t = tensor(r.child());
                    require(!t.name.isEmpty() && !constants.containsKey(t.name), "duplicate / unnamed initializer");
                    budget(constants.size() < MAX_VALUES, "initializer budget"); constants.put(t.name, t); break;
                case 11: r.wire(tag, 2); budget(inputs.size() < MAX_VALUES, "input budget"); inputs.add(value(r.child())); break;
                case 12: r.wire(tag, 2); require(outputs.isEmpty(), "multiple outputs"); outputs.add(value(r.child())); break;
                case 15: throw bad("sparse initializers are unsupported");
                default: r.skip(tag);
            }
        }
    }
    private Node node(Reader r) {
        Node n = new Node();
        while (r.hasNext()) {
            int tag = r.tag();
            switch (tag >>> 3) {
                case 1: r.wire(tag, 2); budget(n.inputs.size() < 8, "node input budget"); n.inputs.add(name(r)); break;
                case 2: r.wire(tag, 2); require(n.output == null, "multiple node outputs"); n.output = name(r); break;
                case 3: r.wire(tag, 2); name(r); break;
                case 4: r.wire(tag, 2); require(n.op == null, "duplicate operator"); n.op = name(r); break;
                case 5:
                    r.wire(tag, 2); budget(n.attrs.size() < 16, "attribute budget"); Attribute a = attribute(r.child());
                    require(!a.name.isEmpty() && !n.attrs.containsKey(a.name), "duplicate / unnamed attribute"); n.attrs.put(a.name, a); break;
                case 7: r.wire(tag, 2); String domain = name(r); require(domain.isEmpty() || domain.equals("ai.onnx"), "custom node domain: " + domain); break;
                default: r.skip(tag);
            }
        }
        require(n.op != null && n.output != null && !n.output.isEmpty(), "missing node output / operator"); return n;
    }
    private Attribute attribute(Reader r) {
        Attribute a = new Attribute(); boolean payload = false;
        while (r.hasNext()) {
            int tag = r.tag(), field = tag >>> 3;
            if (field == 1) { r.wire(tag, 2); a.name = name(r); }
            else if (field == 20) { r.wire(tag, 0); long type = r.varint(); require(type > 0 && type <= 15, "invalid attribute type"); a.type = (int) type; }
            else if (field >= 2 && field <= 5) {
                require(!payload, "duplicate attribute payload"); payload = true;
                if (field == 2) { r.wire(tag, 5); a.f = r.f32(); a.payloadType = 1; }
                if (field == 3) { r.wire(tag, 0); a.i = r.varint(); a.payloadType = 2; }
                if (field == 4) { r.wire(tag, 2); a.s = name(r); a.payloadType = 3; }
                if (field == 5) { r.wire(tag, 2); a.tensor = tensor(r.child()); a.payloadType = 4; }
            } else if (field == 8) {
                require(!payload || a.payloadType == 7, "mixed attribute payload"); payload = true; a.payloadType = 7;
                require((tag & 7) == 0 || (tag & 7) == 2, "invalid integer-list wire"); Reader p = (tag & 7) == 2 ? r.child() : r;
                while (p.hasNext()) { budget(a.ints.size() < MAX_RANK, "attribute axis budget"); a.ints.add(p.varint()); if (p == r) break; }
            } else if (field == 6 || field == 7 || field == 9 || field == 10 || field == 11 || field == 22) throw bad("unsupported attribute payload");
            else r.skip(tag);
        }
        require(payload && a.type == a.payloadType, "attribute type / payload mismatch: " + a.name); return a;
    }
    private Tensor tensor(Reader r) {
        Tensor t = new Tensor(); List<Long> dims = new ArrayList<Long>(), integers = new ArrayList<Long>();
        List<Integer> floats = new ArrayList<Integer>(); boolean raw = false; int integerField = 0;
        while (r.hasNext()) {
            int tag = r.tag(), field = tag >>> 3;
            if (field == 1) {
                require((tag & 7) == 0 || (tag & 7) == 2, "invalid tensor dimensions wire"); Reader p = (tag & 7) == 2 ? r.child() : r;
                while (p.hasNext()) { budget(dims.size() < MAX_RANK, "tensor rank budget"); dims.add(p.varint()); if (p == r) break; }
            } else if (field == 2) {
                r.wire(tag, 0); require(t.type == 0, "duplicate tensor dtype"); long dtype = r.varint();
                require(dtype == 1 || dtype == 6 || dtype == 7, "unsupported tensor dtype"); t.type = (int) dtype;
            }
            else if (field == 8) { r.wire(tag, 2); t.name = name(r); }
            else if (field == 9) { r.wire(tag, 2); require(!raw, "duplicate raw_data"); raw = true; t.data = r.bytes(); }
            else if (field == 4) {
                require((tag & 7) == 2 || (tag & 7) == 5, "invalid float_data wire"); Reader p = (tag & 7) == 2 ? r.child() : r;
                while (p.hasNext()) { budget(floats.size() < (1 << 20), "typed float budget (use raw_data for larger tensors)"); floats.add(Float.floatToRawIntBits(p.f32())); if (p == r) break; }
            } else if (field == 5 || field == 7) {
                require(integerField == 0 || integerField == field, "mixed typed integer storage");
                integerField = field;
                require((tag & 7) == 2 || (tag & 7) == 0, "invalid integer_data wire"); Reader p = (tag & 7) == 2 ? r.child() : r;
                while (p.hasNext()) { budget(integers.size() < (1 << 20), "typed integer budget"); integers.add(p.varint()); if (p == r) break; }
            } else if (field == 13) throw bad("external tensor data is unsupported");
            else if (field == 14) { r.wire(tag, 0); require(r.varint() == 0, "external tensor data is unsupported"); }
            else if (field == 6 || field == 10 || field == 11 || field == 3) throw bad("unsupported tensor storage");
            else r.skip(tag);
        }
        require(t.type == 1 || t.type == 6 || t.type == 7, "only FLOAT / INT32 / INT64 tensors are supported");
        require(integerField == 0 || (t.type == 6 && integerField == 5) || (t.type == 7 && integerField == 7),
                "typed integer field does not match tensor dtype");
        t.shape = new int[dims.size()]; long elements = 1;
        for (int i = 0; i < dims.size(); i++) {
            long d = dims.get(i); require(d >= 0 && d <= Integer.MAX_VALUE, "invalid initializer dimension");
            budget(d == 0 || elements <= MAX_BYTES / d, "initializer size overflow"); t.shape[i] = (int) d; elements *= d;
        }
        int size = t.type == 7 ? 8 : 4; budget(elements <= MAX_BYTES / size, "initializer byte budget"); int expected = (int) elements * size;
        require(!raw || (floats.isEmpty() && integers.isEmpty()), "raw and typed data conflict");
        if (!raw) {
            require(t.type == 1 ? integers.isEmpty() && floats.size() == elements : floats.isEmpty() && integers.size() == elements, "typed initializer element count mismatch");
            t.data = new byte[expected]; ByteBuffer b = ByteBuffer.wrap(t.data).order(ByteOrder.LITTLE_ENDIAN);
            if (t.type == 1) for (int bits : floats) b.putInt(bits);
            else for (long v : integers) { if (size == 8) b.putLong(v); else b.putInt((int) v); }
        }
        require(t.data.length == expected, "initializer byte count mismatch"); constantBytes += expected; budget(constantBytes <= MAX_BYTES, "total constant budget"); return t;
    }
    private Value value(Reader r) {
        Value v = new Value();
        while (r.hasNext()) {
            int tag = r.tag();
            if (tag == 10) v.name = name(r);
            else if (tag == 18) {
                Reader type = r.child(); boolean tensor = false;
                while (type.hasNext()) {
                    require(type.tag() == 10 && !tensor, "only tensor value types are supported"); tensor = true; Reader t = type.child();
                    while (t.hasNext()) {
                        int tt = t.tag();
                        if (tt == 8) {
                            require(v.type == 0, "duplicate value dtype"); long dtype = t.varint();
                            require(dtype == 1 || dtype == 6 || dtype == 7, "unsupported value dtype"); v.type = (int) dtype;
                        }
                        else if (tt == 18) {
                            Reader shape = t.child(); List<Integer> dims = new ArrayList<Integer>();
                            while (shape.hasNext()) {
                                require(shape.tag() == 10, "invalid shape field"); Reader dim = shape.child(); int dvalue = -1; boolean seen = false;
                                while (dim.hasNext()) {
                                    int dt = dim.tag();
                                    if (dt == 8) { require(!seen, "duplicate dimension"); seen = true; long d = dim.varint(); require(d > 0 && d <= Integer.MAX_VALUE, "invalid dimension"); dvalue = (int) d; }
                                    else if (dt == 18) { require(!seen, "duplicate dimension"); seen = true; name(dim); }
                                    else dim.skip(dt);
                                }
                                budget(dims.size() < MAX_RANK, "value rank budget"); dims.add(dvalue);
                            }
                            v.shape = new int[dims.size()]; for (int i = 0; i < dims.size(); i++) v.shape[i] = dims.get(i);
                        } else t.skip(tt);
                    }
                }
            } else r.skip(tag);
        }
        require(v.name != null && !v.name.isEmpty() && v.shape != null && v.type != 0, "incomplete graph value"); return v;
    }
    private String name(Reader r) {
        Reader text = r.child(); int length = text.end - text.pos;
        names += length; budget(names <= 8 * 1024 * 1024 && length <= 65536, "name budget");
        try {
            String s = StandardCharsets.UTF_8.newDecoder().onMalformedInput(CodingErrorAction.REPORT).decode(ByteBuffer.wrap(text.data, text.pos, length)).toString();
            require(s.indexOf('\0') < 0, "embedded NUL in name"); return s;
        } catch (CharacterCodingException e) { throw bad("invalid UTF-8"); }
    }
    static final class Node { String op, output; final List<String> inputs = new ArrayList<String>(); final Map<String, Attribute> attrs = new LinkedHashMap<String, Attribute>(); }
    static final class Tensor { String name = ""; int type; int[] shape; byte[] data; }
    static final class Value { String name; int type; int[] shape; }
    static final class Attribute { String name = "", s; int type, payloadType; long i; float f; Tensor tensor; final List<Long> ints = new ArrayList<Long>(); }
    static final class Reader {
        final byte[] data; final int end; int pos;
        Reader(byte[] data, int start, int end) { this.data = data; this.pos = start; this.end = end; }
        boolean hasNext() { return pos < end; }
        long varint() {
            long result = 0;
            for (int i = 0; i < 10; i++) {
                require(pos < end, "truncated varint"); int b = data[pos++] & 255; require(i != 9 || (b & 254) == 0, "varint overflow");
                result |= (long) (b & 127) << (i * 7); if ((b & 128) == 0) return result;
            }
            throw bad("invalid varint");
        }
        int tag() { long t = varint(); require(t > 0 && t <= Integer.MAX_VALUE && (t >>> 3) != 0, "invalid protobuf tag"); return (int) t; }
        void wire(int tag, int expected) { require((tag & 7) == expected, "wrong protobuf wire type"); }
        Reader child() { long size = varint(); require(size >= 0 && size <= end - pos, "truncated length-delimited field"); int start = pos; pos += (int) size; return new Reader(data, start, pos); }
        byte[] bytes() { Reader c = child(); byte[] b = new byte[c.end - c.pos]; System.arraycopy(data, c.pos, b, 0, b.length); return b; }
        float f32() { require(end - pos >= 4, "truncated fixed32"); int b = (data[pos] & 255) | (data[pos+1] & 255) << 8 | (data[pos+2] & 255) << 16 | (data[pos+3] & 255) << 24; pos += 4; return Float.intBitsToFloat(b); }
        void skip(int tag) {
            switch (tag & 7) {
                case 0: varint(); break;
                case 1: require(end-pos >= 8, "truncated fixed64"); pos += 8; break;
                case 2: child(); break;
                case 5: f32(); break;
                default: throw bad("unsupported protobuf wire type");
            }
        }
    }
}
