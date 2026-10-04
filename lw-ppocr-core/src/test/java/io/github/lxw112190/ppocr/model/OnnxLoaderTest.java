package io.github.lxw112190.ppocr.model;

import io.github.lxw112190.ppocr.runtime.InferenceSession;
import io.github.lxw112190.ppocr.runtime.TensorShape;
import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Collections;
import org.junit.Assert;
import org.junit.Test;

/** Tiny protobuf fixtures: no model downloads, native libraries or Python in normal Maven tests. */
public final class OnnxLoaderTest {
    private static final int[] INPUT = {1,3,2,-1};

    @Test public void detectsByContentAndLeavesStreamOpen() throws Exception {
        final boolean[] closed = {false};
        ByteArrayInputStream stream = new ByteArrayInputStream(model(node("Relu", "x", "y"), null, INPUT, INPUT)) {
            @Override public void close() { closed[0] = true; }
        };
        try (LwmModel m = ModelLoader.load(stream);
             InferenceSession s = new InferenceSession(m, Collections.singletonList(new TensorShape(1,3,2,2)))) {
            float[] input = {-1,2,-3,4,-5,6,-7,8,-9,10,-11,12}, output = new float[12];
            s.run(input, output); Assert.assertArrayEquals(new float[] {0,2,0,4,0,6,0,8,0,10,0,12}, output, 0);
        }
        Assert.assertFalse(closed[0]);
        Path path = Files.createTempFile("onnx-content-", ".lwm");
        try { Files.write(path, model(node("Relu","x","y"), null, INPUT, INPUT)); try (LwmModel m = ModelLoader.load(path)) { Assert.assertEquals(1, m.getNodes().size()); } }
        finally { Files.deleteIfExists(path); }
    }
    @Test public void identitySharesInputOutputWithoutSparseIds() {
        try (LwmModel m = load(model(node("Identity","x","y"), null, INPUT, INPUT));
             InferenceSession s = new InferenceSession(m, Collections.singletonList(new TensorShape(1,3,2,2)))) {
            float[] input = sequence(12), output = new float[12]; s.run(input, output); Assert.assertArrayEquals(input, output, 0);
        }
    }
    @Test public void scalarBroadcastSupportsRawAndPackedWeights() {
        for (boolean raw : new boolean[] {false,true}) {
            byte[] weight = tensor("b",1,new int[0], floats(-2),raw);
            try (LwmModel m = load(model(node("Add",new String[] {"x","b"},"y"),weight,INPUT,INPUT));
                 InferenceSession s = new InferenceSession(m, Collections.singletonList(new TensorShape(1,3,2,2)))) {
                float[] input = sequence(12), output = new float[12]; s.run(input, output);
                for (int i = 0; i < output.length; i++) Assert.assertEquals(i - 2, output[i], 0);
            }
        }
    }
    @Test public void foldsShapeSliceConcatIntoDynamicReshape() {
        byte[] controls = cat(tensor("start",7,new int[] {1},longs(3),true),tensor("end",7,new int[] {1},longs(4),true),
                tensor("prefix",7,new int[] {2},longs(1,6),true));
        byte[] graph = cat(node("Shape","x","shape"),node("Slice",new String[] {"shape","start","end"},"width"),
                node("Concat",new String[] {"prefix","width"},"target",integer("axis",0)),node("Reshape",new String[] {"x","target"},"y"));
        try (LwmModel m = load(model(graph,controls,INPUT,new int[] {1,6,-1}))) {
            Assert.assertEquals(1,m.getNodes().size()); Assert.assertEquals(OperatorType.RESHAPE,m.getNodes().get(0).getOperator());
            for (int width : new int[] {17,192}) try (InferenceSession s = new InferenceSession(m,Collections.singletonList(new TensorShape(1,3,2,width)))) {
                float[] input = sequence(6*width), output = new float[input.length]; s.run(input,output); Assert.assertArrayEquals(input,output,0);
            }
        }
    }
    @Test public void sliceStopsAtEndNotAtInputTail() {
        byte[] controls = cat(tensor("start",7,new int[] {1},longs(1),true),tensor("end",7,new int[] {1},longs(3),true),
                tensor("axis",7,new int[] {1},longs(-1),true));
        try (LwmModel m = load(model(node("Slice",new String[] {"x","start","end","axis"},"y"),controls,new int[] {1,3,2,6},new int[] {1,3,2,2}));
             InferenceSession s = new InferenceSession(m)) {
            float[] output = new float[12]; s.run(sequence(36),output);
            Assert.assertArrayEquals(new float[] {1,2,7,8,13,14,19,20,25,26,31,32},output,0);
        }
    }
    @Test public void batchedMatMulBroadcastsRightMatrix() {
        byte[] weights = tensor("b",1,new int[] {2,2},floats(1,0,0,2),true);
        try (LwmModel m = load(model(node("MatMul",new String[] {"x","b"},"y"),weights,new int[] {1,3,2,2},new int[] {1,3,2,2}));
             InferenceSession s = new InferenceSession(m)) {
            float[] input = sequence(12), output = new float[12]; s.run(input,output);
            for (int i=0;i<12;i++) Assert.assertEquals(input[i]*(i%2==0?1:2),output[i],0);
        }
    }
    @Test public void rejectsTruncationAndMalformedWire() {
        byte[] valid = model(node("Relu","x","y"),null,INPUT,INPUT);
        for (int length : new int[] {0,1,3,valid.length-1}) reject(java.util.Arrays.copyOf(valid,length));
        reject(new byte[] {11}); reject(new byte[] {58,127,0});
        byte[] overflow = new byte[11]; java.util.Arrays.fill(overflow,(byte)128); reject(overflow);
    }
    @Test public void rejectsUnsupportedDomainOperatorAndAttribute() {
        reject(model(field(1,cat(text(1,"x"),text(2,"y"),text(4,"Relu"),text(7,"custom"))),null,INPUT,INPUT));
        reject(model(node("Unknown","x","y"),null,INPUT,INPUT));
        reject(model(node("Relu",new String[] {"x"},"y",integer("unused",1)),null,INPUT,INPUT));
        reject(model(node("Conv","x","y"),null,INPUT,INPUT));
        reject(model(node("Softmax",new String[0],"y"),null,INPUT,INPUT));
    }
    @Test public void rejectsExternalWeightsAndDuplicateProducers() {
        byte[] external = field(5,cat(v(2,1),text(8,"b"),field(13,cat(text(1,"location"),text(2,"../secret"))),v(14,1)));
        reject(model(node("Relu","x","y"),external,INPUT,INPUT));
        reject(model(cat(node("Relu","x","y"),node("Relu","x","y")),null,INPUT,INPUT));
        reject(model(node("Relu","missing","y"),null,INPUT,INPUT));
    }
    @Test public void rejectsInvalidInitializerAndRank() {
        reject(model(node("Add",new String[] {"x","b"},"y"),tensor("b",1,new int[] {2},floats(1),true),INPUT,INPUT));
        reject(model(node("Relu","x","y"),null,new int[] {1,3,2,2,1},new int[] {1,3,2,2,1}));
        byte[] invalidDtype = field(5,cat(v(2,0x100000001L),text(8,"b"),field(9,floats(1))));
        reject(model(node("Relu","x","y"),invalidDtype,INPUT,INPUT));
    }
    @Test public void rejectsConfiguredBudgets() {
        byte[] valid = model(node("Relu","x","y"),null,INPUT,INPUT);
        try { OnnxLoader.load(new ByteArrayInputStream(valid),new RuntimeLimits(16,100,100,8)); Assert.fail(); }
        catch (OcrException e) { Assert.assertEquals(OcrErrorCode.RESOURCE_LIMIT,e.getCode()); }
        try { OnnxLoader.load(new ByteArrayInputStream(valid),new RuntimeLimits(10000,100,100,2)); Assert.fail(); }
        catch (OcrException e) { Assert.assertEquals(OcrErrorCode.RESOURCE_LIMIT,e.getCode()); }
    }
    @Test public void rejectsNegativeStepAndInvalidReshape() {
        byte[] controls = cat(tensor("start",7,new int[] {1},longs(1),true),tensor("end",7,new int[] {1},longs(3),true),
                tensor("axis",7,new int[] {1},longs(3),true),tensor("step",7,new int[] {1},longs(-1),true));
        reject(model(node("Slice",new String[] {"x","start","end","axis","step"},"y"),controls,INPUT,INPUT));
        reject(model(node("Reshape",new String[] {"x","target"},"y"),tensor("target",7,new int[] {2},longs(-1,-1),true),INPUT,new int[] {-1,-1}));
    }
    @Test public void detectsExistingLwmWithoutChangingChecksums() {
        try (LwmModel m = ModelLoader.load(new ByteArrayInputStream(LwmLoaderSelfTest.minimalModel()))) { Assert.assertEquals(1,m.getTensors().size()); }
    }
    private static LwmModel load(byte[] bytes) { return OnnxLoader.load(new ByteArrayInputStream(bytes)); }
    private static void reject(byte[] bytes) {
        try (LwmModel ignored = load(bytes)) { Assert.fail("expected rejection"); }
        catch (OcrException e) { Assert.assertTrue(e.getCode()==OcrErrorCode.INVALID_MODEL || e.getCode()==OcrErrorCode.UNSUPPORTED_OPERATOR || e.getCode()==OcrErrorCode.RESOURCE_LIMIT); }
    }
    private static byte[] model(byte[] nodes,byte[] constants,int[] input,int[] output) {
        return cat(v(1,6),field(7,cat(nodes,constants==null?new byte[0]:constants,field(11,value("x",input)),field(12,value("y",output)))),field(8,v(2,14)));
    }
    private static byte[] value(String name,int[] dimensions) {
        ByteArrayOutputStream shape = new ByteArrayOutputStream();
        for(int d:dimensions) write(shape,field(1,d==-1?text(2,"dynamic"):v(1,d)));
        return cat(text(1,name),field(2,field(1,cat(v(1,1),field(2,shape.toByteArray())))));
    }
    private static byte[] node(String op,String input,String output) { return node(op,new String[] {input},output); }
    private static byte[] node(String op,String[] inputs,String output,byte[]... attrs) {
        ByteArrayOutputStream n = new ByteArrayOutputStream();
        for(String name:inputs) write(n,text(1,name)); write(n,text(2,output)); write(n,text(4,op));
        for(byte[] a:attrs) write(n,field(5,a)); return field(1,n.toByteArray());
    }
    private static byte[] integer(String name,long value) { return cat(text(1,name),v(3,value),v(20,2)); }
    private static byte[] tensor(String name,int type,int[] shape,byte[] data,boolean raw) {
        ByteArrayOutputStream dims = new ByteArrayOutputStream(); for(int d:shape) write(dims,var(d));
        return field(5,cat(field(1,dims.toByteArray()),v(2,type),text(8,name),field(raw?9:4,data)));
    }
    private static byte[] floats(float... f) { ByteBuffer b=ByteBuffer.allocate(f.length*4).order(ByteOrder.LITTLE_ENDIAN); for(float x:f)b.putFloat(x);return b.array(); }
    private static byte[] longs(long... f) { ByteBuffer b=ByteBuffer.allocate(f.length*8).order(ByteOrder.LITTLE_ENDIAN); for(long x:f)b.putLong(x);return b.array(); }
    private static float[] sequence(int n) { float[] result=new float[n];for(int i=0;i<n;i++)result[i]=i;return result; }
    private static byte[] text(int field,String s) { return field(field,s.getBytes(StandardCharsets.UTF_8)); }
    private static byte[] field(int number,byte[] value) { return cat(var(number*8+2),var(value.length),value); }
    private static byte[] v(int number,long value) { return cat(var(number*8),var(value)); }
    private static byte[] var(long value) { ByteArrayOutputStream out=new ByteArrayOutputStream();while((value & ~127L)!=0){out.write((int)value & 127 | 128);value >>>=7;}out.write((int)value);return out.toByteArray(); }
    private static byte[] cat(byte[]... arrays) { ByteArrayOutputStream out=new ByteArrayOutputStream();for(byte[] a:arrays)write(out,a);return out.toByteArray(); }
    private static void write(ByteArrayOutputStream out,byte[] value) { out.write(value,0,value.length); }
}
