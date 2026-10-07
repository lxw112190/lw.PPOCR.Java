package io.github.lxw112190.ppocr.vector;

import org.junit.Test;
import static org.junit.Assert.*;

public final class VectorStrideTwoLoadTest {
    @Test public void exactArrayEndOffsetsAndSpecialBitsMatchEveryOtherElement() {
        int lanes=VectorSupport.F32.length();
        int[] bits={0,0x80000000,0x7f800000,0xff800000,0x7fc01234,0x3f800000,1,0x80000001};
        for(int offset:new int[]{0,1,7}) {
            // Exactly the last selected element: an ordinary two-full-load
            // implementation would try to read one element past this array.
            float[] input=new float[offset+2*lanes-1],actual=new float[lanes];
            for(int i=0;i<input.length;i++)input[i]=Float.intBitsToFloat(bits[i%bits.length]);
            VectorStrideTwoLoad.load(input,offset).intoArray(actual,0);
            for(int i=0;i<lanes;i++)assertEquals("lane "+i,
                    Float.floatToRawIntBits(input[offset+2*i]),Float.floatToRawIntBits(actual[i]));
        }
    }

    @Test public void unusedOddSamplesCannotContaminateResults() {
        int lanes=VectorSupport.F32.length();
        float[] input=new float[2*lanes-1],actual=new float[lanes];
        for(int i=0;i<input.length;i++)input[i]=(i&1)==0?i*.25f:Float.NaN;
        VectorStrideTwoLoad.load(input,0).intoArray(actual,0);
        for(int i=0;i<lanes;i++)assertEquals(i*.5f,actual[i],0);
    }
}
