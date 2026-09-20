package io.github.lxw112190.ppocr.benchmark;

import java.lang.reflect.Field;
import java.lang.reflect.Method;
import java.util.Locale;

/** Prints the host Vector API species used to select optional benchmark matrices. */
public final class VectorCapabilitiesMain {
    private VectorCapabilitiesMain() { }

    public static void main(String[] args) throws ReflectiveOperationException {
        // Keep the benchmark module Java-8 API compatible. The optional Vector
        // module is discovered only when this Java-25 benchmark is launched.
        Class<?> floatVector = Class.forName("jdk.incubator.vector.FloatVector");
        Class<?> vectorSpecies = Class.forName("jdk.incubator.vector.VectorSpecies");
        Field preferredField = floatVector.getField("SPECIES_PREFERRED");
        Object preferredSpecies = preferredField.get(null);
        Method bitSize = vectorSpecies.getMethod("vectorBitSize");
        Method lanes = vectorSpecies.getMethod("length");
        int preferredBits = ((Number) bitSize.invoke(preferredSpecies)).intValue();
        int preferredLanes = ((Number) lanes.invoke(preferredSpecies)).intValue();
        System.out.printf(Locale.ROOT,
                "{\"benchmark\":\"vector-capabilities\",\"preferred_bits\":%d,"
                        + "\"preferred_lanes\":%d,\"run_512\":%s}%n",
                preferredBits, preferredLanes, Boolean.toString(preferredBits >= 512));
    }
}
