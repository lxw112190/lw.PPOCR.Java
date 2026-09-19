package io.github.lxw112190.ppocr.vector;

import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorSpecies;

/** Internal Vector API tuning knobs used by benchmarks and CI. */
final class VectorSupport {
    static final VectorSpecies<Float> F32 = chooseSpecies();

    private VectorSupport() { }

    private static VectorSpecies<Float> chooseSpecies() {
        String configured = System.getProperty("lwppocr.vectorBits", "preferred");
        if ("128".equals(configured)) return FloatVector.SPECIES_128;
        if ("256".equals(configured)) return FloatVector.SPECIES_256;
        if ("512".equals(configured)) return FloatVector.SPECIES_512;
        return FloatVector.SPECIES_PREFERRED;
    }
}
