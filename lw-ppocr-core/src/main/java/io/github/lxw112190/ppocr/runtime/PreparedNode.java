package io.github.lxw112190.ppocr.runtime;

import io.github.lxw112190.ppocr.model.OperatorType;
import java.nio.ByteBuffer;

/** Shape-specialized node metadata addressed directly by execution index. */
final class PreparedNode {
    final int index;
    final OperatorType operator;
    final int[] inputs;
    final int[] outputs;
    final ByteBuffer parameters;

    PreparedNode(int index, OperatorType operator, int[] inputs, int[] outputs,
                 ByteBuffer parameters) {
        this.index = index;
        this.operator = operator;
        this.inputs = inputs;
        this.outputs = outputs;
        this.parameters = parameters;
    }

    OperatorType getOperator() { return operator; }
}
