package io.github.lxw112190.ppocr.model;

import java.util.Arrays;

public final class NodeInfo {
    private final OperatorType operator;
    private final int[] inputs;
    private final int[] outputs;
    private final long parameterOffset;
    private final long parameterSize;

    public NodeInfo(OperatorType operator, int[] inputs, int[] outputs,
                    long parameterOffset, long parameterSize) {
        this.operator = operator;
        this.inputs = inputs.clone();
        this.outputs = outputs.clone();
        this.parameterOffset = parameterOffset;
        this.parameterSize = parameterSize;
    }

    public OperatorType getOperator() { return operator; }
    public int[] getInputs() { return inputs.clone(); }
    public int[] getOutputs() { return outputs.clone(); }
    public long getParameterOffset() { return parameterOffset; }
    public long getParameterSize() { return parameterSize; }

    @Override
    public String toString() {
        return "NodeInfo{" + operator + " inputs=" + Arrays.toString(inputs) +
                " outputs=" + Arrays.toString(outputs) + "}";
    }
}
