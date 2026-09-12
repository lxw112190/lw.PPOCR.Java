package io.github.lxw112190.ppocr.model;

public enum OperatorType {
    CONV(1), ADD(2), MUL(3), DIV(4), ERF(5), HARD_SIGMOID(6),
    BATCH_NORMALIZATION(7), REDUCE_MEAN(8), RELU(9), AVERAGE_POOL(10),
    SQUEEZE(11), TRANSPOSE(12), UNSQUEEZE(13), MAT_MUL(14), SOFTMAX(15),
    RESHAPE(16), CONCAT(17), CONV_TRANSPOSE(18), MAX_POOL(19), RESIZE(20),
    SIGMOID(21), SUB(22), SQRT(23), POW(24), SLICE(25);

    private final int code;

    OperatorType(int code) {
        this.code = code;
    }

    public int getCode() {
        return code;
    }

    public static OperatorType fromCode(long code) {
        for (OperatorType value : values()) {
            if (value.code == code) {
                return value;
            }
        }
        throw new OcrException(OcrErrorCode.UNSUPPORTED_OPERATOR,
                "unsupported LWM operator: " + code);
    }

    public int expectedParameterSize() {
        switch (this) {
            case HARD_SIGMOID: return 16;
            case BATCH_NORMALIZATION: return 24;
            case REDUCE_MEAN: return 48;
            case AVERAGE_POOL:
            case MAX_POOL:
            case CONV:
            case CONV_TRANSPOSE: return 64;
            case SQUEEZE:
            case TRANSPOSE:
            case UNSQUEEZE: return 40;
            case SOFTMAX:
            case CONCAT: return 16;
            case RESIZE: return 32;
            case SLICE: return 136;
            default: return 0;
        }
    }
}
