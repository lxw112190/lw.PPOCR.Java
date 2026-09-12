package io.github.lxw112190.ppocr.model;

public enum DataType {
    F32(1, 4),
    I32(2, 4),
    I64(3, 8),
    U8(4, 1);

    private final int code;
    private final int byteSize;

    DataType(int code, int byteSize) {
        this.code = code;
        this.byteSize = byteSize;
    }

    public int getCode() {
        return code;
    }

    public int getByteSize() {
        return byteSize;
    }

    public static DataType fromCode(long code) {
        for (DataType value : values()) {
            if (value.code == code) {
                return value;
            }
        }
        throw new OcrException(OcrErrorCode.INVALID_MODEL, "unsupported tensor data type: " + code);
    }
}
