package io.github.lxw112190.ppocr.model;

import java.util.Arrays;

public final class TensorInfo {
    public static final int CONSTANT = 1;
    public static final int INPUT = 2;
    public static final int OUTPUT = 4;

    private final DataType dataType;
    private final int[] dimensions;
    private final int flags;
    private final long dataOffset;
    private final long dataSize;
    private final long workspaceOffset;
    private final long workspaceSize;

    public TensorInfo(DataType dataType, int[] dimensions, int flags, long dataOffset,
                      long dataSize, long workspaceOffset, long workspaceSize) {
        this.dataType = dataType;
        this.dimensions = dimensions.clone();
        this.flags = flags;
        this.dataOffset = dataOffset;
        this.dataSize = dataSize;
        this.workspaceOffset = workspaceOffset;
        this.workspaceSize = workspaceSize;
    }

    public DataType getDataType() { return dataType; }
    public int getRank() { return dimensions.length; }
    public int[] getDimensions() { return dimensions.clone(); }
    public int getFlags() { return flags; }
    public boolean isConstant() { return (flags & CONSTANT) != 0; }
    public boolean isInput() { return (flags & INPUT) != 0; }
    public boolean isOutput() { return (flags & OUTPUT) != 0; }
    public long getDataOffset() { return dataOffset; }
    public long getDataSize() { return dataSize; }
    public long getWorkspaceOffset() { return workspaceOffset; }
    public long getWorkspaceSize() { return workspaceSize; }

    @Override
    public String toString() {
        return "TensorInfo{" + dataType + " " + Arrays.toString(dimensions) + " flags=" + flags + "}";
    }
}
