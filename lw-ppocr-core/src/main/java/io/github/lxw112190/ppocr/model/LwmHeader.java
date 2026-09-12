package io.github.lxw112190.ppocr.model;

public final class LwmHeader {
    private final int formatMajor;
    private final int formatMinor;
    private final int flags;
    private final long tensorCount;
    private final long nodeCount;
    private final long inputCount;
    private final long outputCount;
    private final long inputOffset;
    private final long outputOffset;
    private final long tensorOffset;
    private final long nodeOffset;
    private final long parameterOffset;
    private final long parameterSize;
    private final long stringOffset;
    private final long stringSize;
    private final long weightOffset;
    private final long weightSize;
    private final long fileSize;
    private final long workspaceSize;
    private final long checksum;

    public LwmHeader(int formatMajor, int formatMinor, int flags, long tensorCount, long nodeCount,
                     long inputCount, long outputCount, long inputOffset, long outputOffset,
                     long tensorOffset, long nodeOffset, long parameterOffset, long parameterSize,
                     long stringOffset, long stringSize, long weightOffset, long weightSize,
                     long fileSize, long workspaceSize, long checksum) {
        this.formatMajor = formatMajor;
        this.formatMinor = formatMinor;
        this.flags = flags;
        this.tensorCount = tensorCount;
        this.nodeCount = nodeCount;
        this.inputCount = inputCount;
        this.outputCount = outputCount;
        this.inputOffset = inputOffset;
        this.outputOffset = outputOffset;
        this.tensorOffset = tensorOffset;
        this.nodeOffset = nodeOffset;
        this.parameterOffset = parameterOffset;
        this.parameterSize = parameterSize;
        this.stringOffset = stringOffset;
        this.stringSize = stringSize;
        this.weightOffset = weightOffset;
        this.weightSize = weightSize;
        this.fileSize = fileSize;
        this.workspaceSize = workspaceSize;
        this.checksum = checksum;
    }

    public int getFormatMajor() { return formatMajor; }
    public int getFormatMinor() { return formatMinor; }
    public int getFlags() { return flags; }
    public long getTensorCount() { return tensorCount; }
    public long getNodeCount() { return nodeCount; }
    public long getInputCount() { return inputCount; }
    public long getOutputCount() { return outputCount; }
    public long getInputOffset() { return inputOffset; }
    public long getOutputOffset() { return outputOffset; }
    public long getTensorOffset() { return tensorOffset; }
    public long getNodeOffset() { return nodeOffset; }
    public long getParameterOffset() { return parameterOffset; }
    public long getParameterSize() { return parameterSize; }
    public long getStringOffset() { return stringOffset; }
    public long getStringSize() { return stringSize; }
    public long getWeightOffset() { return weightOffset; }
    public long getWeightSize() { return weightSize; }
    public long getFileSize() { return fileSize; }
    public long getWorkspaceSize() { return workspaceSize; }
    public long getChecksum() { return checksum; }
}
