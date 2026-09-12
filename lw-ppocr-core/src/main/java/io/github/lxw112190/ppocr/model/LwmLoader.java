package io.github.lxw112190.ppocr.model;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/** Defensive loader for the little-endian LWM v0.1 format emitted by lw.PPOCR.C. */
public final class LwmLoader {
    private static final int HEADER_SIZE = 160;
    private static final int TENSOR_SIZE = 80;
    private static final int NODE_SIZE = 72;
    private static final int CHECKSUM_OFFSET = 128;
    private static final int HEADER_FLAG_NO_MEMORY_PLAN = 1;
    private static final long NO_WORKSPACE = 0xffffffffffffffffL;

    private LwmLoader() { }

    public static LwmModel load(Path path) {
        return load(path, RuntimeLimits.defaults());
    }

    public static LwmModel load(Path path, RuntimeLimits limits) {
        if (path == null || limits == null) {
            throw new OcrException(OcrErrorCode.INVALID_ARGUMENT, "path and limits are required");
        }
        try (FileChannel channel = FileChannel.open(path, StandardOpenOption.READ)) {
            long size = channel.size();
            checkFileSize(size, limits);
            if (size > Integer.MAX_VALUE) {
                throw invalid(OcrErrorCode.RESOURCE_LIMIT, "model exceeds ByteBuffer capacity");
            }
            ByteBuffer mapped = channel.map(FileChannel.MapMode.READ_ONLY, 0, size);
            return parse(mapped, limits);
        } catch (IOException e) {
            throw new OcrException(OcrErrorCode.IO_ERROR, "unable to read model: " + path, e);
        }
    }

    public static LwmModel load(InputStream input) {
        return load(input, RuntimeLimits.defaults());
    }

    public static LwmModel load(InputStream input, RuntimeLimits limits) {
        if (input == null || limits == null) {
            throw new OcrException(OcrErrorCode.INVALID_ARGUMENT, "input and limits are required");
        }
        try {
            ByteArrayOutputStream out = new ByteArrayOutputStream();
            byte[] buffer = new byte[8192];
            long total = 0;
            int read;
            while ((read = input.read(buffer)) != -1) {
                total += read;
                checkFileSize(total, limits);
                out.write(buffer, 0, read);
            }
            ByteBuffer bytes = ByteBuffer.wrap(out.toByteArray()).asReadOnlyBuffer();
            return parse(bytes, limits);
        } catch (IOException e) {
            throw new OcrException(OcrErrorCode.IO_ERROR, "unable to read model stream", e);
        }
    }

    private static LwmModel parse(ByteBuffer source, RuntimeLimits limits) {
        ByteBuffer bytes = source.duplicate().order(ByteOrder.LITTLE_ENDIAN);
        int fileSize = bytes.capacity();
        if (fileSize < HEADER_SIZE) {
            throw invalid(OcrErrorCode.INVALID_MODEL, "model is smaller than the LWM header");
        }
        if (bytes.get(0) != 'L' || bytes.get(1) != 'W' || bytes.get(2) != 'M' || bytes.get(3) != '0') {
            throw invalid(OcrErrorCode.INVALID_MODEL, "invalid LWM magic");
        }
        int major = u16(bytes, 4);
        int minor = u16(bytes, 6);
        if (major != 0 || minor != 1) {
            throw invalid(OcrErrorCode.UNSUPPORTED_MODEL_VERSION, "unsupported LWM format version");
        }
        if (u32(bytes, 8) != HEADER_SIZE) {
            throw invalid(OcrErrorCode.INVALID_MODEL, "invalid LWM header size");
        }

        int flags = u32(bytes, 12);
        long tensorCount = u32(bytes, 16);
        long nodeCount = u32(bytes, 20);
        long inputCount = u32(bytes, 24);
        long outputCount = u32(bytes, 28);
        long inputOffset = u64(bytes, 32);
        long outputOffset = u64(bytes, 40);
        long tensorOffset = u64(bytes, 48);
        long nodeOffset = u64(bytes, 56);
        long parameterOffset = u64(bytes, 64);
        long parameterSize = u64(bytes, 72);
        long stringOffset = u64(bytes, 80);
        long stringSize = u64(bytes, 88);
        long weightOffset = u64(bytes, 96);
        long weightSize = u64(bytes, 104);
        long declaredFileSize = u64(bytes, 112);
        long workspaceSize = u64(bytes, 120);
        long checksum = u64(bytes, 128);

        if (flags != HEADER_FLAG_NO_MEMORY_PLAN || workspaceSize != 0) {
            throw invalid(OcrErrorCode.INVALID_MODEL, "unsupported LWM flags or workspace plan");
        }
        if (u64(bytes, 136) != 0 || u64(bytes, 144) != 0 || u64(bytes, 152) != 0) {
            throw invalid(OcrErrorCode.INVALID_MODEL, "non-zero LWM reserved header field");
        }
        if (declaredFileSize != fileSize) {
            throw invalid(OcrErrorCode.INVALID_MODEL, "LWM file_size does not match actual file");
        }
        if (tensorCount > limits.getMaxTensorCount() || nodeCount > limits.getMaxNodeCount()) {
            throw invalid(OcrErrorCode.RESOURCE_LIMIT, "LWM graph exceeds configured limits");
        }
        requireAligned(inputOffset, "input");
        requireAligned(outputOffset, "output");
        requireAligned(tensorOffset, "tensor");
        requireAligned(nodeOffset, "node");
        requireAligned(parameterOffset, "parameter");
        requireAligned(stringOffset, "string");
        requireAligned(weightOffset, "weight");
        requireRange(inputOffset, inputCount * 4L, fileSize, "input table");
        requireRange(outputOffset, outputCount * 4L, fileSize, "output table");
        requireRange(tensorOffset, tensorCount * TENSOR_SIZE, fileSize, "tensor table");
        requireRange(nodeOffset, nodeCount * NODE_SIZE, fileSize, "node table");
        requireRange(parameterOffset, parameterSize, fileSize, "parameter section");
        requireRange(stringOffset, stringSize, fileSize, "string section");
        requireRange(weightOffset, weightSize, fileSize, "weight section");
        if (inputOffset < HEADER_SIZE || end(inputOffset, inputCount * 4L) > outputOffset ||
                end(outputOffset, outputCount * 4L) > tensorOffset ||
                end(tensorOffset, tensorCount * TENSOR_SIZE) > nodeOffset ||
                end(nodeOffset, nodeCount * NODE_SIZE) > parameterOffset ||
                end(parameterOffset, parameterSize) > stringOffset ||
                end(stringOffset, stringSize) > weightOffset ||
                end(weightOffset, weightSize) != fileSize) {
            throw invalid(OcrErrorCode.INVALID_MODEL, "LWM sections overlap or are out of order");
        }
        if (checksum == 0 || checksum(bytes) != checksum) {
            throw invalid(OcrErrorCode.CHECKSUM_MISMATCH, "LWM content checksum mismatch");
        }

        List<TensorInfo> tensors = new ArrayList<TensorInfo>((int) tensorCount);
        for (int i = 0; i < tensorCount; i++) {
            int base = offsetToInt(tensorOffset + i * (long) TENSOR_SIZE);
            int typeCode = u32(bytes, base);
            DataType type = DataType.fromCode(typeCode);
            int rank = u32(bytes, base + 4);
            int flagsValue = u32(bytes, base + 40);
            if (rank > limits.getMaxRank() || (flagsValue & ~7) != 0 || u32(bytes, base + 44) != 0) {
                throw invalid(OcrErrorCode.INVALID_MODEL, "invalid tensor type, rank, flags, or reserved field");
            }
            int[] dimensions = new int[rank];
            long elements = 1;
            boolean dynamic = false;
            for (int j = 0; j < 8; j++) {
                int dimension = i32(bytes, base + 8 + j * 4);
                if (j < rank) {
                    if (dimension == -1) {
                        dynamic = true;
                    } else if (dimension <= 0 || elements > Long.MAX_VALUE / dimension) {
                        throw invalid(OcrErrorCode.INVALID_MODEL, "invalid or overflowing tensor dimensions");
                    } else {
                        elements *= dimension;
                    }
                    dimensions[j] = dimension;
                } else if (dimension != 0) {
                    throw invalid(OcrErrorCode.INVALID_MODEL, "unused tensor dimension is non-zero");
                }
            }
            long dataOffset = u64(bytes, base + 48);
            long dataSize = u64(bytes, base + 56);
            long workspaceOffset = u64(bytes, base + 64);
            long workspaceBytes = u64(bytes, base + 72);
            if (workspaceOffset != NO_WORKSPACE || workspaceBytes != 0) {
                throw invalid(OcrErrorCode.INVALID_MODEL, "unsupported tensor workspace plan");
            }
            if ((flagsValue & TensorInfo.CONSTANT) != 0) {
                if (dynamic || (dataOffset & 7L) != 0 || dataOffset < weightOffset ||
                        !contains(weightOffset, weightSize, dataOffset, dataSize) ||
                        elements > Long.MAX_VALUE / type.getByteSize() ||
                        dataSize != elements * type.getByteSize()) {
                    throw invalid(OcrErrorCode.INVALID_MODEL, "invalid constant tensor data range or size");
                }
            } else if (dataOffset != 0 || dataSize != 0) {
                throw invalid(OcrErrorCode.INVALID_MODEL, "non-constant tensor contains file data");
            }
            tensors.add(new TensorInfo(type, dimensions, flagsValue, dataOffset, dataSize,
                    workspaceOffset, workspaceBytes));
        }

        List<NodeInfo> nodes = new ArrayList<NodeInfo>((int) nodeCount);
        for (int i = 0; i < nodeCount; i++) {
            int base = offsetToInt(nodeOffset + i * (long) NODE_SIZE);
            OperatorType operator = OperatorType.fromCode(u16(bytes, base));
            int inputCountForNode = u16(bytes, base + 2);
            int outputCountForNode = u16(bytes, base + 4);
            if (inputCountForNode > 8 || outputCountForNode == 0 || outputCountForNode > 4 ||
                    u16(bytes, base + 6) != 0 || u32(bytes, base + 68) != 0) {
                throw invalid(OcrErrorCode.INVALID_MODEL, "invalid node arity or reserved field");
            }
            int[] inputs = new int[inputCountForNode];
            int[] outputs = new int[outputCountForNode];
            for (int j = 0; j < 8; j++) {
                int value = u32(bytes, base + 8 + j * 4);
                if (j < inputCountForNode) {
                    requireTensorIndex(value, tensorCount, "node input");
                    inputs[j] = value;
                } else if (value != 0) {
                    throw invalid(OcrErrorCode.INVALID_MODEL, "unused node input is non-zero");
                }
            }
            for (int j = 0; j < 4; j++) {
                int value = u32(bytes, base + 40 + j * 4);
                if (j < outputCountForNode) {
                    requireTensorIndex(value, tensorCount, "node output");
                    outputs[j] = value;
                } else if (value != 0) {
                    throw invalid(OcrErrorCode.INVALID_MODEL, "unused node output is non-zero");
                }
            }
            long nodeParameterOffset = u64(bytes, base + 56);
            long nodeParameterSize = u32(bytes, base + 64);
            if (nodeParameterSize != operator.expectedParameterSize() ||
                    (nodeParameterSize == 0 && nodeParameterOffset != 0) ||
                    (nodeParameterSize != 0 && ((nodeParameterOffset & 7L) != 0 ||
                            !contains(parameterOffset, parameterSize, nodeParameterOffset, nodeParameterSize)))) {
                throw invalid(OcrErrorCode.INVALID_MODEL, "invalid node parameter range or size");
            }
            if (nodeParameterSize != 0 && u16(bytes, offsetToInt(nodeParameterOffset)) != 1) {
                throw invalid(OcrErrorCode.UNSUPPORTED_MODEL_VERSION, "unsupported operator parameter version");
            }
            nodes.add(new NodeInfo(operator, inputs, outputs, nodeParameterOffset, nodeParameterSize));
        }

        List<Integer> inputs = graphIndexes(bytes, inputOffset, inputCount, tensorCount, tensors, true);
        List<Integer> outputs = graphIndexes(bytes, outputOffset, outputCount, tensorCount, tensors, false);
        LwmHeader header = new LwmHeader(major, minor, flags, tensorCount, nodeCount, inputCount, outputCount,
                inputOffset, outputOffset, tensorOffset, nodeOffset, parameterOffset, parameterSize,
                stringOffset, stringSize, weightOffset, weightSize, declaredFileSize, workspaceSize, checksum);
        return new LwmModel(bytes, header, inputs, outputs, tensors, nodes);
    }

    private static List<Integer> graphIndexes(ByteBuffer bytes, long offset, long count, long tensorCount,
                                              List<TensorInfo> tensors, boolean input) {
        List<Integer> result = new ArrayList<Integer>((int) count);
        for (int i = 0; i < count; i++) {
            int index = u32(bytes, offsetToInt(offset + i * 4L));
            requireTensorIndex(index, tensorCount, input ? "graph input" : "graph output");
            if (input ? !tensors.get(index).isInput() : !tensors.get(index).isOutput()) {
                throw invalid(OcrErrorCode.INVALID_MODEL, "graph tensor flag is missing");
            }
            result.add(index);
        }
        return Collections.unmodifiableList(result);
    }

    private static void checkFileSize(long size, RuntimeLimits limits) {
        if (size < 0 || size > limits.getMaxModelFileSize()) {
            throw invalid(OcrErrorCode.RESOURCE_LIMIT, "model exceeds configured file size limit");
        }
    }

    private static long checksum(ByteBuffer bytes) {
        long value = 0xcbf29ce484222325L;
        for (int i = 0; i < bytes.capacity(); i++) {
            int valueByte = i >= CHECKSUM_OFFSET && i < CHECKSUM_OFFSET + 8 ? 0 : bytes.get(i) & 0xff;
            value ^= valueByte;
            value *= 0x100000001b3L;
        }
        return value;
    }

    private static boolean contains(long sectionOffset, long sectionSize, long offset, long size) {
        return offset >= sectionOffset && offset - sectionOffset <= sectionSize && size <= sectionSize - (offset - sectionOffset);
    }

    private static void requireRange(long offset, long size, long fileSize, String label) {
        if (offset < 0 || size < 0 || offset > fileSize || size > fileSize - offset) {
            throw invalid(OcrErrorCode.INVALID_MODEL, label + " exceeds model bounds");
        }
    }

    private static long end(long offset, long size) {
        if (offset < 0 || size < 0 || offset > Long.MAX_VALUE - size) {
            return Long.MAX_VALUE;
        }
        return offset + size;
    }

    private static void requireAligned(long value, String label) {
        if ((value & 7L) != 0) {
            throw invalid(OcrErrorCode.INVALID_MODEL, label + " section is not 8-byte aligned");
        }
    }

    private static void requireTensorIndex(int value, long count, String label) {
        if (value < 0 || ((long) value) >= count) {
            throw invalid(OcrErrorCode.INVALID_MODEL, "invalid " + label + " tensor index");
        }
    }

    private static int offsetToInt(long value) {
        if (value < 0 || value > Integer.MAX_VALUE) {
            throw invalid(OcrErrorCode.RESOURCE_LIMIT, "model offset exceeds Java buffer capacity");
        }
        return (int) value;
    }

    private static int u16(ByteBuffer bytes, int offset) { return bytes.getShort(offset) & 0xffff; }
    private static int u32(ByteBuffer bytes, int offset) { return bytes.getInt(offset); }
    private static int i32(ByteBuffer bytes, int offset) { return bytes.getInt(offset); }
    private static long u64(ByteBuffer bytes, int offset) { return bytes.getLong(offset); }

    private static OcrException invalid(OcrErrorCode code, String message) {
        return new OcrException(code, message);
    }
}
