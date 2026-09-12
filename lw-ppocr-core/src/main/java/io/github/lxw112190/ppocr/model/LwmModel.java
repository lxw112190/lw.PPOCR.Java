package io.github.lxw112190.ppocr.model;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.Collections;
import java.util.List;

/** Immutable validated LWM model metadata and read-only model storage. */
public final class LwmModel implements AutoCloseable {
    private final ByteBuffer bytes;
    private final LwmHeader header;
    private final List<Integer> graphInputs;
    private final List<Integer> graphOutputs;
    private final List<TensorInfo> tensors;
    private final List<NodeInfo> nodes;
    private boolean closed;

    LwmModel(ByteBuffer bytes, LwmHeader header, List<Integer> graphInputs,
             List<Integer> graphOutputs, List<TensorInfo> tensors, List<NodeInfo> nodes) {
        this.bytes = bytes.asReadOnlyBuffer().order(ByteOrder.LITTLE_ENDIAN);
        this.header = header;
        this.graphInputs = Collections.unmodifiableList(graphInputs);
        this.graphOutputs = Collections.unmodifiableList(graphOutputs);
        this.tensors = Collections.unmodifiableList(tensors);
        this.nodes = Collections.unmodifiableList(nodes);
    }

    public LwmHeader getHeader() { ensureOpen(); return header; }
    public List<Integer> getGraphInputs() { ensureOpen(); return graphInputs; }
    public List<Integer> getGraphOutputs() { ensureOpen(); return graphOutputs; }
    public List<TensorInfo> getTensors() { ensureOpen(); return tensors; }
    public List<NodeInfo> getNodes() { ensureOpen(); return nodes; }

    /** Returns a read-only little-endian view of a validated constant payload. */
    public ByteBuffer constantData(int tensorIndex) {
        ensureOpen();
        TensorInfo tensor = tensors.get(tensorIndex);
        if (!tensor.isConstant()) {
            throw new IllegalArgumentException("tensor is not constant: " + tensorIndex);
        }
        if (tensor.getDataSize() > Integer.MAX_VALUE) {
            throw new OcrException(OcrErrorCode.RESOURCE_LIMIT, "constant tensor is too large");
        }
        ByteBuffer view = bytes.duplicate().order(ByteOrder.LITTLE_ENDIAN);
        view.position((int) tensor.getDataOffset());
        view.limit((int) (tensor.getDataOffset() + tensor.getDataSize()));
        return view.slice().asReadOnlyBuffer().order(ByteOrder.LITTLE_ENDIAN);
    }

    @Override
    public void close() {
        closed = true;
    }

    private void ensureOpen() {
        if (closed) {
            throw new OcrException(OcrErrorCode.INVALID_ARGUMENT, "LWM model is closed");
        }
    }
}
