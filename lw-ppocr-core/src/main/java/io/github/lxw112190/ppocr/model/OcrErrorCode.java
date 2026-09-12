package io.github.lxw112190.ppocr.model;

/** Stable categories for model/runtime failures. */
public enum OcrErrorCode {
    INVALID_ARGUMENT,
    IO_ERROR,
    INVALID_MODEL,
    UNSUPPORTED_MODEL_VERSION,
    CHECKSUM_MISMATCH,
    UNSUPPORTED_OPERATOR,
    RESOURCE_LIMIT
}
