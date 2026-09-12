package io.github.lxw112190.ppocr.model;

/** Runtime exception with a machine-readable failure category. */
public final class OcrException extends RuntimeException {
    private final OcrErrorCode code;

    public OcrException(OcrErrorCode code, String message) {
        super(message);
        this.code = code;
    }

    public OcrException(OcrErrorCode code, String message, Throwable cause) {
        super(message, cause);
        this.code = code;
    }

    public OcrErrorCode getCode() {
        return code;
    }
}
