package io.github.lxw112190.ppocr.ppocr;

import io.github.lxw112190.ppocr.model.OcrErrorCode;
import io.github.lxw112190.ppocr.model.OcrException;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.nio.CharBuffer;
import java.nio.charset.CharacterCodingException;
import java.nio.charset.CodingErrorAction;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/** Immutable UTF-8 recognition dictionary; CTC class zero is reserved for blank. */
public final class PaddleOcrDictionary implements AutoCloseable {
    private static final int MAX_BYTES = 1024 * 1024;
    private final List<String> labels;
    private boolean closed;

    private PaddleOcrDictionary(List<String> labels) {
        this.labels = Collections.unmodifiableList(new ArrayList<String>(labels));
    }

    public static PaddleOcrDictionary load(Path path) {
        if (path == null) throw invalid("dictionary path is required");
        try {
            return fromBytes(Files.readAllBytes(path));
        } catch (IOException e) {
            throw new OcrException(OcrErrorCode.IO_ERROR, "unable to read recognition dictionary", e);
        }
    }

    public static PaddleOcrDictionary load(InputStream input) {
        if (input == null) throw invalid("dictionary input is required");
        try {
            ByteArrayOutputStream output = new ByteArrayOutputStream();
            byte[] buffer = new byte[8192];
            int total = 0;
            int read;
            while ((read = input.read(buffer)) != -1) {
                if (read > MAX_BYTES - total) throw invalid("recognition dictionary is too large");
                output.write(buffer, 0, read);
                total += read;
            }
            return fromBytes(output.toByteArray());
        } catch (IOException e) {
            throw new OcrException(OcrErrorCode.IO_ERROR, "unable to read recognition dictionary", e);
        }
    }

    private static PaddleOcrDictionary fromBytes(byte[] bytes) {
        int start = bytes.length >= 3 && (bytes[0] & 0xff) == 0xef &&
                (bytes[1] & 0xff) == 0xbb && (bytes[2] & 0xff) == 0xbf ? 3 : 0;
        List<String> labels = new ArrayList<String>();
        boolean nonEmpty = false;
        int cursor = start;
        while (cursor < bytes.length) {
            int end = cursor;
            while (end < bytes.length && bytes[end] != '\n') end++;
            int labelEnd = end > cursor && bytes[end - 1] == '\r' ? end - 1 : end;
            String label = decode(bytes, cursor, labelEnd - cursor);
            labels.add(label);
            nonEmpty |= !label.isEmpty();
            cursor = end < bytes.length ? end + 1 : bytes.length;
        }
        if (labels.isEmpty() || !nonEmpty) throw invalid("recognition dictionary is empty");
        return new PaddleOcrDictionary(labels);
    }

    private static String decode(byte[] bytes, int offset, int length) {
        try {
            CharBuffer chars = StandardCharsets.UTF_8.newDecoder()
                    .onMalformedInput(CodingErrorAction.REPORT)
                    .onUnmappableCharacter(CodingErrorAction.REPORT)
                    .decode(ByteBuffer.wrap(bytes, offset, length));
            return chars.toString();
        } catch (CharacterCodingException e) {
            throw invalid("recognition dictionary contains invalid UTF-8");
        }
    }

    public int labelCount() { ensureOpen(); return labels.size(); }
    public int classCount() { ensureOpen(); return labels.size() + 2; }

    /** Maps a non-blank CTC class to its text label; the final class is a space. */
    public String labelForClass(int classIndex) {
        ensureOpen();
        if (classIndex <= 0 || classIndex >= classCount()) {
            throw new OcrException(OcrErrorCode.INVALID_ARGUMENT, "CTC class index is out of range");
        }
        return classIndex == labels.size() + 1 ? " " : labels.get(classIndex - 1);
    }

    @Override
    public void close() { closed = true; }

    private void ensureOpen() {
        if (closed) throw new OcrException(OcrErrorCode.INVALID_ARGUMENT, "recognition dictionary is closed");
    }

    private static OcrException invalid(String message) {
        return new OcrException(OcrErrorCode.INVALID_MODEL, message);
    }
}
