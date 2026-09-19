package io.github.lxw112190.ppocr.ppocr;

import io.github.lxw112190.ppocr.model.OcrErrorCode;
import io.github.lxw112190.ppocr.model.OcrException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/** Immutable page-level OCR result with explicit reading-order projection. */
public final class OcrResult {
    private final List<OcrLineResult> lines;

    public OcrResult(List<OcrLineResult> lines) {
        if (lines == null) throw new OcrException(OcrErrorCode.INVALID_ARGUMENT, "OCR lines are required");
        List<OcrLineResult> copy = new ArrayList<OcrLineResult>(lines.size());
        for (OcrLineResult line : lines) {
            if (line == null) throw new OcrException(OcrErrorCode.INVALID_ARGUMENT, "OCR lines cannot contain null");
            copy.add(line);
        }
        this.lines = Collections.unmodifiableList(copy);
    }

    /** Internal constructor for a list already validated and made immutable by ReadingOrder. */
    private OcrResult(List<OcrLineResult> sortedLines, boolean trusted) {
        this.lines = sortedLines;
    }

    public List<OcrLineResult> getLines() { return lines; }

    public String getText() {
        StringBuilder text = new StringBuilder();
        for (int i = 0; i < lines.size(); i++) {
            if (i != 0) text.append('\n');
            text.append(lines.get(i).getText());
        }
        return text.toString();
    }

    public OcrResult sorted(int order) {
        return new OcrResult(ReadingOrder.sortLines(lines, order), true);
    }
}
