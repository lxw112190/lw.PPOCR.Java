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
        List<DetectionBox> boxes = new ArrayList<DetectionBox>(lines.size());
        for (OcrLineResult line : lines) boxes.add(line.getBox());
        List<DetectionBox> sortedBoxes = ReadingOrder.sort(boxes, order);
        boolean[] used = new boolean[lines.size()];
        List<OcrLineResult> sortedLines = new ArrayList<OcrLineResult>(lines.size());
        for (DetectionBox box : sortedBoxes) {
            for (int i = 0; i < lines.size(); i++) {
                if (!used[i] && lines.get(i).getBox() == box) {
                    used[i] = true;
                    sortedLines.add(lines.get(i));
                    break;
                }
            }
        }
        return new OcrResult(sortedLines);
    }
}
