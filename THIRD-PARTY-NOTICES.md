# Third-party notices

The Java runtime JARs have no third-party runtime dependency. The repository
and release bundle include PP-OCRv6 Tiny model assets used for reproducible
Golden tests and out-of-the-box evaluation.

## PP-OCRv6 Tiny model assets

The following files are derived from the PP-OCR/PaddleOCR ecosystem and are
redistributed under the Apache License 2.0:

- `lw-ppocr-core/src/test/resources/golden/det/det.lwm`;
- `lw-ppocr-core/src/test/resources/golden/cls/cls.lwm`;
- `lw-ppocr-core/src/test/resources/golden/rec/rec.lwm`;
- `lw-ppocr-core/src/test/resources/golden/rec/ppocr_keys.txt`;
- `lw-ppocr-core/src/test/resources/golden/ocr/sample.jpg`;
- derived Golden input and output fixtures stored beside those assets.

The LWM files are converted Object-form derivatives of the corresponding model
weights. Their source repository, pinned source commit, and SHA-256 values are
recorded in the adjacent `manifest.json` files. See
`licenses/PaddleOCR-models-APACHE-2.0.txt` for the full license text.

PP-OCR and PaddleOCR names remain the property of their respective owners. No
endorsement by the upstream project is implied.

## Build and test dependency

JUnit 4.13.2 is used only while running tests. It is not included in the
runtime JARs or the release bundle.
