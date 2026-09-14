# lw.PPOCR.Java Quick Start / 快速开始

The Release ZIP already contains the runtime, PP-OCRv6 Tiny models, dictionary,
and sample image. No separate model download is required.

Release ZIP 已包含运行库、PP-OCRv6 Tiny 模型、字典和示例图片，无需单独下载模型。
The bundled model set is located at `models/ppocrv6-tiny/`.
内置模型位于 `models/ppocrv6-tiny/`。

## 1. Requirements / 环境要求

- JDK 25
- A CPU-supported Windows, Linux, or macOS system
- No Python, ONNX Runtime, OpenCV, JNI, or native dynamic library

Run all commands from the extracted Release directory.

请在 Release 解压后的根目录运行以下命令。

## 2. Create `QuickStart.java`

```java
import io.github.lxw112190.ppocr.imageio.PaddleOcrImageIo;
import io.github.lxw112190.ppocr.ppocr.OcrResult;
import io.github.lxw112190.ppocr.ppocr.PaddleOcr;
import java.nio.file.Path;
import java.nio.file.Paths;

public class QuickStart {
    public static void main(String[] args) {
        Path root = Paths.get("models", "ppocrv6-tiny");
        try (PaddleOcr ocr = PaddleOcr.load(
                root.resolve("det.lwm"),
                root.resolve("cls.lwm"),
                root.resolve("rec.lwm"),
                root.resolve("ppocr_keys.txt"))) {
            OcrResult result = PaddleOcrImageIo.recognize(
                    ocr, root.resolve("sample.jpg"));
            System.out.println("Lines: " + result.getLines().size());
            System.out.println(result.getText());
        }
    }
}
```

## 3. Run / 运行

The JDK 25 source launcher can compile and run this single file directly:

```text
java --class-path "lib/*" QuickStart.java
```

The bundled sample should report `Lines: 16` followed by the recognized text.

运行仓库内置示例后，应输出 `Lines: 16` 和识别文本。

## 4. Optional Vector API acceleration / 可选 Vector API 加速

Import `io.github.lxw112190.ppocr.vector.VectorBackend` and
`io.github.lxw112190.ppocr.ppocr.PaddleOcrOptions`, then use this load call:

```java
try (PaddleOcr ocr = PaddleOcr.load(
        root.resolve("det.lwm"),
        root.resolve("cls.lwm"),
        root.resolve("rec.lwm"),
        root.resolve("ppocr_keys.txt"),
        PaddleOcrOptions.defaults(),
        new VectorBackend())) {
    // Recognize the sample as shown above.
}
```

Launch with:

```text
java --add-modules jdk.incubator.vector --class-path "lib/*" QuickStart.java
```

The Scalar path is the compatibility baseline. The Vector backend is optional
and requires JDK 25 at compile time and runtime.

Scalar 是兼容性基线；Vector 后端为可选加速模块，编译和运行时都需要 JDK 25。

For Maven dependencies, BGR input, lifecycle, concurrency, and tuning, see
[`docs/installation.md`](docs/installation.md). Model provenance and custom
conversion are documented in [`docs/models.md`](docs/models.md).
