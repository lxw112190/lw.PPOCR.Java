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

## 5. Small / Medium ONNX (0.3.0)

The ZIP includes `release/ppocrv6-onnx-manifest.json` and an optional downloader,
not the larger ONNX binaries. From the extracted directory:

```text
python scripts/prepare-onnx-models.py --output models
```

Python 3 is only used for acquisition; Java inference has no Python or native
runtime dependency. Manual downloads with SHA-256 verification are also supported.
See [the ONNX guide](docs/onnx-models.md) for Java loading and matching dictionaries.
For the guide's example, use `Paths.get("models")` and the bundled sample at
`models/ppocrv6-tiny/sample.jpg`.

ZIP 附带固定哈希的 ONNX 清单和可选 Python 3 下载助手，不包含大型 ONNX 文件。
下载后可直接加载 Small/Medium，无需转换；Java 推理不依赖 Python 或原生库。
两档模型共用 `ppocrv6-tiny/cls.onnx` 和 `ppocrv6-shared/PP-OCRv6_small_rec_dict.txt`，
不能使用 Tiny 的 `ppocr_keys.txt` 替代。也可根据清单手动下载并校验 SHA-256。
