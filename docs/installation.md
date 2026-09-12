# lw.PPOCR.Java 安装与使用

## 环境要求

- JDK 25；
- CPU；
- 不需要 Python、Paddle Inference、ONNX Runtime、OpenCV、JNI 或本地动态库；
- 模型由 `lw.PPOCR.C` 离线转换为 LWM v0.1，Java Runtime 不解析 ONNX。

GitHub Actions 是本仓库的构建与测试权威环境，覆盖 Linux、Windows 和 macOS。

## Maven 模块

应用只使用核心运行时：

```xml
<dependency>
    <groupId>io.github.lxw112190</groupId>
    <artifactId>lw-ppocr-core</artifactId>
    <version>0.1.0-SNAPSHOT</version>
</dependency>
```

如果应用需要从 `BufferedImage`、文件或输入流读取图片，再加入可选适配器：

```xml
<dependency>
    <groupId>io.github.lxw112190</groupId>
    <artifactId>lw-ppocr-imageio</artifactId>
    <version>0.1.0-SNAPSHOT</version>
</dependency>
```

`lw-ppocr-vector` 是 JDK 25 可选加速后端，支持部分逐元素、MatMul 和 1x1
卷积，并对其余算子回退到 Scalar。编译和运行都需要加入
`--add-modules jdk.incubator.vector`；不使用该模块时，Scalar 正确性路径不受影响。

```xml
<dependency>
    <groupId>io.github.lxw112190</groupId>
    <artifactId>lw-ppocr-vector</artifactId>
    <version>0.1.0-SNAPSHOT</version>
</dependency>
```

Vector 后端按实例注入 DET、CLS 和 REC，三者可共享同一个无状态实例：

```java
KernelBackend backend = new VectorBackend();
PaddleOcrDetector detector = new PaddleOcrDetector(detModel, 320, backend);
PaddleOcrClassifier classifier = new PaddleOcrClassifier(clsModel, backend);
PaddleOcrRecognizer recognizer = new PaddleOcrRecognizer(recModel, dictionary, backend);
PaddleOcr ocr = new PaddleOcr(detector, classifier, recognizer);
```

启动应用时加入：

```text
java --add-modules jdk.incubator.vector ...
```

## 模型准备

准备同一套 LWM v0.1 模型和字典文件：

```text
models/
├─ det.lwm
├─ cls.lwm             # 可选；不使用方向分类时省略
├─ rec.lwm
└─ ppocr_keys.txt
```

模型转换和模型下载属于 `lw.PPOCR.C` 的离线职责。Java 加载器会在发布模型前校验完整文件、校验和、张量、节点、参数和图索引。

## 纯 BGR API

核心模块接收 BGR8 图像视图，不拥有调用方的像素数组：

```java
try (PaddleOcr ocr = PaddleOcr.load(
        Path.of("models/det.lwm"),
        Path.of("models/cls.lwm"),
        Path.of("models/rec.lwm"),
        Path.of("models/ppocr_keys.txt"))) {
    BgrImage image = new BgrImage(pixels, width, height, stride);
    OcrResult result = ocr.recognize(image);
}
```

不使用 CLS 时，将第二个模型路径传为 `null`：

```java
PaddleOcr ocr = PaddleOcr.load(detector, null, recognizer, dictionary);
```

模型、识别器和 OCR 管线都实现 `AutoCloseable`，应用应使用 try-with-resources 或在生命周期结束时显式关闭。

## ImageIO API

加入 `lw-ppocr-imageio` 后，可以直接处理常用 Java 图片来源：

```java
try (PaddleOcr ocr = PaddleOcr.load(detector, classifier, recognizer, dictionary)) {
    OcrResult result = PaddleOcrImageIo.recognize(ocr, Path.of("sample.png"));
}
```

ImageIO 不进入核心 Runtime；需要更广泛图片格式时，可在应用层先转换为 BGR8。

## 并发使用

单个 `PaddleOcr` 实例包含可复用的可变 Session 工作区，不应被多个线程同时调用。并发请求使用独立 Session 的 worker pool：

```java
try (OcrWorkerPool pool = new OcrWorkerPool(Arrays.asList(
        PaddleOcr.load(detector, classifier, recognizer, dictionary),
        PaddleOcr.load(detector, classifier, recognizer, dictionary)))) {
    OcrResult result = pool.recognize(image);
}
```

Worker pool 会共享不可变模型内容的语义由各 worker 管理；每个 worker 的执行工作区彼此独立。关闭 pool 会等待正在执行的请求完成。

## 当前范围

v0.1-preview 面向固定形状 FP32 的 PP-OCRv6 Tiny/Small/Medium 合同，Scalar 是稳定参考路径。当前不承诺任意 ONNX 拓扑、动态模型发现、GPU 或 Android；Vector API 后端是 JDK 25 可选加速路径，仍会对未优化算子回退 Scalar。性能数字仅用于同机研发比较，不构成发布性能承诺。
