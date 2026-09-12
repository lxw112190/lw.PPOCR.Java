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

`lw-ppocr-vector` 是 JDK 25 可选加速后端，覆盖 Tiny 模型使用的全部 Conv、
DET 的 2 倍上采样 ConvTranspose、广播、激活、归约和 MatMul；其他通用形状
仍会回退到 Scalar。编译和运行都需要加入
`--add-modules jdk.incubator.vector`；不使用该模块时，Scalar 正确性路径不受影响。

```xml
<dependency>
    <groupId>io.github.lxw112190</groupId>
    <artifactId>lw-ppocr-vector</artifactId>
    <version>0.1.0-SNAPSHOT</version>
</dependency>
```

完整 OCR 可以直接加载同一个无状态 Vector 后端，并配置动态 DET 最大边和
REC 宽度组并行：

```java
PaddleOcrOptions options = PaddleOcrOptions.builder()
        .setDetectionMaximumSideLength(320)
        .setRecognitionParallelism(4)
        .build();

try (PaddleOcr ocr = PaddleOcr.load(
        Path.of("models/det.lwm"),
        Path.of("models/cls.lwm"),
        Path.of("models/rec.lwm"),
        Path.of("models/ppocr_keys.txt"),
        options,
        new VectorBackend())) {
    OcrResult result = PaddleOcrImageIo.recognize(ocr, Path.of("sample.jpg"));
}
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

动态 DET 默认最大边为 960。低延迟场景可通过
`setDetectionMaximumSideLength(320)` 使用与仓库 Tiny Full OCR Golden 相同的
输入策略；最大边越大，通常检测细节更充分，但推理时间和工作区也会增加。

不使用 CLS 时，将第二个模型路径传为 `null`：

```java
PaddleOcr ocr = PaddleOcr.load(detectorPath, null, recognizerPath, dictionaryPath);
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

单张图片包含多行文字时，可以显式开启 REC 宽度组并行：

```java
PaddleOcrOptions options = PaddleOcrOptions.builder()
        .setRecognitionParallelism(4)
        .build();
```

实现会先按 192/320/480/640/960 目标宽度分组：同一宽度组内顺序执行，不同
宽度组按预估工作量从大到小进入共享任务队列，空闲线程会继续领取下一组，最终仍
按原输入及阅读顺序返回。Session 和模型常量不会按文字行重复创建。默认值为 1，
低核或严格限制线程的环境无需改动。

## 性能与内存结果

仓库中的 `sample.jpg` 位于
`lw-ppocr-core/src/test/resources/golden/ocr/sample.jpg`，CI 使用它验证 16 行完整
OCR。性能摘要明确区分 Scalar、Vector 和 Vector REC×4，并报告 DET/CLS/REC
阶段耗时、模型常驻堆、GC 后存活堆、峰值堆和 GC 次数。不同 GitHub Runner
之间波动较大，应只比较同一环境、同一参数和相同提交附近的结果。schema 3
在关闭算子探针时采集计时与内存数据，再额外运行一次已预热 OCR 生成全线程算子
诊断；`summed_thread_ms_per_ocr` 是并行线程耗时之和，不能与墙钟总耗时直接相加比较。

## 当前范围

v0.1-preview 当前验证的是动态形状 FP32 PP-OCRv6 Tiny 合同，Scalar 是稳定参考
路径。当前不承诺任意 ONNX 拓扑、动态模型发现、GPU 或 Android；Vector API
后端是 JDK 25 可选加速路径，对优化范围外的通用形状回退 Scalar。性能数字仅
用于同机研发比较，不构成发布性能承诺。
