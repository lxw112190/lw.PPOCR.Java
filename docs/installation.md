# lw.PPOCR.Java 安装与使用

## 环境要求

- JDK 25；
- CPU；
- 不需要 Python、Paddle Inference、ONNX Runtime、OpenCV、JNI 或本地动态库；
- 官方 Tiny 模型已包含在 Release ZIP 中；只有自定义模型才需要离线转换。

GitHub Actions 是本仓库的构建与测试权威环境，覆盖 Linux、Windows 和 macOS。

## 获取 0.1.0 构件

当前版本尚未发布到 Maven Central。可以从源码将构件安装到本机 Maven 仓库：

```text
mvn --batch-mode --no-transfer-progress clean install
```

完整 Reactor（包括可选 Vector 模块）需要 JDK 25。Tag 构建生成的 GitHub Release
会提供 `lw-ppocr-core-0.1.0.jar`、`lw-ppocr-imageio-0.1.0.jar` 和
`lw-ppocr-vector-0.1.0.jar`，以及 Tiny 模型、字典、示例图片和许可证文件。

## Maven 模块

应用只使用核心运行时：

```xml
<dependency>
    <groupId>io.github.lxw112190</groupId>
    <artifactId>lw-ppocr-core</artifactId>
    <version>0.1.0</version>
</dependency>
```

如果应用需要从 `BufferedImage`、文件或输入流读取图片，再加入可选适配器：

```xml
<dependency>
    <groupId>io.github.lxw112190</groupId>
    <artifactId>lw-ppocr-imageio</artifactId>
    <version>0.1.0</version>
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
    <version>0.1.0</version>
</dependency>
```

完整 OCR 可以直接加载同一个无状态 Vector 后端，并配置动态 DET 最大边和
CLS 行并行及 REC 宽度组并行：

Vector 后端会在连接关系、形状、标量常量和中间张量使用次数全部匹配时，将 Tiny
模型中的 `DIV -> ERF -> ADD -> MUL -> MUL` GELU 表达式融合执行；其他图结构仍按
原始节点逐个执行，不影响 Scalar 正确性路径。性能诊断将整段融合耗时归入 ERF。

```java
Path modelRoot = Paths.get("models", "ppocrv6-tiny");
PaddleOcrOptions options = PaddleOcrOptions.builder()
        .setDetectionMaximumSideLength(320)
        .setClassificationParallelism(4)
        .setRecognitionParallelism(4)
        .build();

try (PaddleOcr ocr = PaddleOcr.load(
        modelRoot.resolve("det.lwm"),
        modelRoot.resolve("cls.lwm"),
        modelRoot.resolve("rec.lwm"),
        modelRoot.resolve("ppocr_keys.txt"),
        options,
        new VectorBackend())) {
    OcrResult result = PaddleOcrImageIo.recognize(
            ocr, modelRoot.resolve("sample.jpg"));
}
```

也可以让运行时按 JVM 可见处理器统一规划 CLS/REC Worker：

```java
PaddleOcrOptions options = PaddleOcrOptions.builder()
        .setDetectionMaximumSideLength(320)
        .setParallelismMode(ParallelismPolicy.AUTO)
        .build();
```

`setParallelism(0)` 是 AUTO 的简写。AUTO 的行级 Worker 最多为 4，并会按检测行数
收缩；低核环境不会产生独立配置 `CLS=4`、`REC=4` 的超额线程计划。默认仍为
MANUAL/1，以保持 0.1 的线程与内存行为；`setParallelism(n)` 可同时设置两个手动上限。

启动应用时加入：

```text
java --add-modules jdk.incubator.vector ...
```

## 模型准备

### 使用官方模型

官方验证的 **PP-OCRv6 Tiny / FP32 / LWM v0.1** 模型已经随
`lw.PPOCR.Java` GitHub Release 一起发布。普通用户不需要自行下载 PaddleOCR
原始模型，也不需要执行 ONNX → LWM 转换。

下载地址：

<https://github.com/lxw112190/lw.PPOCR.Java/releases/latest>

解压后模型位于固定目录：

```text
models/
└── ppocrv6-tiny/
    ├── det.lwm
    ├── cls.lwm
    ├── rec.lwm
    ├── ppocr_keys.txt
    ├── sample.jpg
    └── manifest.json
```

Release 布局是公开契约，应用示例统一从
`Paths.get("models", "ppocrv6-tiny")` 解析模型。CI 会在发布前校验模型 SHA-256、
解压完整候选包，并从这个目录实际执行一次 OCR。

### 使用其他模型

`lw.PPOCR.Java` Runtime 本身不解析 ONNX。如果需要使用其他 PP-OCR 模型，需通过
[`lw.PPOCR.C`](https://github.com/lxw112190/lw.PPOCR.C) 提供的离线转换工具生成
LWM 模型：

```text
官方 PP-OCRv6 Tiny：Release → 直接使用

其他兼容模型：ONNX → lw.PPOCR.C Converter → LWM → lw.PPOCR.Java
```

模型来源、许可证、校验和及兼容策略详见[模型说明](models.md)。

## 纯 BGR API

核心模块接收 BGR8 图像视图，不拥有调用方的像素数组：

```java
Path modelRoot = Paths.get("models", "ppocrv6-tiny");
try (PaddleOcr ocr = PaddleOcr.load(
        modelRoot.resolve("det.lwm"),
        modelRoot.resolve("cls.lwm"),
        modelRoot.resolve("rec.lwm"),
        modelRoot.resolve("ppocr_keys.txt"))) {
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
    OcrResult result = PaddleOcrImageIo.recognize(
            ocr, Paths.get("models", "ppocrv6-tiny", "sample.jpg"));
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

单张图片包含多行文字时，可以分别开启 CLS 行并行和 REC 宽度组并行：

```java
PaddleOcrOptions options = PaddleOcrOptions.builder()
        .setClassificationParallelism(4)
        .setRecognitionParallelism(4)
        .build();
```

CLS 会把固定形状的文字行分配到独立 Session worker，所有 worker 共享已解码的
模型常量，只保留各自的执行和预处理工作区。REC 会先按 192/320/480/640/960
目标宽度分组：同一宽度组内顺序执行，不同
宽度组按预估工作量从大到小进入共享任务队列，空闲线程会继续领取下一组，最终仍
按原输入及阅读顺序返回。Session 和模型常量不会按文字行重复创建。两项默认值均为 1；
可按上文选择 AUTO，低核或严格限制线程的环境也可继续保持 MANUAL/1。

## 性能与内存结果

仓库中的 `sample.jpg` 位于
`lw-ppocr-core/src/test/resources/golden/ocr/sample.jpg`，CI 使用它验证 16 行完整
OCR。Release ZIP 中的同一图片位于 `models/ppocrv6-tiny/sample.jpg`。性能摘要明确
区分 Scalar、Vector 和 Vector CLS×4/REC×4，并报告 DET/CLS/REC
阶段耗时、模型常驻堆、GC 后存活堆、峰值堆和 GC 次数。不同 GitHub Runner
之间波动较大，应只比较同一环境、同一参数和相同提交附近的结果。schema 5
在关闭算子探针时采集计时与内存数据，再额外运行一次已预热 OCR 生成全线程算子
诊断。`operators` 保留整条流水线汇总，`stage_operators` 分别报告 DET、CLS、REC；
`stage_hot_nodes` 进一步列出最慢节点及其已解析张量形状。`summed_thread_ms_per_ocr`
是并行线程耗时之和，不能与墙钟总耗时直接相加比较。
schema 5 另外报告 DB scratch、crop arena、已物化 FP32 常量和预备 projection 权重的字节数，
并按 DET/CLS/REC 分项、按 REC 192/320/480/640/960 宽度桶展开 workspace，
用于区分执行 workspace、后处理缓存和模型常驻内存。

完整流水线会复用 Session 工作区、预处理数组、DB 几何缓冲和按文字行槽位保存的
透视裁剪像素缓冲，避免每次调用重复申请大数组。`PaddleOcr` 本身是单调用者对象；
并发服务应为每个工作线程准备独立实例，或使用 `OcrWorkerPool`。

## 当前范围

v0.1.0 当前验证的是动态形状 FP32 PP-OCRv6 Tiny 合同，Scalar 是稳定参考
路径。当前不承诺任意 ONNX 拓扑、动态模型发现、GPU 或 Android；Vector API
后端是 JDK 25 可选加速路径，对优化范围外的通用形状回退 Scalar。性能数字仅
用于同机研发比较，不构成发布性能承诺。

Java 源码采用 MIT License。仓库及发布候选包中的 PP-OCRv6 Tiny 模型、字典、
示例图片和派生 Golden 数据按 Apache License 2.0 重新分发；使用和再分发前请阅读
根目录的 `THIRD-PARTY-NOTICES.md` 与 `licenses/PaddleOCR-models-APACHE-2.0.txt`。
