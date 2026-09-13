# lw.PPOCR.Java

[English](README.md)

A lightweight pure-Java PP-OCRv6 inference runtime with no native
dependencies. 轻量级纯 Java PP-OCRv6 推理运行时，无原生依赖。

- 不依赖 Python
- 不依赖 Paddle Inference
- 不依赖 ONNX Runtime
- 不依赖 OpenCV 原生库
- 不使用 JNI
- 与 `lw.PPOCR.C` 共用 LWM v0.1 模型格式
- 提供 Scalar 正确性路径，以及可选的 JDK 25 Vector API 后端

## 当前里程碑

本仓库目前已经能够运行完整的 PP-OCRv6 Tiny DET/CLS/REC 流程。防御式
LWM v0.1 加载器会校验完整的不可信模型数据流；运行前会解析动态形状；
基于张量生命周期的工作区规划器会复用对齐后的张量存储，全程不依赖原生内存。

Scalar 正确性后端覆盖仓库内 Tiny 模型使用的全部算子，包括标准二元广播、
BatchNormalization、Conv/ConvTranspose、池化、Resize、布局变换、Softmax，
以及 rank-3 × rank-2 MatMul。Golden 测试会将 Java 图输出和 16 行完整 OCR
结果与锁定版本的 `lw.PPOCR.C` 基线进行对比。

PP-OCR 层目前包含与 C 版本兼容的 DET/CLS/REC 预处理、DB 后处理、透视裁剪、
UTF-8 字典加载、贪心 CTC 解码、方向校正、阅读顺序排序，以及公开的
`PaddleOcr` 流水线。阈值和阅读策略通过不可变的 `PaddleOcrOptions` 提供；
并发调用方可以使用带独立 Session 的 `OcrWorkerPool`。

CLS 和 REC 支持相互独立的可选并行度，同时保持输入顺序和阅读顺序。
应用程序可通过 `PaddleOcrOptions.setClassificationParallelism` 和
`setRecognitionParallelism` 主动启用，二者默认值均为 1。CLS 会把固定形状的
文本行分发到共享已解码模型常量的多个 Session；REC 会优先排队预计宽度最大的
任务组，使空闲 Worker 能立即处理下一组任务，同时避免创建重复 Session。

可选的 JDK 25 Vector API 后端会加速 Tiny 模型使用的全部 Conv 配置、DET 2×
上采样 ConvTranspose、MatMul、归约、激活函数和二元广播。它还会在校验张量连接、
形状、常量和中间结果独占关系后，融合 Tiny 模型使用的精确五节点
`DIV -> ERF -> ADD -> MUL -> MUL` GELU 表达式。通用但未专门优化的形状会继续
回退到 Scalar 正确性实现。

运行时刻意不解析 ONNX。模型转换仍然是 `lw.PPOCR.C` 及其转换器负责的离线工作。

## 构建

项目以 JDK 25 为设计基线。Core 源码生成 Java 8 字节码，并且只使用标准 API；
`lw-ppocr-vector` 面向 JDK 25，编译和运行时均需要启用孵化阶段的 Vector API。
这个可选模块与 Scalar 正确性路径完全分离。

```text
mvn verify
```

## 发布版本

`0.1.0` 是首个 1.0 之前的正式版本。Tag 构建会生成发布候选包，其中包含三个运行时
JAR、PP-OCRv6 Tiny LWM 模型、字典、示例图片、文档和许可证声明。每个 ZIP 都附带
SHA-256 文件；CI 会先解压候选包并运行一次完整 OCR，再上传构建产物。

当前产物尚未发布到 Maven Central。可以执行 `mvn clean install` 安装到本机 Maven
仓库，或使用 Tag 对应的 GitHub Actions Artifact 中的 JAR。维护者发布步骤参阅
[发布清单](docs/releasing.md)，版本变化参阅 [CHANGELOG.md](CHANGELOG.md)。

## 项目结构

```text
lw-ppocr-core/       LWM 模型层与 Scalar 安全基础实现
lw-ppocr-imageio/    可选的标准 Java BufferedImage/ImageIO 适配器
lw-ppocr-vector/     可选的 JDK 25 Vector API 后端
lw-ppocr-benchmark/  无额外依赖的加载和性能基准工具
```

GitHub Actions 是本仓库的权威构建环境，会在 Linux、Windows 和 macOS 上使用
JDK 25 编译和测试。Linux 性能任务会输出模型加载、DB 后处理、预处理、模型工作量，
以及 Scalar、Vector、Vector 四个 CLS/REC Worker 三种模式的完整 OCR 结果。
完整 OCR JSON 会分别记录阶段耗时、GC 活动、模型内存、保留堆和瞬时堆。
Schema 3 的墙钟时间测量不启用算子分析器，之后再单独运行一次已预热的诊断调用。
汇总字段 `operators` 仍然保留，`stage_operators` 会拆分 DET、CLS 和 REC；并行模式下
的算子时间是所有参与线程耗时之和。`stage_hot_nodes` 会报告最慢的已解析图节点及其
张量形状，便于定向优化。

专项性能基准覆盖以下关键工作负载：

- REC 投影 MatMul：`60 × 80` 乘以 `80 × 6906`；
- REC stride-2 Conv：`[1,24,24,480]` 到 `[1,48,12,240]`；
- DET stride-1 Conv：检测边长限制 320 和 960 下的 `[1,64,80,80]`、
  `[1,64,128,128]` 到 16 通道；
- DET 2×2 stride-1 MaxPool：`[1,16,160,160]` 同尺寸输出；
- DET stride-2 Conv：`[1,3,320,320]` Stem，以及检测边长限制 320 和 960 下的
  `[1,32,160,160]`、`[1,32,256,256]` 到 16 通道；
- DET 2×2 同尺寸层：`16 -> 8` 和 `8 -> 16` 通道；
- CLS 重复 5×5 depthwise 层：`[1,64,5,80]`，同一次运行内对比 Scalar 和 Vector。

专项基准可以把内核加速与完整流水线的调度噪声分开。性能输出仅作为开发信号；
v0.x 阶段不会将其作为发布门禁。

同步 OCR 调用之间会复用已准备好的推理 Session、预处理数组、DB 几何临时空间，
以及逐行透视裁剪像素缓冲区。因此一个 `PaddleOcr` 实例设计为同一时刻只由一个
调用方使用；需要并发时，请使用独立实例或 `OcrWorkerPool`。

对于使用 AWT/ImageIO 的应用，`lw-ppocr-imageio` 还提供面向 `Path`、
`InputStream` 和 `BufferedImage` 的 `PaddleOcrImageIo` 便捷方法。
Core 模块本身仍然不依赖 AWT 和 ImageIO。

Maven 依赖、模型目录、BGR/ImageIO 用法、生命周期和并发指导请参阅
[安装文档](docs/installation.md)。

## 使用边界

`0.1.0` 当前经过验证的范围是仓库内动态形状 FP32 PP-OCRv6 Tiny 模型集。运行时不承诺
兼容任意 ONNX 拓扑，也不提供自动模型发现、GPU 或 Android 支持。Vector API
后端是可选组件，对于专门优化范围之外的形状仍会保留 Scalar 回退路径。
图像解码由可选的 `lw-ppocr-imageio` 模块单独提供。

## 许可证

Java 源码采用 [MIT License](LICENSE)，版权所有 © 2026 天天代码码天天。
仓库内 PP-OCRv6 Tiny 模型资产及其 Golden 派生数据按 Apache License 2.0
重新分发，详情见 [第三方声明](THIRD-PARTY-NOTICES.md)。

## 联系与支持

- 作者：天天代码码天天
- QQ：819069052
- QQ Group: 天天代码码天天 | 群号: 264292622
- 项目地址：<https://github.com/lxw112190/lw.PPOCR.Java>

如果项目对你有帮助，可以扫码支持维护：

<img src="docs/assets/sponsor.jpg" alt="捐赠二维码" width="240">
