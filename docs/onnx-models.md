# ONNX models / ONNX 模型接入

主线现在可直接加载经过锁定的 PP-OCRv6 Tiny、Small、Medium FP32 ONNX。
Java core 仍然只依赖 JDK 标准库，生成 Java 8 目标字节码；Vector 后端需要 JDK 25。
已发布的 v0.2.1 包仍包含原来的 Tiny LWM，本次主线功能尚未发布。

## 获取模型

从仓库根目录运行：

```powershell
python scripts/prepare-onnx-models.py --output build-local-data/onnx-models
```

Python 只用于可选的下载和开发验证，不是 Java 应用运行依赖。也可以自行下载
[固定清单](../release/ppocrv6-onnx-manifest.json) 内的文件，并校验 SHA-256。
下载脚本拒绝覆盖 hash 不符的已有文件。

本机已有参考模型时，可只读复制：

```powershell
python scripts/prepare-onnx-models.py --output build-local-data/onnx-models --local-source E:/My-Code/PPOCR/lw.PPOCR.C/models
```

目录结构：

```text
onnx-models/
├── ppocrv6-tiny/
│   ├── det.onnx
│   ├── cls.onnx
│   ├── rec.onnx
│   └── ppocr_keys.txt
├── ppocrv6-small/
│   ├── det.onnx
│   └── rec.onnx
├── ppocrv6-medium/
│   ├── det.onnx
│   └── rec.onnx
└── ppocrv6-shared/
    └── PP-OCRv6_small_rec_dict.txt
```

Small 和 Medium 共用方向分类器及 18,710 类 REC 字典；Tiny 为 6,906 类。
不能混用两套字典。模型和字典按 SHA-256 锁定，不提交大体积 ONNX 文件和临时输出。

## Java 调用 / Usage

无需转换模型，原有 `PaddleOcr.load` 自动按文件内容识别 LWM 或 ONNX：

```java
Path root = Paths.get("build-local-data", "onnx-models");
Path variant = root.resolve("ppocrv6-small"); // 或 ppocrv6-medium
try (PaddleOcr ocr = PaddleOcr.load(
        variant.resolve("det.onnx"),
        root.resolve("ppocrv6-tiny/cls.onnx"),
        variant.resolve("rec.onnx"),
        root.resolve("ppocrv6-shared/PP-OCRv6_small_rec_dict.txt"))) {
    OcrResult result = PaddleOcrImageIo.recognize(ocr, Paths.get("sample.jpg"));
    System.out.println(result.getText());
}
```

上述类型分别来自 `java.nio.file`、`io.github.lxw112190.ppocr.ppocr` 和
`io.github.lxw112190.ppocr.imageio`。不使用 ImageIO 时，直接传入已解码的 BGR 图像。
可通过现有的 options/backend 重载使用 `VectorBackend`，启动时加
`--add-modules jdk.incubator.vector`。ONNX 与 LWM 可混用，例如保持原来的 Tiny CLS。
低层加载入口是 `ModelLoader.load(Path/InputStream)` 或 `OnnxLoader.load`；
`LwmLoader` 仍只接受 LWM。流由调用方关闭。

## 明确边界 / Supported subset

这不是通用 ONNX Runtime：

- 标准域、opset 7..14、IR 3..10；单输入单输出、batch=1、FP32 NCHW RGB。
- 数值算子沿用现有 25 种 IR 算子；Identity 消除，Constant 合并。
- Shape / Slice / Concat / Squeeze / Unsqueeze 的有界整数控制链在加载时折叠。
- 以两组代表尺寸识别 PP-OCR 图的变化轴；Reshape 至多一个变化轴。
  任意符号表达式、动态分支、动态 rank 不在支持范围。
- MatMul 支持 rank >= 2 的矩阵维和右对齐批次广播；不支持向量 MatMul。
- Slice 支持正步长、负索引、显式 ends；零长度数值输出和负步长不支持。
- Resize 只支持固定 scales 的 nearest / asymmetric / floor；不缩放 batch/channel。
- pre-opset-13 Softmax 只接受最后一轴，避免误执行旧版 flatten 语义。
- 拒绝自定义域、未知算子/属性、外部权重、稀疏权重、训练图、局部函数。
- 输入及归一化模型上限各 256 MiB；最多 4,096 节点、16,384 值、rank 8、
  每节点 16 属性、名称总量 8 MiB；更严格的 RuntimeLimits 也生效。
- FP32 常量可用 raw_data 或 typed float_data；typed 存储最多 1,048,576 元素。
  整数控制向量最多 8 元素。大权重需 raw_data。

REC 缓存仍使用 192/320/480/640/960 桶，DET 保留最大边和 32 对齐策略。
Small/Medium 更大的权重、注意力和激活会增加耗时及内存；
本次验证兼容性，不宣称速度与 Tiny 相同，也不改变原性能门禁。

## 可复现验证 / Validation

普通 `mvn verify` 包含不需网络/模型下载的 protobuf、动态形状、广播 MatMul、
有限 Slice 和拒绝边界测试，也保留原 Tiny LWM Golden。

真实模型验证的 Python 依赖仅用于独立测试 oracle：

```powershell
python -m pip install -r scripts/requirements-onnx-tests.txt
python scripts/generate-onnx-golden.py --models build-local-data/onnx-models --output build-local-data/onnx-golden
mvn --batch-mode --no-transfer-progress package -DskipTests
java -Xmx1g --add-modules jdk.incubator.vector -cp "lw-ppocr-core/target/classes;lw-ppocr-imageio/target/classes;lw-ppocr-vector/target/classes;lw-ppocr-benchmark/target/classes" io.github.lxw112190.ppocr.benchmark.OnnxCompatibilityMain build-local-data/onnx-models build-local-data/onnx-golden vector record-stage-inputs
python scripts/verify-onnx-full-ocr.py --models build-local-data/onnx-models --fixtures build-local-data/onnx-golden
```

把 `vector` 换成 `scalar` 可测试 Scalar。Linux/macOS classpath 分隔符用 `:`。
必须同时运行 Java 验证和最后的 Python 验证：前者导出实际每行输入，
后者独立检查 CTC 文本、置信度、CLS 标签和旋转。
三平台 CI 已配置这两条路径，产出图输出与完整 OCR 参考报告。

本机 JDK 25 验证：

- 七个模型解析；REC 17/192/320/480/640/960、DET 32/320/960 形状。
- 16 组独立 ORT 图输出：REC 17/192/960，DET 32×64/320×320，以及共享 CLS。
- Small/Medium 各 16 行 sample.jpg；相同裁剪输入下的 ORT 文本和 CLS 旋转一致。
- 图输出最大绝对误差门限 1e-3、平均误差门限 1e-4；score 误差门限 1e-3。

原 C 文本保存在 `golden/onnx/*-full-ocr.txt`，Java 管线文本另存
`*-pipeline-full-ocr.txt`，后者由相同裁剪的独立 ORT 验证。
两套记录不等同：Small 品牌行有 `OEMODM` / `OEM ODM` 差异，
Medium 品牌行有半角 / 全角冒号差异。不会通过删空格、归一化标点来掩盖差异。
独立 ORT 检查的是相同 Java 裁剪，不证明 C 与 Java 的 DB 框和裁剪完全一致，
也不代表大数据集 CER 已经验证。

## 参考与许可 / Provenance

- [ONNX protobuf specification](https://github.com/onnx/onnx/blob/main/onnx/onnx.proto)
- [lw.PPOCR.C](https://github.com/lxw112190/lw.PPOCR.C) 的 PP-OCR 形状控制折叠和有界加载思路。
  资产基线 `d58832dcf99264b28a9472ddd5159ba1279d323a`。
- [SimdPaddleOCR](https://github.com/sdcb/SimdPaddleOCR) 的托管 ONNX 读取和模型族参考。
- 本机 lw.PPOCR.Vulkan 的 ONNX 属性/形状契约及 Small/Medium 清单，用作只读对照。

三份参考源码均未修改；未将其原生库或托管库加入 Java runtime。
模型许可为 Apache-2.0，Java 代码仍为 MIT，见
[第三方说明](../THIRD-PARTY-NOTICES.md)。
