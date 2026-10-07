# Java/C# CPU 对照与实验 FMA（2026-10-07）

## 范围

这轮增加可重复的端到端对照工具，以及大通道 1×1 卷积的可选 FP32 FMA。
没有把 C#、ONNX Runtime 或任何原生库引入 Java runtime；不修改既有性能门限。
Java 的默认精确路径保持不变，上一轮 32 像素面板优化见
[Medium 优化记录](medium-optimization-20261007.md)。

C# 参考来自本机 `lw.PPOCR.C/build/SimdPaddleOCR-ci`，锁定提交
`3e4192f2ec03b84701d3c5c1658e277fdaa7aa12`。这是带内部比较宽度钩子的参考 checkout，
**不是对任意最新版 SimdPaddleOCR、NuGet 包或其默认配置的性能承诺**。
工具使用 `git archive` 导出独立的被忽略副本，只编译副本；不修改参考仓库。
该提交 CPU FP32 路径的 INT8 VNNI 1×1 开关关闭，不比较 GPU/量化模型。

## 对照口径

- 模型和字典逐项检查 `release/ppocrv6-onnx-manifest.json` 的 9 个 SHA-256。
- 仓库 `golden/ocr/sample.jpg` 为 500×500；图像 SHA-256 为
  `30c417c9f758a3b62718729f5a944f7d2e10cdd2bde0e8ce6785523ddb68ffe9`。
- Java ImageIO 仅解码一次，两边读取同一份 packed BGR 文件并记录像素哈希；
  模型加载、图片解码、结果核对和报告生成不计入 OCR 时间。
- DET 最大边 960（此图实际输入 512×512），bitmap=.3、box=.6、unclip=1.6、
  maxCandidates=1000、不 dilation；CLS 阈值 .9。
- 两边使用 REC 192/320/480/640/960 桶，C# 启用其既有内部 `UseLwAdaptiveWidth` 钩子，
  缺少钩子时明确失败。不能用 C# 默认 32 对齐宽度与 Java 宽度桶直接计算优化比。
- CPU 预算 8、4 行 worker、DET intra-op=4、REC intra-op=2；不是进程 CPU affinity。
- 每个配置独立进程，warmup=5（包含首张）、timed=10；复跑时交替运行顺序。
  各自保留运行时默认 GC，托管堆上限均为 2 GiB。
- Java JSON 强制 UTF-8，避免 Windows stdout code page 损坏中文对照文本。

同配置不保证同裁剪。报告检查实际 DET shape、CPU/线程预算、参数、像素、宽度桶，
并展示逐行文本、框坐标、旋转和识别宽度。如果框或宽度不同，会标记
`identical_ordered_crops=false`；不能把两端耗时差全部解释成计算内核差。
逐行比较按返回顺序执行，顺序不同也会显式表现为不一致，不自行重排消除差异。
这里只有一张样图，没有 GT 数据集，**不计算或宣称 CER、没有据此验收通用准确率**。

内存同时报告 post-GC 托管堆和 Windows 进程 lifetime PeakWorkingSet。
后者 50 ms 轮询 OS 峰值，包含加载、预热、计时及报告阶段；不是 timed-only 峰值。
JVM heap、CLR managed heap、进程工作集不能混为同一个指标。post-GC 值是近似诊断，
两种 GC 的 collection count 含义也不同，不能横向当作相同指标比较。

## 本机结果

Ryzen 7 7735H（8 核 / 16 线程）、Windows 10 企业版 19045、16 GiB 内存、
平衡电源方案；JDK 25.0.4.1、.NET 10.0、Vector preferred=256 bit。
Java 基于 `630f528` 加本轮未提交的面板/FMA/对照改动；以下是两轮各 10 次的
mean 的等权平均，不把不同轮次的 median/p95 混成一个统计量。

| 模型 | Java 默认乘加 ms | Java 可选 FMA ms | C# 参考 ms |
| --- | ---: | ---: | ---: |
| Tiny | 159.835 | 173.144 | 128.802 |
| Small | 607.025 | 560.173 | 391.555 |
| Medium | 3069.575 | 2460.700 | 1998.603 |

Medium 两轮 FMA 降耗时 19.19% / 20.50%，合并 mean 降 19.84%。
Small 两轮 3.23% / 11.58%，合并 mean 降 7.72%，波动更明显。
Tiny 没有符合条件的大通道卷积，本轮差异只是运行噪声，**不是 FMA 加速收益**。
Medium FMA 耗时仍比这份 C# 参考高约 23.1%，没有宣称持平或超越。
两端几何/宽度工作量仍略不同，以上是端到端现状，不是完全相同矩阵工作的内核比值。

| 模型 | Java FMA 进程峰值 MiB | C# 进程峰值 MiB | Java post-GC heap MiB | CLR post-GC heap MiB |
| --- | ---: | ---: | ---: | ---: |
| Tiny | 392–415 | 141–146 | 53.5–53.7 | 117.8–120.4 |
| Small | 511–524 | 373–385 | 146.3–146.4 | 309.7 |
| Medium | 836–838 | 1112–1173 | 431.1 | 940.2 |

Java 默认 Medium 进程峰值为 825–833 MiB，FMA 不是减少内存的改动。
本机 Java Medium 的进程峰值低于参考 C#，Tiny/Small 则更高。
Java 较小的托管堆不能直接解释成所有模型的进程占用都较小。

输出检查：两端三模型都是 16 行；Java FMA 与默认 Java 的逐行文本、旋转和
宽度桶在两轮全部一致。Java/C# 按返回顺序文本完全一致分别为 15/16、16/16、14/16。
Tiny 差异为半/全角右括号；Medium 两条差异为半/全角冒号。
Small 有一条宽度 Java=640 / C#=480；Medium 有一条 Java=960 / C#=640。
框坐标最大差约 Tiny 1.588 / Small 1.154 / Medium 1.245 像素。
因此三模型都明确标为 `identical_ordered_crops=false`，没有隐藏这一差异。

采用 UTF-8 修正后的原始报告目录是
`benchmark-results/csharp-comparison-20261007-084129-853/`。
初次诊断的旧 Windows code page 报告不用于文本结论或本表跑分。

## 本轮校验

- Maven `verify`：188 项测试通过；大通道/FMA 单测额外通过 128/512-bit species。
- Python 汇总契约 5 项、性能门禁 helper 4 项和发布布局 5 项通过。
- FMA 对齐 16 个 ORT 图输出：max absolute error 最高 8.893e-5，
  每个 case 的 mean absolute error 最高 2.632e-8；仍使用原 max=1e-3、mean=1e-4 容限。
  REC argmax、Small/Medium 完整 Java Golden 文本、同裁剪 ORT CTC/CLS/旋转均通过。
  ORT REC score 最大误差 Small 3.307e-7 / Medium 1.013e-6。
- 原 7 个聚焦性能门禁、Tiny 全 OCR 和两个独立 2 CPU 分配检查全部通过；
  Tiny DET320 mean=91.717 ms、34090 bytes/OCR、GC=0；2 CPU 分配
  11898 / 11904 bytes/OCR、GC=0。不更改门限，不用此 DET320 数字比较上面的 DET960。
- 独立 Medium FMA 分配检查（8 CPU、warmup=10、timed=5、DET960）：
  16 行、74352 bytes/OCR、GC=0、post-GC heap=452454728 bytes。
  此次 mean=2523.457 ms，仅用于分配稳定性验证，不混入前述两轮平均。
- CI 保留三平台测试，新增可选 FMA 的真实 ONNX 校验和固定 species 单测。
  以上是本机验证；远程 CI 尚未执行。

## FMA 开关

```text
-Dlwppocr.vectorFma=true
```

仅影响 `VectorLargePointwise` 的 prepared 卷积：仍复用每 worker 的 32 像素面板，
不复制权重、不新增每调用数组、不改变输入通道遍历顺序。
向量使用 `FloatVector.fma`，尾部使用 `Math.fma`，累加器在紧凑微内核内维护。
FMA 一次舍入与默认乘加两次舍入不同，不能声称 Scalar 位级一致。
单测以 `Math.fma` 为独立参考，覆盖偏移、bias、尾部、保护区、分片复用、
并行私有 scratch、NaN/Infinity/-0，并在 preferred/128/512-bit species 验证。
本节测量的旧配置中 Tiny 不命中该大通道路径；启用标志不代表每个模型都有加速。
后续 FMA 专属命中范围和较充分预热的对照见
[第二轮 OCR 吞吐优化记录](ocr-throughput-optimization-20261007.md)，不要混用两轮配置的结果。

目前保持实验开关：即使样图/Golden 通过，也必须先验证实际数据集 CER、
文字、框和资源使用，再考虑默认启用。全文输出 parity 不等于所有中间值位级 parity。

## 复现

Windows PowerShell 7、JDK 25、Maven、.NET SDK 10 和 Python 3；准备固定模型后运行：

```powershell
python scripts/prepare-onnx-models.py --output build-local-data/onnx-models
./scripts/compare-csharp-ocr.ps1 `
  -CSharpSource 'E:\My-Code\PPOCR\lw.PPOCR.C\build\SimdPaddleOCR-ci' `
  -JavaHome 'C:\Program Files\Java\jdk-25.0.4.1' `
  -Maven 'D:\Program Files\Apache\apache-maven-3.6.2\bin\mvn.cmd' `
  -Dotnet 'C:\Program Files\dotnet\dotnet.exe' `
  -Cpu 8 -Warmup 5 -Iterations 10 -Replicas 2
```

路径可通过参数修改。默认先编译并测试 Java，再编译 C# 副本，最后串行跑分；
`-SkipJavaBuild` 仅用于确认 Java 编译输出已更新时。
CPU 预算支持 1/2/4/8，任意其他预算未声明两端规划等价。
C# 源目录必须能导出锁定提交，不能悄悄换成不兼容的新提交。

完整环境、命令输出、逐行 JSON、summary.json 和 BGR 文件在
`benchmark-results/csharp-comparison-<timestamp>/`；参考副本位于
`build-local-data/csharp-comparison-<id>/`。二者均被忽略，不打进 release ZIP。
独立 runner 的 `bin/obj` 也被忽略；CI 仅测试汇总逻辑，不增加 .NET runtime 依赖。
