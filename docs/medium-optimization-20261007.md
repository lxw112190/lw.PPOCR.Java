# Medium 端到端优化记录（2026-10-07）

本次为 `Unreleased` 优化，基线为 `630f528e961c069641588d7f782df6cbd11a5cde`
（v0.3.0 发布准备提交）。版本号、公开 OCR API、模型、检测阈值、字典均未改变。

本记录描述默认非融合乘加路径；后续可选 FMA 与 C# 的独立对照见
[Java/C# CPU 对照记录](java-csharp-comparison-20261007.md)，两轮基线不同，不直接叠加百分比。

## 结论

在下面的同机 sample.jpg 条件下，Medium 两轮平均耗时从 **4.171 秒降至
2.956 秒，减少 29.1%**。主要收益来自 REC；没有使用缩小输入、跳过 CLS、
删减检测框或换模型的方式加速。该结果不代表所有图片、CPU 或 Scalar 后端。

## 环境与测量口径

- Windows 10 Enterprise 10.0.19045，AMD Ryzen 7 7735H，8 核 / 16 逻辑处理器，16 GB 内存。
- Oracle HotSpot JDK 25.0.4.1（25.0.4.1+1-LTS-5）；Vector preferred 为 256 位。
- Windows 平衡电源方案，未固定 CPU 亲和性；每个测量使用独立 JVM。
- JVM：`-XX:ActiveProcessorCount=8 -Xms64m -Xmx2g`；并非 OS CPU 亲和性限制。
- 模型使用 `release/ppocrv6-onnx-manifest.json` 的固定 SHA-256 资产，所有九个资产已核验。
  Medium DET/REC、共享 Tiny CLS、18,710 类字典；没有 C 或 ORT runtime 依赖。
- 图片：仓库的 `lw-ppocr-core/src/test/resources/golden/ocr/sample.jpg`，500×500，
  SHA-256：`30c417c9f758a3b62718729f5a944f7d2e10cdd2bde0e8ce6785523ddb68ffe9`。
- DET limit=960，Vector AUTO：CLS 4 worker、REC 4 worker，DET intra-op=4、REC intra-op=2。
- 每轮 warmup=5（含第一次冷启动），随后计时 10 次；16 行。模型加载和单独的
  operator profile 调用不计入热机时间。性能测量时未同时运行 Maven 或其他基准。
- 进程峰值采用 Windows `PeakWorkingSet64`，每 50 ms 轮询，覆盖模型加载、
  warmup、计时、profile 的整个进程生命周期；不能当作纯计时阶段 RSS 峰值。
- GC 后堆、MXBean heap pool peak、进程工作集、每次分配是不同指标，不能混为一谈。

## Medium 两轮结果

单位为 ms；每一行是一个独立 JVM。第二轮使用同一份已编译代码，仅切换
`lwppocr.disableLargePointwise`。第一轮的基线是原始提交，候选是同一 32 像素算法。

| 轮次 / 路径 | mean | median | p95 | DET mean | REC mean |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 / 原路径 | 4029.360 | 4023.661 | 4178.026 | 863.639 | 3156.500 |
| 1 / 32 像素面板 | 2832.527 | 2831.888 | 2860.865 | 759.844 | 2063.453 |
| 2 / 关闭优化 | 4312.305 | 4276.789 | 4574.173 | 874.941 | 3427.268 |
| 2 / 开启优化 | 3079.329 | 3061.692 | 3170.465 | 887.199 | 2181.495 |

分别减少 29.7%、28.6%。两轮各有相同的 10 次计时，因此两轮 mean 的算术平均
也是 20 次计时的平均：4170.833 → 2955.928 ms。没有把两轮 p95 平均后称作总体 p95。
REC 两轮平均 3291.884 → 2122.474 ms，减少约 35.5%。
DET 的单轮波动较大，不将它单独宣称为稳定加速。

### 内存

| 轮次 / 路径 | GC 后堆（MiB） | 进程生命周期峰值（MiB） | 分配 bytes/OCR | 计时 GC |
| --- | ---: | ---: | ---: | ---: |
| 1 / 原路径 | 431.462 | 834.805 | 243775 | 0 |
| 1 / 面板 | 431.460 | 834.234 | 240076 | 0 |
| 2 / 关闭 | 431.414 | 823.145 | 238233 | 0 |
| 2 / 开启 | 431.482 | 829.238 | 168191 | 0 |

本例各轮 `spatial_scratch_bytes=5016768`、`packed_weight_bytes=14448640`，
graph `workspace_bytes=97624064`。新面板需求没有超过原来 Session 已复用的最大
scratch，故本例不增加它；不承诺每一种图都零增量。新路径每 worker 的 scratch 为
`inputChannels * 32 * 4` bytes，额外请求上限 1 MiB，超过上限回退原内核。
进程峰值约 0.8 GiB，不能把“GC 后堆约 431.5 MiB”写成“程序总内存 431.5 MiB”。

## Tiny / Small 回归观察

相同模型来源、图片、8 CPU、warmup=5、iterations=10、DET960：

| 模型 | 关闭 mean（ms） | 开启 mean（ms） | 观察 |
| --- | ---: | ---: | --- |
| Tiny ONNX | 155.719 | 158.041 | +1.5%，单轮小幅波动；Tiny 没有符合新分派条件的 1×1 权重 |
| Small ONNX | 600.755 | 592.399 | -1.4%，不宣称有显著加速 |

两者仍为 16 行；Small scratch 2188000 → 2193056 bytes（约 +5 KiB）；
Tiny scratch 不变。这不是 Tiny/Small 的专项性能优化。

## 实现与选择

- 仅处理 batch=1、groups=1、1×1、stride=1、无 padding、输入/输出通道均至少
  256、空间至少 64 像素的卷积；其他情况保留原路径。
- 把 32 个空间像素的所有输入通道打包到每 worker 的私有、Session 复用 scratch。
  一块输入在所有输出通道计算期间保持紧凑，避免每个输出通道组扫描整个大 NCHW 输入。
- 四输出通道 × 两个向量的紧凑微内核，复用输入向量和权重广播；最终直接写 NCHW。
  分片按独立空间面板，线程之间不共享 scratch，不竞争输出。
- 使用原始 OI 权重，无新增全模型 packed 权重副本；没有每次调用的数组、FMA
  或归约顺序变化。bias 先累加，随后按原 IC 顺序乘加。
- 沿用现有 prepared Conv 生命周期和融合后处理；不改变框、裁剪、CTC 或模型形状策略。
- 曾比较通道分片、不打包输入及 64 像素面板，前两种端到端收益不足；
  64 面板 mean=3182.534 ms、scratch=5653568 bytes，慢于 32 面板且更占缓存，
  未保留为默认实现。没有用最好看的单算子结果代替端到端结果。

## 验证

- 本机 JDK 25 `mvn verify`：186 个 Java 测试通过。
- 新增 7 个回归测试：偏移与输出保护区、通道/空间尾部、bias/无 bias、
  Scalar 逐位一致、NaN/Infinity/负零、反序分片、四线程私有 scratch 复用、
  面板上界回退和 A/B 开关。固定 128/512 位也通过。
- 全部 16 组 ONNX 图输出通过原 max abs=1e-3、mean abs=1e-4 门限，
  REC/CLS argmax 一致。Medium REC960 最大误差约 4.11e-6。
- Small/Medium 各 16 行严格匹配既有 Java pipeline Golden。相同 Java 裁剪
  输入的独立 ORT 文本、CLS 标签、旋转一致，最大 score 误差分别
  5.31e-7 / 4.13e-7。
- C 的空格/冒号差异仍如实记录，不代表 Java/C 框完全一致，也没有声称大数据集 CER 已验收。
- 本机原 `check-performance-regression.py` 全部通过；Tiny LWM DET320 mean=97.942 ms、
  分配 34064 bytes/OCR。两个独立 2 CPU JVM 分配为 11910 / 11930 bytes/OCR，
  GC=0。未放宽 400 ms、1 MB/OCR、packed 权重或其他门限。
- 额外 2 CPU Medium 稳定性检查（warmup=5、iterations=5、DET960）：16 行、
  mean=5554.486 ms，分配 351166 bytes/OCR、GC=0；这是不同线程预算的分配检查，
  没有用它与 8 CPU 基线计算加速比，也不把 Tiny 的 400 ms 门限套给 Medium。
- CI 新增大通道 prepared/direct 诊断和固定 species 测试，保留三平台真实 ONNX 验证。
  这些是本机结果，GitHub CI 尚待推送后执行。

## 复现（PowerShell，项目根目录）

先安装 Maven / JDK 25，获取固定哈希模型并编译；只在准备阶段运行编译：

```powershell
$env:JAVA_HOME = 'C:\Program Files\Java\jdk-25.0.4.1'
$env:Path = "$env:JAVA_HOME\bin;$env:Path"
python scripts/prepare-onnx-models.py --output build-local-data/onnx-models
mvn --batch-mode --no-transfer-progress package -DskipTests
$java = "$env:JAVA_HOME\bin\java.exe"
$cp = 'lw-ppocr-core/target/classes;lw-ppocr-core/target/test-classes;lw-ppocr-imageio/target/classes;lw-ppocr-vector/target/classes;lw-ppocr-benchmark/target/classes'
$benchArgs = @(
  '-XX:ActiveProcessorCount=8', '-Xms64m', '-Xmx2g',
  '-Dlwppocr.benchmark.modelsRoot=build-local-data/onnx-models',
  '-Dlwppocr.benchmark.variant=medium',
  '-Dlwppocr.benchmark.image=lw-ppocr-core/src/test/resources/golden/ocr/sample.jpg',
  '--add-modules', 'jdk.incubator.vector', '-cp', $cp,
  'io.github.lxw112190.ppocr.benchmark.FullOcrPerformanceMain',
  '5', '10', '960', 'vector', 'auto'
)
New-Item -ItemType Directory -Path benchmark-results -Force | Out-Null
& $java '-Dlwppocr.disableLargePointwise=true' @benchArgs |
  Tee-Object benchmark-results/medium-before.json
if ($LASTEXITCODE -ne 0) { throw 'Baseline failed' }
& $java @benchArgs | Tee-Object benchmark-results/medium-after.json
if ($LASTEXITCODE -ne 0) { throw 'Candidate failed' }
```

再独立复跑一组；若要比较其他模型，将 `variant` 换为 `tiny` / `small`。
不传 `modelsRoot` 时仍使用原来的内置 Tiny LWM；只传 `variant=medium` 会明确报错，
不会悄悄运行 Tiny。JSON 新增 `model_variant` / `model_format`，其余原有 schema 5 字段保留。

原始本机 JSON、环境与 stdout/stderr 位于被忽略的
`benchmark-results/medium-20261007/`。它们和模型、target、临时脚本不纳入提交。
这份记录保留了对照数字和复现口径；以后性能变化应重新同条件测量，不与 CI 跑分直接混比。
