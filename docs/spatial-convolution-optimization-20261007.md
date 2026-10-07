# 空间卷积端到端优化记录（2026-10-07，第三轮）

承接 [第二轮记录](ocr-throughput-optimization-20261007.md)。此次改动没有发布或推送。
以本机现有工作树为基线，不把它误称为干净的 `157a70f` 提交；上一轮未提交的改动全部保留。
C/C# 参考保持只读，不新增 JNI、原生运行时或模型依赖。

## 实现范围

- 7×7、1×7、7×1 普通卷积：四输出通道共享一个输入向量，整个 IC/KH/KW
  reduction 期间累加器留在寄存器，最终只写一次输出。仅在 groups=1、stride=1、
  dilation=1、对称 same padding 和同尺寸输出时命中；支持 batch、偏移和通道尾部。
- 7×7、9×9 depthwise：两个相邻空间向量共享权重广播，各自保留完整 tap 累加器。
  支持 stride-height=1/2、stride-width=1、dilation=1、对称 same padding。
  单向量与 scalar 处理尾部；无逐次临时数组、额外权重副本或新增线程。
- 边缘只遍历有效 tap，不将 padding 乘以非有限权重；保持 Scalar 的 IC/KH/KW
  加法顺序和非融合乘加。此轮内核默认启用，但原有 pointwise FMA 仍默认关闭。
- 继续利用既有通道分片和受限算子线程池，公开 OCR API、图形状、阈值、宽度桶不变。

实现方式借鉴了只读 C 的寄存器空间块思路，而不是引入 C 库或直接移植平台 intrinsic。
Medium DET 第246层的权重为 `[32,32,7,7]`，第248层为 `[32,32,1,7]`；
第172层为 `[256,1,9,9]`、groups=256。Small DET 还有四层 `[96,1,7,7]` depthwise。

回退开关（创建 Session 前设置；同一 JVM 的诊断开关不宜并发切换）：

```text
-Dlwppocr.disableWideConv=true
-Dlwppocr.disableDepthwiseRegister=true
```

这两项关闭的是本轮路径，上一轮的转置卷积、并行 epilogue 和可选 FMA 不受影响。
基准 JSON 的 `wide_conv_register` / `depthwise_register` 表示启用资格，并非实际命中层数。
C# 对照环境另外记录新增两个内核的编译 class SHA-256。

## 算子诊断

入口 `WideConvPerformanceMain vector <profile> 30 30`，CPU8、preferred Vector species。
两种路径分开启动 JVM，未同时运行其他验证。以下是诊断值，不新增 CI 速度阈值。

| profile | 原路径 median ms | 新路径 median ms | 减少 |
| --- | ---: | ---: | ---: |
| dense7：IC/OC32、128×128 | 292.844 | 104.218 | 64.4% |
| strip7：IC/OC32、128×128 | 69.424 | 24.881 | 64.2% |
| depthwise9：256通道、128×128 | 42.274 | 30.550 | 27.7% |
| depthwise7：96通道、128×128 | 9.901 | 8.576 | 13.4% |

每组 checksum 完全相同。不是整个 OCR 变快三倍。
3×3 depthwise 实验从0.336变为0.384 ms，变慢；不纳入新路径。
7×7/9×9 的双向量路径是在单向量实现之后验证的，不能混用两者成绩。

## 第一阶段整图 A/B

在加入双向量 depthwise 和 Small 的7×7覆盖前，已做两轮反转顺序的 Medium 对照。
CPU8、FMA+可选32通道、固定 sample 500×500、DET960、warmup20/timed10。

| 轮次 | 关闭本轮两项 mean ms | 第一阶段候选 mean ms |
| --- | ---: | ---: |
| 1，control先 | 2016.595 | 2004.449 |
| 2，candidate先 | 2098.618 | 1954.870 |
| 两进程均值平均 | 2057.607 | 1979.660 |

平均减少3.79%；16行的文本、框坐标、score、方向及宽度桶完整 JSON 一致。
计时 GC 均为0。第一轮候选 p95=2515.748 ms，明显有抖动；第二轮 p95=2004.016 ms。
此表只描述第一阶段 Java A/B，不叠加上一轮改善，也不代替最终 C# 对照。

## 最终三模型 Java/C# 对照

原始目录 `benchmark-results/csharp-comparison-20261007-122605-815/`。
JDK25 / .NET10，Ryzen7 7735H、平衡电源策略，CPU预算8（不是亲和性）。
同一组固定哈希 FP32 模型与解码 BGR，500×500 sample.jpg；DET limit960，实际输入512×512；
阈值0.3/0.6/1.6，无dilation，CLS0.9，REC 192/320/480/640/960。
DET4 intra-op、4 line worker、REC2 intra-op；JVM `-Xms64m -Xmx2g`，CLR 2GiB heap cap。
独立进程 warmup20/timed10，两轮反转顺序。没有开启 JFR，也不在编译/验证时计时。
C# 为只读固定提交 `3e4192f2ec03b84701d3c5c1658e277fdaa7aa12` 的工作树外构建副本。

复现命令：

```powershell
./scripts/compare-csharp-ocr.ps1 -Cpu 8 -Warmup 20 -Iterations 10 -Replicas 2 `
  -JavaModes fma -SmallChannelFma
```

Java 同时启用 `-Dlwppocr.vectorFma=true -Dlwppocr.smallFmaPointwise=true`；
这两个 FMA 开关默认关闭，新空间卷积不依赖 FMA。以下不能描述成默认配置成绩。

| 模型 | Java mean ms（两轮） | C# mean ms（两轮） | Java均值 / C#均值 |
| --- | --- | --- | --- |
| Tiny | 123.997 / 127.133 | 114.166 / 115.762 | 125.565 / 114.964，耗时+9.22% |
| Small | 447.269 / 438.320 | 395.974 / 396.329 | 442.795 / 396.151，耗时+11.77% |
| Medium | 1872.804 / 1927.539 | 2090.991 / 2086.066 | 1900.172 / 2088.528，耗时-9.02% |

**Medium 在这张图的两轮中都快于固定 C# 参考；不是三模型已全面超过 C#。**
Small/Tiny 仍有差距。不能把不同时间的上一轮数字拼接成受控 Java A/B，
也不能将此前 Tiny 的接近持平当成此次结果。跨进程、JIT及运行环境存在波动。
相同裁剪的 Java A/B 中，最终 Medium 两轮的完整 line JSON 都与 control 相同。

| 模型 | Java进程峰值 MiB（两轮） | C#进程峰值 MiB（两轮） | GC后托管堆 MiB（Java / C#） |
| --- | --- | --- | --- |
| Tiny | 456.1 / 469.3 | 141.6 / 145.1 | 53.8 / 120.3 |
| Small | 502.7 / 487.8 | 374.8 / 376.1 | 146.4 / 309.3 |
| Medium | 959.1 / 965.5 | 1124.3 / 1133.1 | 431.2 / 940.2 |

进程峰值包含加载、warmup、计时及报告，50ms轮询；堆与工作集不可混用。
Medium 的 Java进程峰值较低，Tiny/Small仍较高；不是普遍的进程内存优势。
两端GC收集器及计数定义不同，不据计数直接比较停顿。

两端均16行、旋转一致，完全相同文本的行数 Tiny15 / Small16 / Medium14。
最大顺序框坐标误差1.588 / 1.154 / 1.245px；Small有一行Java640/C#480，
Medium有一行Java960/C#640。因此同设置、同像素，并非完全相同的裁剪/算子负载。
没有样本集 GT / CER 验证，不作普遍模型准确率或多图速度声明。

## 正确性与分配验证

- Maven verify 204项测试通过；preferred/128/512-bit 的 Scalar 位级对照通过。
- 覆盖窄图、batch、偏移、bias/null bias、通道与向量尾部、行步长、保护区、非有限数和 -0。
- 通道并行分片保持结果一致，重复执行可复用；不改变并发预算。
- 新增热机空间卷积分配测试：preferred species、支持计数的 JVM；预热30次，
  5次调用总分配<=100,000 B。不把软件模拟固定 species 作为分配门禁。
- 16个真实 ONNX graph output、REC类别序列及 Small/Medium 样图 pipeline Golden 通过。
  实际 Java 裁剪的独立 ORT CTC/CLS/rotation 通过，最大 score 误差约1.32e-6。
- 对照并不保证 Java/C# 裁剪完全相同，仍有既有标点、框与宽度桶差异；没有多图 CER 验收。
- CI 三平台和原有性能/分配门禁保留，增加两个新内核的固定 species 测试；
  本机通过不代表远程 CI 已执行成功。

最新实测的原 Tiny LWM/DET320 门禁：93.636 ms、34,918 B/OCR、GC0；
两次独立 CPU2 分配为12,184 / 12,107 B/OCR、GC0，所有原阈值不变并通过。
真实 ONNX / DET960 的独立分配探针（warmup20/timed5，FMA+可选32通道）：

| 模型 | mean ms | 分配 B/OCR | 计时 GC |
| --- | ---: | ---: | ---: |
| Tiny | 123.483 | 53,650 | 0 |
| Small | 452.851 | 81,872 | 0 |
| Medium | 1852.848 | 109,602 | 0 |

此表用于 Java 热机分配检查，不替换前面的独立 Java/C# 两轮耗时。
没有增加空间卷积权重打包或 scratch；所有新路径都保留原有常量数组。

## 下一步

Medium 的检测大核热点已降低，下一步应定位 Tiny/Small 的真实 stage 和小通道
pointwise/REC 投影开销，再做相邻反转顺序 A/B。不要将未覆盖大核的 Tiny 实测波动
解释成这一轮代码回归，也不要通过调小 DET、减行或跳过 CLS 追求超过 C#。
速度之外仍需固定裁剪的对照和多图 CER/阅读顺序验证，才能扩大结论。

原始记录位于忽略目录 `benchmark-results/csharp-overtake-20261007/`；
模型与 ORT 验证副本在 `build-local-data/`，不纳入提交。
