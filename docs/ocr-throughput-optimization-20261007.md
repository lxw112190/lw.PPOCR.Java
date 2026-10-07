# OCR 吞吐优化记录（2026-10-07，第二轮）

基线为 `157a70f`。版本仍为 0.3.0 / Unreleased，本轮没有发布或推送。
目标是完整 OCR，不通过缩图、减框、跳过方向分类或换模型提升成绩。
C# 对照固定为只读本机参考的 `3e4192f2ec03b84701d3c5c1658e277fdaa7aa12`，
不声明对比了任意未来版本。

## 保留的优化

1. 2×2、stride=2、无 padding 的 ConvTranspose：两个输出通道共享输入向量加载，
   按 batch/output-channel 分片；复用原算子线程池，不新增线程预算或临时张量。
   其他几何形状仍回退，保持 IC 累加顺序和非融合乘加。
2. Prepared Conv 后处理：卷积所有分片结束后，按向量对齐的通道范围处理
   normalization、post-bias、residual 和 activation，消除大输出的串行阶段。
   每段开始位置与原扁平向量边界一致；最后一段负责尾部。batch>1/小张量/不支持
   的后端使用原串行方法。计算及后处理共用同一个受限线程池，均等待所有任务完成。
3. 可选 `lwppocr.vectorFma=true`：符合条件的 1×1 卷积使用六输出通道 × 两空间向量，
   共十二个累加器；尾部保留四通道/单通道路径。固定 32 像素输入面板，
   保持 IC reduction order，不复制权重，不扩大原 scratch 上限。
   FMA 专属命中范围扩展至 IC/OC>=64（plane>=64）；默认非 FMA 仍为 IC/OC>=256。
   小于范围、分组卷积、非 1×1、stride/padding 不符合要求时继续回退。
   额外的 `lwppocr.smallFmaPointwise=true` 可扩展至 >=32，默认关闭；它必须与
   FMA 一起启用。初期 A/B 中 Small/Medium 受益，Tiny 并非每轮受益，故不自动启用。
4. 修复双源 Vector rearrange 的热机分配：JFR 观察到检测阶段的大量整数向量，
   本机 JDK 源码显示双源重排逐次调用 `shuffle.laneIsValid()`。现在改用正索引的
   单源重排及预先生成的 odd-lane blend mask，交错写回的结果不变。
   新路径和旧转置卷积回退都修复，不新增逐次临时数组或额外近似。

FMA 仍默认关闭；它改变浮点舍入，启用前必须验证实际数据集。
新增后端能力接口不改变公开 OCR API；Scalar 及不支持这些能力的后端保留回退。

## 回退与复现

单独关闭各项，必须在创建模型/session 前设置：

```text
-Dlwppocr.disableSixPointwise=true
-Dlwppocr.disableExtendedFmaPointwise=true
-Dlwppocr.disableParallelEpilogue=true
-Dlwppocr.disableParallelTranspose=true
-Dlwppocr.disableTransposePair=true
```

前两项在 FMA 模式恢复原四通道微内核与 >=256 通道命中范围；其余三项合用恢复原转置卷积和串行
prepared 后处理。固定模型、500×500 sample.jpg、CPU8、DET960、REC4 worker/
2 intra-op、DET4 intra-op、CLS4 worker；两端共享导出的 BGR 字节。
阈值 .3/.6/1.6、无 dilation、CLS .9、REC 192/320/480/640/960 不变。

复测入口（Windows PowerShell 7，依赖路径可通过脚本参数覆盖）：

```powershell
./scripts/compare-csharp-ocr.ps1 -Cpu 8 -Warmup 20 -Iterations 10 -Replicas 2
```

最后一轮针对可选 32 通道配置；只测 FMA 与 C#，不虚构未测默认基线：

```powershell
./scripts/compare-csharp-ocr.ps1 -Cpu 8 -Warmup 20 -Iterations 10 -Replicas 2 `
  -JavaModes fma -SmallChannelFma
```

独立应用的启动参数为 `-Dlwppocr.vectorFma=true -Dlwppocr.smallFmaPointwise=true`。
结果记录启用开关；环境记录编译输出 class SHA-256，避免把未重新编译的源码当作实测版本。

两轮顺序反转，独立进程，编译完成后才测性能；模型和字典核验固定 SHA-256。
JVM `-Xms64m -Xmx2g`、CLR 2 GiB heap cap；CPU预算不是亲和性。
过程工作集峰值每 50 ms 轮询，包含加载、warmup、计时和报告。
GC 后托管堆、进程工作集及每次分配分别记录，不能混用。

## 验证

- 干净 Maven clean verify：198 个测试；preferred、128、512 位向量测试通过。
- 大通道 Scalar/FMA 独立参考，偏移、bias、尾部、NaN/Infinity/-0、保护区、并行复用。
- ConvTranspose 同 Scalar 的位级对照；矩阵/转置模式切换及分片失败后可继续执行。
- 后处理检查 normalization/residual/各类 activation、尾部、保护区、失败后重用。
  GELU 在 HotSpot 解释执行/Vector EXP intrinsic 切换时可有末位差异，测试使用
  2e-7 容差；其余后处理检查位级一致，不引入新的 erf/exp 近似。
- 默认及 FMA 的 16 个真实 ONNX 图误差、REC类别序列、Small/Medium 16行 Golden，
  以及实际裁剪输入的独立 ORT CTC/CLS/score/rotation 核验通过。
- CI 三平台保留，新增固定 species 测试；原性能/分配门禁不放宽。
- 新增真实 DET 尺寸的转置卷积热机分配回归；30次预热后5次调用总分配上限
  100,000 B，仅在 preferred species / 支持分配计数的 JVM 验证。
  独立 ORT CI 覆盖 Scalar、Vector、FMA64、可选 FMA32 四种模式。

## 已淘汰的实验

可调 48/64/96 像素面板、四通道 × 三空间向量、无差别扩大非 FMA 小通道命中范围、额外打包
权重、空间卷积 FMA 和 SIMD 指数近似均未获得可靠整图收益，已从生产代码移除。
8 像素空间卷积也被移除：相邻三模型 A/B 均慢于4像素配置，不能仅凭更大的计算块采用。
其中非 FMA 打包在真实 OCR 中出现严重退化，不能用单算子 checksum/速度通过
代替端到端验证。关闭 intra-op 使 Medium 短测约 5.14 秒，同样未采用。

## 测量证据

初期相邻 FMA 整图对照（各 warmup5 / timed10，独立 JVM）：本轮候选
2696.721 ms，关闭六通道微内核、并行后处理、并行转置卷积、成对转置卷积这四项
后 3036.098 ms，减少 11.2%。当时 FMA 命中范围仍为 >=256，尚未修复双源重排。
这是本轮 Java A/B，不与历史不同时间的 C# 数据拼接，也不代表所有模型或图片。
### 最终无诊断干扰的 Java/C# 两轮

原始目录 `benchmark-results/csharp-comparison-20261007-111033-017/`。
配置为 FMA + 可选32通道，修复双源重排；CPU8、warmup20、timed10，两轮反转顺序。
表中的平均值是两轮进程 mean 的平均，不把排序样本的中位数称为平均值。

| 模型 | Java 两轮均值 ms | C# 两轮均值 ms | Java 耗时相对 C# |
| --- | ---: | ---: | ---: |
| Tiny | 129.061 | 128.745 | +0.25%，基本持平 |
| Small | 452.824 | 419.580 | +7.92% |
| Medium | 2272.787 | 2150.192 | +5.70% |

**尚未全面超过 C#，这轮不能作为“Java已经更快”的发布声明。**
Tiny第一轮略快、第二轮略慢；Small/Medium两轮均仍慢。完整OCR还有优化空间，
下一轮应以剩余 dense/stride-two/depthwise 空间卷积热点为目标，而非继续盲增分块。

| 模型 | Java 进程峰值 MiB（两轮） | C# 进程峰值 MiB（两轮） | Java / C# GC后托管堆 MiB |
| --- | --- | --- | --- |
| Tiny | 437.7 / 510.1 | 145.3 / 143.8 | 53.8 / 120.3 |
| Small | 628.7 / 504.9 | 370.2 / 375.8 | 146.4 / 309.3 |
| Medium | 940.7 / 1028.2 | 1249.3 / 1189.7 | 431.3 / 940.2 |

Java的托管堆较小；进程峰值只有Medium低于C#，Tiny/Small不能据堆数据宣称进程内存更低。
两端GC计数定义和收集器不同，不用计数直接比较停顿。

两端均16行。对应文本相同的行数为 Tiny15、Small16、Medium14；方向一致。
最大顺序框坐标误差分别1.588 / 1.154 / 1.245 px；Small有一行宽度桶Java640/C#480，
Medium有一行Java960/C#640。因此是同设置、同像素的端到端对照，不是完全相同裁剪的内核负载。
不能将这些文本差异直接算作本次新增回归，也没有数据集CER验收。

此前 >=64 FMA 配置的充分预热两轮（目录 `csharp-comparison-20261007-102714-678`）
Medium平均1990.172/C#2055.657 ms，Java均值略快3.19%，但一轮仍略慢。
它发生在另一时段、另一配置且尚未修复双源重排，不能挑选它替换最终表或拼接百分比。
这种波动说明还需要重复、多图集和更稳健的性能验收。

原始实测和实验日志位于忽略目录 `benchmark-results/csharp-overtake-20261007/`；
三模型横向对照位于 `benchmark-results/csharp-comparison-<timestamp>/`。
本轮全部源码和文档可提交，模型下载副本、C# 构建副本、日志、bin/obj 不纳入提交。

这张样图没有数据集 CER ground truth；框/文本及宽度桶差异必须单独报告。
同模型、阈值和像素并不保证两端裁剪完全相同，不能由本测试推断数据集 CER。

## 热机与分配

不能把尚未优化的 Vector API 解释执行作为热机性能。JFR 在加载/预热期间观察到
generic stride-two、depthwise、elementwise 等路径的临时向量对象；完整热机后应
另外检查线程分配计数。Tiny ONNX/DET960 的扩展 FMA 独立探针在 warmup10 时
仍约 9.93 MB/OCR，warmup20 的另一进程约 47.5 KB/OCR、GC0；不能据此保证
每个进程在第20次恰好达到同一个编译状态。最终两端统一 warmup20。
JFR诊断开启的时间不并入无监控横向性能表。

最后一次原 CI Tiny LWM/DET320 门禁实测 94.372 ms、35,296 B/OCR、GC0；两个可见 CPU
的独立重复为 12,155 / 11,963 B/OCR。所有原阈值保持不变并通过。
扩展 FMA Medium ONNX/DET960 的 warmup10 探针为 1973.031 ms、109,085 B/OCR、GC0，
后端共享空间打包权重 79,360 B，无新增 pointwise 打包权重。

FMA32 的后续进程暴露出 Tiny 约 13.2 MB/OCR、Medium 约 60.9 MB/OCR 的稳定分配，
不能简单依赖延长预热解决。单通道尾部隔离和去除空调用没有消除它，热机 JFR
与 JDK 源码最终指向双源重排的动态掩码。修复后的相同探针：

| ONNX / DET960 | 修复前分配 B/OCR | 修复后分配 B/OCR | 修复后计时 GC |
| --- | ---: | ---: | ---: |
| Tiny | 13,223,314 | 53,656 | 0 |
| Medium | 60,918,811 | 109,595 | 0 |

这些是独立完整 OCR 分配探针，不与 C# 未测的逐线程分配做直接比较。
修复后16个图的最大绝对误差为 9.6321106e-5（验收1e-3），
Small/Medium 实际裁剪的 ORT 文本、方向、score 核验通过；不改变输出几何或阈值。
