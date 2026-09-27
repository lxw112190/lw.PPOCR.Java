# 本机 x64 完整 OCR 优化前基线（2026-09-27）

## 代码与环境

- Java 基线：`b43557245911917e7956960aef2ad73fcff2295f`。
- C 参考：`E:\My-Code\PPOCR\lw.PPOCR.C`，检查时 HEAD 为 `8e32fbddd08b3b102e9165131dde756827930918`。
- CPU：AMD Ryzen 7 7735H with Radeon Graphics，8 个物理核、16 个逻辑处理器。
- 内存：16,366,567,424 字节物理内存（约 15.24 GiB）。
- 系统：Windows 10 企业版，10.0.19045，build 19045。
- 电源方案：平衡，`381b4222-f694-41f0-9685-ff5bb260df2e`。
- JDK：Oracle Java 25.0.4.1，HotSpot 64-Bit Server VM，`25.0.4.1+1-LTS-5`。
- JDK 路径：`C:\Program Files\Java\jdk-25.0.4.1`。
- Vector API preferred species：256 bit，8 个 float lane。

运行源代码来自 `git archive b435572` 的快照。portable 模块使用 `--release 8`，Vector 模块使用 `--release 25 --add-modules jdk.incubator.vector`。不使用工作区未提交候选，也不使用已有 `target/classes`。

## 固定输入与测量口径

- 图片：`E:\My-Code\PPOCR\lw.PPOCR.C\models\ppocrv6-tiny\sample.jpg`，500×500，16 行。
- 图片 SHA-256：`30C417C9F758A3B62718729F5A944F7D2E10CDD2BDE0E8CE6785523DDB68FFE9`，与 Java Golden 图片相同。
- 模型：Java 仓库中的 Tiny DET／CLS／REC LWM 和字典。
- DET 最大边：960；backend：vector；vectorBits：preferred；pointwise block：auto（本基线为 12）。
- JVM：`-Xms64m -Xmx512m`，分别设置 `-XX:ActiveProcessorCount=1/4/8`，OCR parallelism 为 auto。
- Steady：每个独立 JVM 预热 10 次、计时 20 次；1／4／8 CPU 顺序重复三轮。
- Comparable：另起 JVM，4 CPU，预热 1 次、计时 99 次。
- 单张图片重复执行，不是数据集准确率测试。图片加载和模型加载不计入每次 OCR 时延；每次 OCR 包括检测预处理、DET、DB、裁剪、CLS、旋转、REC 和排序。
- FullOcrPerformanceMain 额外执行一次算子 profile，该次不计入 timed latency／allocation／GC，但包含在外部整个 JVM 生命周期内存峰值中。

“CPU”是 JVM 可见 CPU 预算，不是线程亲和性，也不等于 OCR worker 数。基线 auto 在 CPU=1 时使用 1 个 CLS／REC worker，CPU=4/8 时均使用 4 个 CLS／REC worker。当前此基准尚未把 ParallelismPlan 中 DET／REC 算子内并行参数接入执行器。

进程内存由外部 PowerShell 每 50 ms 采样 Windows `WorkingSet64` 和 `PeakWorkingSet64`。下表为整个 JVM 生命周期的 OS 峰值范围，包括加载、预热、计时和额外 profile；不是 timed-only RSS，也不是 JVM heap。Windows 内置 Linux procfs RSS 指标为 unsupported。

## 稳态完整 OCR

表中代表值是三次独立 JVM 的 median／P95 再取中位数，阶段均值为三轮阶段均值的算术平均。

| JVM 可见 CPU | 三轮 median（ms） | median 代表值（ms） | P95 代表值（ms） | 进程峰值范围（MiB） | 分配字节／OCR | timed GC |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 588.878 / 532.945 / 555.619 | 555.619 | 662.762 | 161.27–166.95 | 9,963–10,078 | 0 |
| 4 | 321.143 / 315.635 / 341.342 | 321.143 | 403.227 | 316.73–323.88 | 10,096–10,099 | 0 |
| 8 | 309.623 / 332.999 / 316.160 | 316.160 | 384.103 | 316.49–365.48 | 10,097–10,104 | 0 |

所有稳态运行均输出 16 行。4 CPU 中位数相对 1 CPU 下降约 42.2%；8 CPU 相对 4 CPU 仅下降约 1.55%，与轮间波动相比很小。

| JVM 可见 CPU | OCR mean（ms） | DET mean（ms） | CLS mean（ms） | REC mean（ms） | retained heap delta（MiB） | workspace（MiB） |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 576.328 | 195.553 | 32.619 | 345.694 | 40.03 | 22.84 |
| 4 | 334.283 | 146.973 | 10.807 | 174.162 | 46.60 | 23.57 |
| 8 | 327.669 | 145.116 | 10.823 | 169.411 | 46.60 | 23.57 |

## Comparable（独立 JVM，1 次预热 + 99 次计时）

| JVM 可见 CPU | mean（ms） | median（ms） | P95（ms） | DET（ms） | CLS（ms） | REC（ms） | 进程峰值（MiB） | 分配字节／OCR | timed GC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4 | 340.995 | 322.969 | 380.905 | 156.038 | 10.850 | 171.658 | 380.83 | 64,340,129 | 50 |

同样输出 16 行。1 次预热的 allocation／GC 含运行升温阶段，不能替代上面的稳态分配指标；它与稳态 10 次预热后约 10 KB/OCR、GC=0 的差异也需要保留在后续完整 OCR 比较中。

## 后续整体优化的比较合同

以 4 CPU、DET960、相同 Tiny 资产和图片、相同堆大小的端到端 OCR 为主对比；同时保留 1／8 CPU profile，以检查并行扩展和内存。

C 版可迁移的整体方向包括：物理布局和预编译执行计划、卷积 epilogue 融合、DET 算子分片、REC 行级与算子内并行的统一预算、CTC 列块权重复用。下一版以完整 OCR 的时延、输出正确性和进程内存为验收对象。微内核数字只用于定位热点。

## 原始记录和复现

完整原始结果位于忽略目录：

`benchmark-results/local-20260927-094907-b435572/`

其中保存 `environment.json`、`results.json`、每个 profile 的 raw JSON 和 stderr，以及编译使用的源码快照与 classes。`environment.json` 记录所有模型和字典 SHA-256，`results.json` 记录每次的精确命令。该目录是本机证据，文档可以提交。

从仓库根目录复现：

```powershell
.\scripts\measure-local-ocr.ps1 -Revision b435572 -Replicas 3
```

比较新版本时把 `-Revision` 换成该版本提交 SHA，其余条件保持一致。机器处于平衡电源方案，结果含实际调度／频率噪声，不把其他机器或 C 版文档数字当作同机 A/B。

## 第一轮端到端优化候选

候选基于上述提交的未提交源码修改；冻结快照、源码差异、环境、精确命令及原始数据保存在
`benchmark-results/local-20260927-114459-b435572/`。目录中的 `b435572` 是起点，不是候选代码已提交的声明。
这轮没有修改 C 仓库，也没有调整 DB 阈值、图片、模型、字典或检测尺寸。

迁入的两条整体路径：

- AUTO 的 DET/REC intra-op 预算实际接入图执行器，大型卷积按输出通道、MatMul 按行分片；调用线程参与运算，后台共用最多 7 个守护线程，不为每个宽度 Session 创建线程池。任意分组卷积、batch>1 和小算子保留串行。
- Vector REC 末端投影采用列块外层调度：128 个类别的权重块在最多 32 行之间复用，减少每 4 行重新遍历全部权重。保持内积累加顺序和原有 softmax，未引入 FMA 或改变解码规则。该做法借鉴 C 版 panel-outer 路径，但不是把 C 的 NHWC 执行器整体移植到 Java。

三轮独立 JVM，仍为 10 次预热 + 20 次计时，统计口径与基线一致：

| JVM CPU | 原 median（ms） | 候选三轮 median（ms） | 候选代表值（ms） | 时延缩短 | 原／候选 P95（ms） |
| --- | ---: | --- | ---: | ---: | --- |
| 4 | 321.143 | 208.616 / 209.968 / 208.549 | 208.616 | 35.04% | 403.227 / 217.937 |
| 8 | 316.160 | 188.522 / 195.467 / 194.354 | 194.354 | 38.53% | 384.103 / 206.000 |

候选三轮阶段均值：4 CPU DET 81.802 ms、REC 119.931 ms；8 CPU DET 83.609 ms、REC 100.459 ms。
全部仍输出 16 行。另有一次 1 CPU 检查：median 453.211 ms，相比基线代表值 555.619 ms 缩短约 18.43%；它只测一次，不与三轮统计等同。

### 内存和升温代价

| JVM CPU | 原进程峰值（MiB） | 候选进程峰值（MiB） | 原 retained heap（MiB） | 候选 retained heap（MiB） |
| --- | --- | --- | ---: | ---: |
| 4 | 316.73–323.88 | 301.19–333.68 | 46.60 | 50.23 |
| 8 | 316.49–365.48 | 420.77–454.58 | 46.60 | 50.22 |

峰值仍包含整个 JVM 生命周期，不是仅计时区间。图 workspace 仍为 23.57 MiB；增大的投影行块是额外 scratch，最新 JSON 单独输出 `projection_scratch_bytes`，不能把它漏算后宣称内存不变。8 CPU 进程峰值明显上升，当前候选不能称为内存全面优化。

4 CPU 三轮分配为 16,250 / 16,253 / 5,250,176 字节/OCR，对应 GC 0 / 0 / 1。
8 CPU 三轮约 33,735–33,754 字节/OCR，GC 均为 0。新执行路径在 10 次预热后仍可能继续 JIT 升温，不能只报告较小分配值。
另一次 4 CPU、50 次预热 + 30 次计时的诊断为 median 211.690 ms、分配 14,268 字节/OCR、GC 0；
它支持“充分升温后低分配”的判断，但不能替代同口径三轮中的异常值。

### 正确性与验收边界

本机 JDK 25 编译并运行 Core + Vector 全部 151 个 JUnit 测试通过，覆盖固定 C Golden 的
Scalar/Vector Full OCR、AUTO 并行、动态 REC/DET、分片偏移／尾部／异常恢复及投影分块位级一致性。
portable 快照仍以 `--release 8` 编译。DET320、8 CPU 的 CI 配置本机抽检 mean 148.573 ms、
分配 32,293 字节/OCR、GC 0，低于原有 400 ms／1 MB 门禁；未改动门禁或 CI 预热参数。
尚未运行 GitHub 三平台 CI，也没有在 100 张真实数据集上重测 CER。

这轮已证明整体时延收益，但 NHWC 物理布局、卷积 epilogue 融合和 8 CPU 启动峰值仍未解决，不能把整个 C x64 优化路线标为完成。

复测未提交候选：

```powershell
.\scripts\measure-local-ocr.ps1 -WorkingTree -Replicas 3 -ProcessorCounts 4,8 -SkipComparable
```

`-Warmup 50 -TimedIterations 30` 只用于另行诊断充分升温，不与默认 10/20 数据混用。
关闭 intra-op 或投影列块进行路径排查，分别使用 `-Dlwppocr.disableIntraOp=true`
和 `-Dlwppocr.disableProjectionPanel=true`。算子 profile 在调用线程记录节点完成的墙钟时间，
内部 shard 不重复计为独立节点；因此它不是所有算子线程 CPU 时间的加总。

## 第二轮：物理卷积计划与中间缓冲融合

冻结候选位于 `benchmark-results/local-20260927-122337-b435572/`，仍是未提交源码；
CPU、模型、图片、DET960、堆大小及 10 次预热 / 20 次计时均保持不变。

本轮把卷积参数和数组绑定提前到 Session 准备阶段，并将符合单消费者约束的
Conv、BN、通道偏置或残差、激活合成物理指令。MemoryPlanner 与执行器共享融合计划，
移除中间张量空间；后处理在最终输出上原地完成。它不是 NHWC 布局或寄存器写回融合。
图分支和公开输出保留原语义，Scalar 路径不受影响。

| JVM CPU | 基线 median（ms） | 第一轮（ms） | 本轮三次 median（ms） | 本轮代表值（ms） | 相比基线时延缩短 |
| --- | ---: | ---: | --- | ---: | ---: |
| 4 | 321.143 | 208.616 | 197.644 / 196.994 / 200.016 | 197.644 | 38.46% |
| 8 | 316.160 | 194.354 | 173.192 / 174.192 / 180.632 | 174.192 | 44.90% |

代表 P95 分别为 210.717 / 182.788 ms。三轮阶段均值：4 CPU DET 73.323 ms、REC 118.294 ms；
8 CPU DET 75.427 ms、REC 91.994 ms。相比第一轮代表值，整体时延进一步缩短约 5.26% / 10.38%。
全部仍输出 16 行；实际准备的融合计划累计为 DET 55、CLS 128、REC 136（跨缓存 Session 的合计，不是每图调用次数）。

### 内存与未完成事项

总图 workspace 从 24,713,216 降为 23,330,816 字节（约降低 5.59%）；
REC workspace 从 6,912,000 降为 5,529,600 字节（降低 20%）。
但 retained heap 仍约 50.28 / 50.27 MiB，没有观察到总保留堆下降。
进程全生命周期峰值为 4 CPU 336.75–339.41 MiB、8 CPU 405.38–437.77 MiB，
仍高于原基线，不能宣称整体内存已经优化成功。

4 CPU 三轮分配为 16,248 / 5,250,166 / 10,484,085 字节/OCR，GC 为 0 / 1 / 2；
8 CPU 为 33,735 / 33,735 / 33,754 字节/OCR，GC 均为 0。
保留全部异常值，不用增加预热或放宽门禁掩盖升温分配。

Core + Vector 共 154 个测试通过，额外固定 128 / 512 bit Vector 的四项测试分别通过。
冻结源码的 portable 模块仍以 `--release 8` 编译。
原 DET320 门禁口径（10 次预热、5 次计时）本机 8 CPU 检查 mean 98.620 ms、32,299 字节/OCR、GC 0；
补充开启 JFR 的 2 CPU 为 mean 187.895 ms、11,435 字节/OCR，
4 CPU 为 mean 127.664 ms、10,158,094 字节/OCR。
后者超过原 1 MB 分配门禁，说明该候选尚不能作为稳定通过 CI 的完成版本。
JFR 在升温期间记录到 `VectorMatMulKernel.multiplyRange` 的 Vector / float 数组分配，
后续应修复该路径的编译升温行为，不能仅报告 8 CPU 的通过结果。

曾尝试寄存器 epilogue 内核，但端到端时延和分配明显退化，已移除该实验，保留原卷积内核。
未修改 C 仓库、CI 阈值或预热次数；未提交、推送，也未运行 GitHub 三平台 CI 或真实 100 图 CER。
NHWC 布局、寄存器写回融合、冷启动及多核峰值内存仍属于后续工作。

可用 `-Dlwppocr.disableConvFusion=true` 做融合路径 A/B 排查。

## 第三轮：MatMul 升温分配修复后的复测

第二轮 JFR 证据指向大型 MatMul 方法的 Vector 临时对象分配。
将四行双向量算术提取为更小、只接受数组和标量参数的热内核，保持累加顺序，
让列微块直接调用该内核。修复后的冻结快照位于
`benchmark-results/local-20260927-123543-b435572/`；删除未使用局部变量和缩进整理不改变该快照的运算。

| JVM CPU | 三次 median（ms） | 代表 median（ms） | 代表 P95（ms） | 三次分配字节/OCR | timed GC |
| --- | --- | ---: | ---: | --- | --- |
| 4 | 194.909 / 196.972 / 195.312 | 195.312 | 213.421 | 16,375 / 16,373 / 16,365 | 全部 0 |
| 8 | 177.350 / 173.342 / 177.876 | 177.350 | 183.962 | 33,858 / 33,903 / 54,162 | 全部 0 |

相较原基线代表 median，时延分别缩短约 39.18% / 43.91%。
4 CPU DET / REC 阶段均值为 73.435 / 116.134 ms；8 CPU 为 74.704 / 92.124 ms。
workspace 不变，retained heap 为 50.30 / 50.32 MiB；
进程全生命周期峰值为 321.96–342.29 / 401.26–452.57 MiB，仍没有解决多核峰值问题。

修复后重新运行全部 154 个测试通过。DET320、4 CPU、不带 JFR 的原口径抽检为
mean 122.799 ms、15,475 字节/OCR、GC 0。补充带 JFR 的 4 / 8 CPU 为
mean 125.840 / 102.460 ms、15,470 / 32,485 字节/OCR、GC 均为 0。
但是带 JFR 的 2 CPU 仍出现 mean 217.130 ms、64,027,435 字节/OCR、GC 3，
因此不能把“所有 CPU 配置均稳定通过分配门禁”作为已完成验收。
这条记录是诊断结果，不与未启用 JFR 的正常性能数据混用。

随后移除 JFR、保持原命令其余参数不变，2 CPU 抽检为 mean 186.690 ms、median 184.820 ms、
11,792 字节/OCR、GC 0。正常命令的 2 / 4 / 8 CPU 抽检均在原门禁内，
但这些是本机单次抽检，并不证明 GitHub 三平台 CI 已通过或冷启动分配已解决。

## 空间卷积候选与 0.2.1 发布验证状态

空间卷积新增四像素输入打包和输出通道方向的连续权重，在 DET、CLS、REC
共用预编译执行接口；模型权重缓存跨宽度／worker Session 复用，线程 scratch
在准备阶段分配。它仍不是整图 NHWC 布局，也没有改变归约顺序、模型或阈值。

未限制缓存的实验快照为 `benchmark-results/local-20260927-125125-b435572/`。
同口径三轮代表 median：1 CPU 481.340 ms、4 CPU 194.003 ms、8 CPU 163.185 ms。
8 CPU 进程峰值范围为 299.76–328.08 MiB；额外权重 103,136 字节，
但加上投影权重后超过原 2,300,000 字节门禁，因此不作为最终发布配置。

80 KiB 缓存预算、超预算回退旧内核的实验快照为
`benchmark-results/local-20260927-125946-b435572/`。该实现通过本机离线 Maven
`verify`，共 161 个测试，无失败、错误或跳过；Core class major version 为 52。
其三轮代表 median：1 CPU 478.094 ms、4 CPU 188.828 ms、8 CPU 166.284 ms。
不过 8 CPU 进程峰值回到 395.40–427.10 MiB，暴露了回退路径的升温代价。

当前 0.2.1 候选保留 80 KiB 模型缓存预算，超预算时改为在复用的 Session
scratch 中重打包，继续执行相同空间内核；不按调用创建数组。
`packed_weight_bytes` 已合计投影和空间权重，新增空间子项和 scratch 字段单独记录。

**该最后改动及 0.2.1 版本更新尚未完成复测**：执行审批服务因账户额度限制
拒绝启动验证命令，并非测试通过。上述快照数字只对应各自源码，不能当作当前
0.2.1 的实测结果。发布前先执行 `mvn --batch-mode --no-transfer-progress clean verify`，
推送 main 并等待三平台测试和原性能门禁通过，再创建新版本标签；不得放宽门禁。
完整 OCR Golden 不替代真实数据集 CER 验证。C 仓库保持只读。
