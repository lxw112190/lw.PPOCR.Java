# DET stem CI 时序稳定性修复（2026-10-07）

基线 `d915a29`；原远程标签 `v0.3.1` 指向该提交，此时序修复已提交为 `03f0c26`，
仍记录在 `Unreleased`，不包含在原标签中。后续 macOS 分配失败及 gather 修复见
[stride-two-gather-fix-20261007.md](stride-two-gather-fix-20261007.md)。

## 证据与边界

用户提供的 CI 中 `det-stem-stride-two-conv.json` median=8.525 ms，超过原5 ms门槛。
该发布准备提交只改版本和文档，不改卷积实现。本机旧协议10次预热的三个独立
进程 median=1.780 / 1.380 / 1.373 ms，未复现 CI 的同样数值。
没有该失败 runner 的原始采样和编译日志，不能声称已经证实是 runner 噪声或 JIT。

第一次仅加强预热到100次且不少于1秒，CPU2下三个进程出现1.289 / 4.615 / 1.403 ms；
慢进程的30次采样都约4～6 ms，不是只删除一个离群值就能解决。
这说明只增加预热并不足以稳定不同 JVM 的结果。

## 实现

1. 将 `VectorConv3x3Stride2Kernel` 的八输出通道向量 tile 从大外围循环提取为
   高频调用、数组入参的 `vectorTile`。保持原 IC/KH/KW 顺序、gather、bias、
   边界过滤、重叠尾块和非融合乘加；没有缩小输入、减少通道或新增临时张量。
   此拆分旨在让 C2 更早编译小热循环，不能据本机时序反推远程失败的唯一原因。
2. CI stem 固定请求100次预热，并至少运行1秒，两条件都满足才开始30次采样。
   不是按速度是否达标决定预热或选择重跑结果；实际调用次数与耗时都会写入 JSON。
   downsample 默认预热协议不变。
3. 保留全部按发生顺序的 `samples_ms`，输出 requested/actual warmup、elapsed、
   JVM版本和可见CPU。CI另上传 lscpu 和 JVM 信息，原失败时的 artifacts 上传仍保留。
4. 门禁核对固定320×320/IC3/OC16/output160×160、Vector stem、
   checksum `10973022362502076197`、预热与采样预算及全部样本的上中位数。
   **仍严格要求 median < 5 ms**，没有选择最快进程、丢弃慢样本或放宽分配门槛。

复现：

```text
java -XX:ActiveProcessorCount=2 --add-modules jdk.incubator.vector -cp <classpath> \
  io.github.lxw112190.ppocr.benchmark.DetStrideTwoConvPerformanceMain vector stem 100 30 1000
```

第五个参数是最低预热毫秒数，0～60000；诊断可以显式传0，但 CI门禁不会接受无时间下限的结果。

## 本机结果

Ryzen7 7735H / JDK25.0.4.1 / preferred256，同一fixture，独立顺序启动：

| 可见 CPU | 三进程 median ms | 实际预热调用数 |
| --- | --- | --- |
| 2 | 1.459 / 1.489 / 1.572 | 470 / 488 / 464 |
| 8 | 1.472 / 1.566 / 1.480 | 509 / 466 / 490 |

全部 checksum 与旧实现相同，5 ms门禁通过。这些不是远程runner成绩，也不是端到端提速声明。

- Maven verify及 preferred/128/512-bit Scalar 位级对照通过；偏移、batch、尾部、
  null bias、非有限权重和 padding 均覆盖。
- 新增 preferred-species stem 热机分配回归：100次预热后5次调用总分配<=100,000 B。
- Python门禁8项测试通过，包含原失败8.525 ms仍被拒绝、预热缺失/不足、篡改形状或
  checksum、漏报/伪造采样及假中位数不能过关。
- 原有全部 CI 门禁在本机通过：stem median=1.507 ms，Tiny LWM/DET320
  mean=96.278 ms、35,296 B/OCR、GC0；两次CPU2分配12,184 / 12,178 B/OCR。
  其他六个算子及所有原阈值也保持不变并通过，见 `stem-ci-final-gates.log`。

额外的 ONNX/DET960/20次预热分配探针里，Small约22.9 MB/OCR、Medium约60.9 MB/OCR，
不能套用此前某次进程的低分配数字。仅用HEAD的旧stride-two class覆盖新class后，
独立Medium对照仍为60,918,798 B/OCR，候选为60,918,792 B/OCR；检测阶段约61 MB。
这证明该量级分配也存在于旧内核对照，不能据新进程结果归因于本次拆分或宣称已解决。
应作为独立JIT/分配问题进一步取证；它不属于原Tiny CI分配门禁，也没有为它放宽任何门槛。
本次不宣称改善Small/Medium整体内存或跨语言性能。

本地原始结果在忽略目录 `benchmark-results/stem-ci-*.json` 和相应日志，不纳入提交。

## 标签注意

不要移动或强制覆盖已经推送的 `v0.3.1`。重新运行该标签 CI仍检出 `d915a29`，
不会自动带入本次 main 修复。先提交修复并确认新main CI，再单独准备新补丁版本
（例如0.3.2，须同步全部POM和发布说明）；新版本的标签与POM必须匹配。
