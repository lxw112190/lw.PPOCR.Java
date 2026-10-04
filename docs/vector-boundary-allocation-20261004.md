# Vector 空间卷积边界分配回归修复

## 问题与本机复现

CI 的七项 kernel 延迟检查和整链路延迟均通过，但 Tiny LWM、DET320、Vector AUTO、预热 10 次/正式 5 次时，`allocated_bytes_per_ocr` 达到 66,428,400 bytes，超过原有 1,000,000 bytes 门槛。

在 `eddd1bf` 本机基线上，使用 Ryzen 7 7735H、Windows 10 19045、Oracle JDK 25.0.4.1、preferred 256-bit Vector，同样参数并设置 `-XX:ActiveProcessorCount=2`，复现：

| 运行 | 平均 ms/OCR | 分配 bytes/OCR | 正式阶段 GC 次数 |
|---|---:|---:|---:|
| 修复前，2 CPU | 177.221 | 66,424,258 | 3 |
| 修复后，2 CPU，第 1 个独立 JVM | 178.880 | 11,925 | 0 |
| 修复后，2 CPU，第 2 个独立 JVM | 183.203 | 11,944 | 0 |
| 修复后，2 CPU，第 3 个独立 JVM | 183.810 | 11,557 | 0 |

修复前 4/8 CPU 运行未出现同样的大分配。不能只用更大的 CPU budget 验证分配稳定性，也不应增大预热或放宽门槛掩盖问题。修复前后这四次运行均启用相同 JFR profile 设置、`-Xms64m -Xmx512m`，不使用图片或模型替换。

## 分配定位与改动

完整 JFR 包括冷机/预热期间的正常 Vector 编译阶段分配，因此没有直接把整个录制的 Vector 分配量当作稳态回归。筛选录制末段约 1.5 秒的 `jdk.ObjectAllocationSample` 并按照第一个项目栈帧归因，主要分配集中于 `VectorPreparedConv.microtile`。正式区间线程分配计数和后续独立阶段诊断均显示 DET/REC 的大分配。

旧边界微内核维护四个 FloatVector 累加器，每个累加器在每个 tap 中分别受 validity 分支控制。在复现 workload 中，这个方法持续产生 Vector 包装对象/数组分配，而不是一次性的 Workspace 或权重打包分配。

现把边界 tile 分拆为独立像素的紧凑数组入口 `boundaryPixel`，每个入口只有一个 Vector 累加器。内部仍按原 IC/KH/KW 顺序逐项 `mul` 后 `add`，不改用 FMA，不合并浮点算术；仍先判断 padding validity 再加载权重，不能以“乘零”替代跳过无效像素，否则非有限权重会改变结果。四像素 interior 热路径、权重预算和 scratch 复用策略不变。

这是定位后修复并在本机复验的内核问题；没有声称解释所有 JDK/hardware 上的 Vector 分配，或完全解释此前 Small 偶发的 322 MiB/OCR 分配。后者需独立多轮验证。

## 验证与 CI 防回归

- Maven `verify`：179 项 Java 测试通过，无失败/错误/跳过。
- 新增宽度 1..8、无/有 bias、偏移、前后数组 guard、复用 scratch、行分片的精确 Scalar 对照；无效 padding 含 Infinity 的覆盖扩展至宽度 1..5。
- 固定 128、512-bit species 各跑 4 项 PreparedConv 正确性测试通过；256-bit 为本机默认。固定 512-bit 测试只声明正确性，不宣称本机原生支持 512-bit 或对应性能。
- 三种真实 ONNX 的 16 项 ORT graph-output 数值 Golden 通过；Small/Medium sample full OCR 各 16 行，pipeline text Golden 通过。保留原先 C 保存文本的空格/标点差异，不把它报告为完全 C parity。
- 七项原 kernel 性能检查通过。修复后本机默认 CPU 的 Full OCR 为 90.512 ms、63,424 bytes/OCR、GC=0、workspace_efficiency=1、packed_weight_bytes=2,271,584；默认 CPU 与上表 2 CPU 不用于直接速度对比。
- 保留原延迟、1 MB/OCR、GC、打包权重和 Workspace 效率门槛。CI 额外用两个新 JVM 强制 2 CPU、10 次预热/5 次计时，并分别检查 1 MB/OCR 与 GC≤1；检查 workload 参数，防止 Scalar/更多 CPU/不同 shape 等测试替代掉故障 workload。
- 4 项 Python 守卫单测覆盖原故障数值、阈值边界、缺失/不支持的指标和错误 workload。
- benchmark artifact 上传改为 `if: always()`，便于门禁失败时下载 JSON/JFR；原主要 JFR 录制保留。

本机上述真实输出执行更新后的 `scripts/check-performance-regression.py` 通过。GitHub Ubuntu Temurin 25 上的修复尚待提交、推送后由 CI 验证；本机结果不能替代三平台验收。

原始本机结果保存在被 Git 忽略的 `benchmark-results/allocation-regression-20261004/`，包括修复前/后 JFR、原七项基准 JSON、三次独立 2 CPU 完整 JSON，以及 ONNX 验证日志。未修改参考 C/C# 项目。
