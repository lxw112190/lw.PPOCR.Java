# Stride-two gather 分配修复（2026-10-07）

基线 `03f0c26`。本次为 `Unreleased` 修复，未修改公开 OCR API、模型、检测阈值、
Scalar 归约顺序、CI 分配/延迟门槛或参考 C/C# 项目。

## CI 故障与本机复现

用户提供的 macOS Maven 日志中，core / ImageIO 测试通过，Vector 的
`VectorStrideTwoTileTest.hotPreferredStemHasBoundedAllocation` 失败：
100 次预热后，5 次 stem 调用共分配 **165,542,400 B**，超过原 **100,000 B** 上限。
这次不是此前 DET stem 的 5 ms 时序门禁。

旧内核使用带 indexMap 的 `FloatVector.fromArray`，其无硬件 gather 的回退路径可能
产生 Vector/数组临时对象。在同一本机旧 class、JDK 25、`-XX:UseAVX=0` 下，
preferred species 为 128-bit，同一断言复现 **551,808,000 B / 5 次**。
修复后为 **0 B / 5 次**。这是 x64 无 AVX 的回退路径验证，不是本机运行了 macOS/ARM；
没有远程 JFR，不能声称两台机器的全部分配栈均已逐一验证。

## 实现与第二次回归定位

- 所有 stride-two gather 改成两次连续读取、静态单源 shuffle 和静态 blend mask。
  输入的有效元素仍为 `offset + 2*i`，不改变浮点位模式。
- 第二次读取从 `offset + lanes - 1` 开始，最大读取下标为
  `offset + 2*lanes - 2`，恰好是最后一个有效元素。数组末端没有额外越界读取，
  不需要 masked load 或运行时 index vector。
- 同时覆盖 dense 3×3 stride-two、固定 REC 3×3 stride-two 和通用 stride-two，
  不是只改失败测试调用的 stem。
- 初次替换后，完整 Tiny LWM OCR 出现约 4.9 MB/OCR 的新分配。
  JFR 末段样本定位到 `VectorBackend.generalStrideTwo → VectorStrideTwoLoad.load → blend`。
  将通用 12/8/4/1 输出通道块的行循环提取为紧凑数组/标量入口后，恢复约 12 KB/OCR。
  IC/KH/KW 遍历、bias 初始化、padding 有效性判断、标量尾部和非融合乘加全部保留。

## 本机验证

环境：Windows 10 19045、Ryzen 7 7735H、Oracle JDK 25.0.4.1，默认 preferred256。

- Maven `verify`：**211 项** Java 测试，0 失败 / 错误 / 跳过。
- 新读取测试覆盖准确数组末端、非零 offset、正负零、NaN payload、Infinity、subnormal，
  以及未选中的奇数位 NaN；原有 Scalar 位级和真实 Tiny Full OCR Golden 保留。
- 新通用行入口测试覆盖 groups、dilation、batch、bias/null bias、数组 guard、窄宽度、
  12/8/4/1 通道块和剩余通道。默认 species 的 stem 与所有通道块各 5 次实测分配为 0 B。
- 无 AVX 的定向 Maven 测试 **36 项全部通过**，包括 preferred128 的严格分配断言
  和真实 Tiny Full OCR Golden；CI 增加 Linux x64 的同配置检查。macOS 仍运行原分配断言。
- 固定 128 / 512-bit 的各 57 项定向测试均无失败；各 2 项仅限 preferred-species
  的分配测试按原条件跳过，其余 55 项执行通过。固定512不代表本机有原生 AVX512。
- Python 性能守卫 8 项、发布布局 5 项测试通过。
- 三模型真实 ONNX 的 16 组独立 graph-output Golden 通过；Small/Medium sample
  完整 OCR 各 16 行，pipeline 文本 Golden 通过。既有 C 文本空格/标点差异仍在，
  不把 `c_reference_text_parity=false` 报告成完全 C parity。

原性能门禁的新进程结果如下，全部原阈值不变并通过：

| workload | median ms | 原上限 ms |
| --- | ---: | ---: |
| REC projection MatMul | 1.021 | <5 |
| REC stride-two | 2.317 | <10 |
| DET stem | 0.898 | <5 |
| DET downsample | 2.644 | <12 |
| DET downsample960 | 13.221 | <35 |
| DET stride-one | 3.892 | <15 |
| DET stride-one960 | 15.279 | <60 |

Tiny LWM / DET320 / Vector AUTO：10 次预热、5 次计时。
8 CPU mean=91.861 ms、35,302 B/OCR；两个独立 2 CPU JVM 为 12,190 / 11,800 B/OCR。
三次计时阶段 GC 均为 0，原 1,000,000 B/OCR、GC、Workspace 和权重预算守卫全部通过。
这些结果不能与不同机器的 CI 时间直接计算端到端提速比例。

额外 ONNX / DET960 探针：8 CPU、相同 sample、20 次预热 / 5 次计时，开启既有 opt-in FMA：

| 模型 | mean ms | 分配 B/OCR | 计时阶段 GC |
| --- | ---: | ---: | ---: |
| Tiny | 116.181 | 53,624 | 0 |
| Small | 418.771 | 81,853 | 0 |
| Medium | 1868.849 | 109,582 | 0 |

这三个探针不是原 Tiny LWM 门禁，也不代表所有 JVM 上 Small/Medium 的大分配都已根除。
此前时序修复文档中的 22.9 / 60.9 MB 是旧进程的真实结果，保留为历史证据。

原始 JSON、日志及 JFR 在被忽略的 `benchmark-results/gather-fix-*` 和
`benchmark-results/spatial-conv-20261007-regression/`，不提交机器生成文件。
本机通过不能替代 Ubuntu / Windows / macOS 的远程验收；需提交、推送后确认新 main CI。

## 发布边界

不要移动或强制覆盖远程 `v0.3.1`（指向 `d915a29`）。重跑该旧标签不会带入这次修复。
当前 POM 仍为 0.3.1；先确认 main CI，再单独准备后续补丁版（例如 0.3.2），
同步版本与发布说明后再创建新标签。
