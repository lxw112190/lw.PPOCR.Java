# Stride-two Vector 返回边界修复（2026-10-07）

基线 `0bfab24`。此修复不移动标签、不改变公开 API、模型、浮点算术、padding、
工作负载或性能/分配门槛。参考 C/C# 仓库仍为只读。

## 新的 macOS 失败

用户提供 Run `37622805080` 的 macOS 日志：原 stem 和通用 OC=1 的 5 次调用均为 0 B；
`hotPreferredGenericRowsHaveBoundedAllocation` 在 **OC=4** 时分配 **1,920,960 B**，
超过 **100,000 B / 5 次**。该运行已有新读取类，不能归因于未包含首次修复的旧标签。
Scalar 数值对照和真实 Tiny OCR Golden 在该日志中通过。

fixture 是 IC3、32×160、OC4、3×3、stride2、dilation2、padding2、输出16×80。
preferred128 有4条float lane，每个 kernel tap 的合法向量块计数合计为每次8,004次读取，
5次共40,020次。故失败值恰好是：

```text
40,020 × 48 B = 1,920,960 B
```

## 控制实验，不靠增加预热掩盖分配

本机 Windows 10 / Ryzen7 7735H / Oracle JDK25.0.4.1，禁用 AVX、2 CPU、preferred128。
保持 `0bfab24` 的读取辅助方法不内联：

```text
-XX:UseAVX=0 -XX:ActiveProcessorCount=2
-XX:CompileCommand=dontinline,io.github.lxw112190.ppocr.vector.VectorStrideTwoLoad::load
```

OC=4 在第100次调用之后，5次分别分配384,192 B，总计 **1,920,960 B**；
在固定第1,100次调用之后，仍为完全相同的总数。不是只在一个采样里出现一次尖峰。
OC=1/8/12 同量；OC=16/25 因多次通道块读取为2倍/3倍。
基线 JFR 包含 `blend → VectorStrideTwoLoad.load → strideTwoRow*` 的float数组分配栈。

这是辅助方法无法内联时的受控机制复现。它与 macOS 的计数完全吻合，但没有原失败
进程的编译日志，不能宣称已经知道 macOS 当时拒绝内联的具体理由或所有分配栈。
无 AVX 的 x64也不是 ARM/macOS 的替代验收。

## 实现

- 在 dense、REC 和通用12/8/4/1通道行循环中直接进行连续读取、shuffle与blend。
  从用户辅助方法返回 FloatVector 的边界从所有这六个热调用点移除。
- cached shuffle/mask仍共享。`VectorStrideTwoLoad.load` 仅保留为位模式测试参考，
  不再是生产热路径；没有新增 gather、临时数组、scratch或每调用权重打包。
- 重叠读取的最大下标仍恰好为 `offset + 2*lanes - 2`，原数组末端测试保留。
  bias、IC/KH/KW 顺序、非融合乘加、dilation、group和标量尾部不变。
- 正式断言仍是100次预热、5次计数、总分配≤100,000 B，不按结果追加预热或挑选样本。
  新增逐调用分配、实际预热耗时和 JVM/CPU/arch 信息，打印在计数之后。

## CI 诊断和防回归

1. 原三平台 `verify` 记录 JVM 编译 XML（含原失败进程本身），存于忽略的模块 `target/`。
2. 三平台新增非内联边界检查：禁止读取辅助方法、通用行入口、dense tile和REC pixel
   被内联；数组/标量入口内部的 Vector 操作仍须通过原数值和严格分配断言。
   不强制 C2、不启用同步编译、不改变主性能进程的 JVM 配置。
3. 原 Linux x64无 AVX检查、128/512固定species和三模型ONNX验证保留。
4. 失败后独立诊断探针记录所有OC的固定100 / 1,100次先前调用预算、逐调用分配，
   JIT XML与JFR。第二组诊断不替换第一组正式断言，不让失败的 job 变绿。
5. `test-jit-evidence-<os>` 始终上传 Surefire报告、dump和编译XML；探针若运行，
   还包含 `probe.txt / probe-jit.xml / probe.jfr`，保留7天。

## 本机已完成验证

- Maven verify：211项Java测试，0失败/错误/跳过。
- 禁止上述所有边界内联：native preferred256与无AVX+2CPU preferred128的
  各36项测试通过。stem与OC=1/4/8/12/16/25的5次实测分配均为0 B。
- 同一非内联控制探针：新代码OC=4在100 / 1,100次先前调用后的分配均为0 B。
- 固定128 / 512-bit各57项定向测试无失败；各2项仅限preferred的分配测试按原
  条件跳过，其余55项执行通过。不宣称本机原生支持512-bit。
- 新CI日志配置及quiet非内联命令守卫通过；模块target实际生成JVM编译XML。
  严格Vector与opt-in FMA各16组真实ONNX独立数值Golden通过，Small/Medium
  完整OCR各16行、pipeline文本Golden通过；既有C参考文本空格/标点差异仍保留。
- Python性能守卫8项、发布布局5项通过；CI YAML解析通过。
- 原七项kernel延迟与Tiny LWM完整OCR守卫通过，分配、GC、Workspace和权重上限未改变。
  8 CPU mean=97.221 ms、35,315 B/OCR；两个独立2CPU进程为12,171 / 12,184 B/OCR，
  三个计时阶段GC均为0。门禁原始结果见 `generic-boundary-final-gates.log`。

额外ONNX探针不用于声称此次端到端提速：本次最初Small/Medium耗时明显高于早先
另一轮进程，必须用当前同机旧类对照，不应拿旧文档的418/1869 ms直接算回归比例。
忽略目录中仅将这四个生产类按HEAD源码重新编译（源码逐一核对）并置于对照classpath
前端，其余模型、class和参数相同。相邻新JVM、8CPU、DET960、20次预热/5次计时、
同一500×500 sample、开启既有FMA选项的结果：

| 模型 / 进程 | mean ms | median ms | 分配 B/OCR |
| --- | ---: | ---: | ---: |
| Small，候选首轮 | 916.906 | 914.961 | 81,853 |
| Small，HEAD对照 | 983.632 | 974.760 | 81,859 |
| Small，候选后续 | 454.549 | 453.632 | 81,840 |
| Medium，候选首轮 | 2821.661 | 2812.889 | 109,602 |
| Medium，HEAD对照 | 1919.259 | 1897.541 | 109,595 |
| Medium，候选后续 | 1909.109 | 1895.422 | 109,595 |

计时GC均为0。Medium相邻对照基本一致；Small同一候选前后进程差异很大，
不能把一个旧慢进程对一个新快进程报告为可靠的50%以上提速，也不能把首次候选
对更早的另一轮快进程直接报告为代码引入的2倍回归。本次不调整Small/Medium
调度或门禁来挑选成绩，不宣称解决所有跨进程吞吐波动；此修复的验收重点是
消除可受控复现的Vector返回边界分配，并保留全部原有正确性/性能守卫。

生成的JSON、基线类、日志和JFR位于忽略的 `benchmark-results/generic-boundary-*`，
不进入提交。需推送后确认新的macOS CI；本机通过不等于已在远程ARM runner验收。
