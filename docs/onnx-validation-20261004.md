# ONNX 本机验证记录（2026-10-04）

## 基线

- Java 起点：`105c5bf`，本次变更尚未提交。
- C 只读参考和下载资产锁：`d58832dcf99264b28a9472ddd5159ba1279d323a`。
- 本机 Windows，JDK 25 路径：`C:/Program Files/Java/jdk-25.0.4.1`。
- sample.jpg SHA-256：`30c417c9f758a3b62718729f5a944f7d2e10cdd2bde0e8ce6785523ddb68ffe9`。
  与本机 C 项目的 sample.jpg 字节一致。
- 模型 SHA-256：[固定清单](../release/ppocrv6-onnx-manifest.json)。
- 独立参考：ONNX Runtime 1.21.0，CPUExecutionProvider，单线程；
  它只在测试中使用，不是 Java runtime 依赖。

## 结果

全模块 Maven verify：178 项测试，0 failure、0 error、0 skip。
新增 ONNX / runtime plan 用例 17 项。原 Tiny LWM Golden 保留。

七个官方 ONNX 模型解析通过。三个 REC 验证 17/192/320/480/640/960
宽度形状，三个 DET 验证 32/320/960 正方形形状，共享 CLS 验证固定形状。
实际数值对照共 16 组：三个 REC 各 17/192/960，三个 DET 各
32×64 / 320×320，加共享 CLS。

| 模型族 | Scalar 最大绝对误差（本次数值用例） | Vector 最大绝对误差 |
|---|---:|---:|
| Tiny | 5.0831e-4 | 8.8931e-5 |
| Small | 4.6027e-4 | 1.2934e-5 |
| Medium | 2.5243e-4 | 4.1127e-6 |

全部低于 1e-3 门限，平均误差低于 1e-4。REC/CLS argmax 对齐。

Small 和 Medium 在完整 sample.jpg 上各输出 16 行。
Scalar 与 Vector 分别执行实际 OCR；导出同样裁剪后的 CLS/REC 输入，
由独立 ORT 检查全部 32 行 CTC 文本、score、分类标签与旋转。
Scalar 最大全图逐行 score 误差：Small 3.2099e-6，Medium 3.2529e-5。
Vector：Small 5.3069e-7，Medium 4.1325e-7；全部低于 1e-3。

## C 参考差异没有消除

本次不声称 C 和 Java 全链路字节级一致：

- Small 的品牌行：C 样例 `OEMODM`，Java 裁剪 + ORT 为 `OEM ODM`。
- Medium 的品牌行：C 样例为半角冒号，Java 裁剪 + ORT 为全角冒号。
- 对应 Java 管线文本 SHA-256：
  - Small：`9cd560aaff37f1013cf10ebd9c616f4e2446b800985a9d15aa408d4010cdba95`。
  - Medium：`80ee582e0d0b62b87477b83400b793c26307b16c77adbca53419845c5442b1eb`。

原 C 文本与 Java 管线 Golden 分开保存。没有删空格或归一化标点。
ORT 校验的范围是相同 Java 裁剪的 CLS/REC，不证明 DB 框、裁剪几何与 C 一致。
没有在大数据集上测 CER，也没有在本轮宣称 Small/Medium 性能优化。

## 修复和交付边界

真实 Small REC 数值对照发现原 Slice 忽略 ends，
Q/K/V 拆分时会复制输入尾部并覆盖相邻 workspace。
现改为有输出边界、预编译的 SlicePlan，加入哨兵和定长切片回归。
MatMul 扩展为矩阵批次广播；形状控制链加载时折叠。

三平台 JDK 25 CI 已配置 Scalar / Vector 真实模型与独立 ORT 检查，
但本机不能代表 Linux/macOS/GitHub runner 已验证；需提交推送后观察 CI。
原性能门禁没有放宽，现有 v0.2.1 release 不被改写。
大体积模型与生成资产仅在已忽略的 build-local-data 中，
证据输出位于已忽略的 benchmark-results。

运行方法和 ONNX 子集边界见 [ONNX 接入说明](onnx-models.md)。
