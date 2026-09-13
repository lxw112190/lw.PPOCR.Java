# lw.PPOCR.Java 发布清单

本文档用于维护者发布 `v0.1.x`。Tag 和已经公开的 Release 不应移动或覆盖；发现问题时
应修复后提升版本号。

## 1. 发布前检查

1. 确认当前分支为 `main`，工作区只包含计划发布的改动；
2. 确认根 POM 与全部子模块使用同一非 `SNAPSHOT` 版本；
3. 确认 `CHANGELOG.md`、双语 README、安装文档、MIT License 和第三方声明已更新；
4. 使用 JDK 25 执行：

   ```text
   mvn --batch-mode --no-transfer-progress clean verify
   ```

5. 推送发布准备提交，等待 Linux、Windows、macOS 构建测试和 Linux 性能任务成功。

## 2. 创建不可变 Tag

以 `0.1.0` 为例：

```text
git tag -a v0.1.0 -m "lw.PPOCR.Java v0.1.0"
git push origin v0.1.0
```

Tag CI 会拒绝与 POM 版本不一致的 Tag。它会重新运行测试，生成
`lw.PPOCR.Java-v0.1.0.zip` 和 `lw.PPOCR.Java-v0.1.0.zip.sha256`，解压后使用包内
JAR、模型、字典和示例图片运行一次 16 行完整 OCR。全部验证成功后，CI 会创建同名
GitHub Release、上传 ZIP 与 `.sha256`，并保留相同文件作为 Actions Artifact。

## 3. 验证候选包

从 Tag 对应的 GitHub Release 下载 ZIP 与 `.sha256`，在新的空目录中校验：

Linux/macOS：

```text
sha256sum -c lw.PPOCR.Java-v0.1.0.zip.sha256
unzip lw.PPOCR.Java-v0.1.0.zip
cd lw.PPOCR.Java-v0.1.0
sha256sum -c SHA256SUMS.txt
```

Windows PowerShell：

```powershell
$expected = (Get-Content .\lw.PPOCR.Java-v0.1.0.zip.sha256).Split()[0]
$actual = (Get-FileHash .\lw.PPOCR.Java-v0.1.0.zip -Algorithm SHA256).Hash.ToLowerInvariant()
if ($actual -ne $expected) { throw "SHA-256 mismatch" }
Expand-Archive .\lw.PPOCR.Java-v0.1.0.zip -DestinationPath .\verified
Set-Location .\verified\lw.PPOCR.Java-v0.1.0
Get-Content .\SHA256SUMS.txt | ForEach-Object {
    if ($_ -match '^([0-9a-fA-F]{64})\s+\*?(.+)$') {
        $expected = $matches[1]
        $path = $matches[2].Replace('/', '\')
        $actual = (Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash
        if ($actual -ne $expected) { throw "SHA-256 mismatch: $path" }
    }
}
```

Windows 内部文件清单使用 GNU `sha256sum` 格式，上述 PowerShell 会逐项核对其中的
全部文件，并同时检查外层 ZIP 的 SHA-256。

## 4. 确认 GitHub Release

确认不可变 Tag `v0.1.0` 对应的 GitHub Release 已发布，说明来自 `CHANGELOG.md`
对应章节，并包含 ZIP 与 `.sha256`。CI 遇到已经存在的同名 Release 会失败，不会覆盖
已发布资产；如需修复，应提升版本号并重新发布。

## 5. 发布后的开发版本

Release 发布后，将 Maven 版本提升到下一个开发版本，例如 `0.1.1-SNAPSHOT`，并在
`CHANGELOG.md` 顶部建立 `Unreleased` 小节。需要撤回时保留原 Tag 和 Release，在说明中
标记问题并发布更高版本，不能重写已公开的历史。
