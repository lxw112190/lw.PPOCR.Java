param(
    [string]$CSharpSource='E:\My-Code\PPOCR\lw.PPOCR.C\build\SimdPaddleOCR-ci',
    [string]$JavaHome='C:\Program Files\Java\jdk-25.0.4.1',
    [string]$Maven='D:\Program Files\Apache\apache-maven-3.6.2\bin\mvn.cmd',
    [string]$Dotnet='C:\Program Files\dotnet\dotnet.exe',
    [string]$ModelsRoot='build-local-data/onnx-models',
    [string]$Image='lw-ppocr-core/src/test/resources/golden/ocr/sample.jpg',
    [ValidateSet(1,2,4,8)][int]$Cpu=8,
    [ValidateRange(1,100)][int]$Warmup=5,
    [ValidateRange(1,1000)][int]$Iterations=10,
    [ValidateRange(1,10)][int]$Replicas=1,
    [ValidateSet('tiny','small','medium')][string[]]$Variants=@('tiny','small','medium'),
    [ValidateSet('default','fma')][string[]]$JavaModes=@('default','fma'),
    [switch]$SmallChannelFma,
    [switch]$SkipJavaBuild
)
# Windows / PowerShell 7. Runs sequentially; no benchmark runs during compilation.
$ErrorActionPreference='Stop'
$repo=(Resolve-Path (Join-Path $PSScriptRoot '..')).Path
Push-Location $repo
$savedJavaHome=$env:JAVA_HOME
$savedCpu=$env:DOTNET_PROCESSOR_COUNT
$savedHeap=$env:DOTNET_GCHeapHardLimit
try {
    if($JavaModes.Count -eq 0) { throw 'Choose at least one Java mode' }
    $models=(Resolve-Path -LiteralPath $ModelsRoot).Path
    $imagePath=(Resolve-Path -LiteralPath $Image).Path
    $java=Join-Path $JavaHome 'bin/java.exe'
    $reference='3e4192f2ec03b84701d3c5c1658e277fdaa7aa12'
    $manifest=Get-Content release/ppocrv6-onnx-manifest.json -Raw | ConvertFrom-Json
    foreach($asset in $manifest.assets) {
        if((Get-FileHash -LiteralPath (Join-Path $models $asset.path)).Hash -ne $asset.sha256) {
            throw "Model/dictionary hash mismatch: $($asset.path)"
        }
    }
    $output=Join-Path $repo ('benchmark-results/csharp-comparison-'+(Get-Date -Format 'yyyyMMdd-HHmmss-fff'))
    New-Item -ItemType Directory -Path $output | Out-Null
    # Export only committed files to an isolated directory, never build the user's reference checkout.
    $snapshot=Join-Path $repo ('build-local-data/csharp-comparison-'+[Guid]::NewGuid().ToString('N'))
    New-Item -ItemType Directory -Path $snapshot | Out-Null
    & git -C $CSharpSource archive --format=tar "--output=$snapshot/source.tar" $reference
    if($LASTEXITCODE -ne 0) { throw 'Cannot export pinned C# commit' }
    & tar -xf "$snapshot/source.tar" -C $snapshot
    if($LASTEXITCODE -ne 0) { throw 'Cannot extract C# snapshot' }
    $env:JAVA_HOME=$JavaHome
    if(!$SkipJavaBuild) {
        & $Maven --batch-mode --no-transfer-progress '-Dmaven.repo.local=.maven-repository' verify
        if($LASTEXITCODE -ne 0) { throw 'Java verification failed' }
    }
    & $Dotnet build scripts/csharp-comparison/Comparison.csproj -c Release "-p:SimdPaddleOcrRoot=$snapshot" --nologo
    if($LASTEXITCODE -ne 0) { throw 'C# build failed' }
    $cp='lw-ppocr-core/target/classes;lw-ppocr-imageio/target/classes;lw-ppocr-vector/target/classes;lw-ppocr-benchmark/target/classes'
    $main='io.github.lxw112190.ppocr.benchmark.CrossRuntimeComparisonMain'
    $bgr=Join-Path $output 'sample.bgr'
    $pixelJson=& $java --add-modules jdk.incubator.vector -cp $cp $main export-image $imagePath $bgr
    if($LASTEXITCODE -ne 0) { throw 'Image export failed' }
    $pixel=$pixelJson | ConvertFrom-Json
    [ordered]@{
        java_commit=(& git rev-parse HEAD).Trim(); java_status=@(& git status --short)
        csharp_commit=$reference; csharp_snapshot=$snapshot; measured_at=(Get-Date).ToString('o')
        cpu_budget=$Cpu; cpu=Get-CimInstance Win32_Processor | Select-Object Name,NumberOfCores,NumberOfLogicalProcessors
        java_modes=$JavaModes
        java_small_channel_fma=$SmallChannelFma.IsPresent
        java_compiled_classes=@(foreach($path in @(
            'lw-ppocr-vector/target/classes/io/github/lxw112190/ppocr/vector/VectorLargePointwise.class',
            'lw-ppocr-vector/target/classes/io/github/lxw112190/ppocr/vector/VectorBackend.class',
            'lw-ppocr-vector/target/classes/io/github/lxw112190/ppocr/vector/VectorConvTranspose2x2.class',
            'lw-ppocr-vector/target/classes/io/github/lxw112190/ppocr/vector/VectorWideConv.class',
            'lw-ppocr-vector/target/classes/io/github/lxw112190/ppocr/vector/VectorDepthwiseRegister.class',
            'lw-ppocr-core/target/classes/io/github/lxw112190/ppocr/runtime/ParallelKernels.class',
            'lw-ppocr-benchmark/target/classes/io/github/lxw112190/ppocr/benchmark/CrossRuntimeComparisonMain.class')) {
            [ordered]@{path=$path;sha256=(Get-FileHash -LiteralPath $path).Hash}
        })
        os=Get-CimInstance Win32_OperatingSystem | Select-Object Caption,Version,BuildNumber
        physical_memory_bytes=(Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory
        power_scheme=@(& powercfg /getactivescheme)
        java_version=@(& $java -version 2>&1 | ForEach-Object {"$_"}); dotnet_version=@(& $Dotnet --version)
        image_sha256=(Get-FileHash -LiteralPath $imagePath).Hash; pixels=$pixel; assets=$manifest.assets
        contract='CPU FP32; DET960 .3/.6/1.6 no dilation; CLS .9; REC buckets 192/320/480/640/960; independent processes'
        memory_method='2GiB heap caps; post-GC managed heap is runtime-specific; 50ms polling of lifetime OS peak working set includes loading/warmup/reporting'
        scope='Timed OCR excludes image decode, model loading, result validation and reporting. CPU budget is not processor affinity. No CER ground truth.'
    } | ConvertTo-Json -Depth 10 | Set-Content (Join-Path $output 'environment.json') -Encoding utf8NoBOM
    $env:DOTNET_PROCESSOR_COUNT="$Cpu"
    $env:DOTNET_GCHeapHardLimit='80000000' # CLR parses this limit as hexadecimal (2GiB).
    $dll=Join-Path $repo 'scripts/csharp-comparison/bin/Release/net10.0/LwPpocrComparison.dll'
    foreach($replica in 1..$Replicas) {
        foreach($variant in $Variants) {
            $order=if($replica%2) { @('java','java-fma','csharp') } else { @('csharp','java-fma','java') }
            $order=@($order | Where-Object { $_ -eq 'csharp' -or
                ($_ -eq 'java' -and $JavaModes -contains 'default') -or
                ($_ -eq 'java-fma' -and $JavaModes -contains 'fma') })
            foreach($runtime in $order) {
                $name="$variant-$runtime-r$replica"
                $stdout=Join-Path $output "$name.json"
                $stderr=Join-Path $output "$name.stderr.log"
                $common=@("`"$models`"",$variant,"`"$bgr`"","$($pixel.width)","$($pixel.height)","$Warmup","$Iterations")
                if($runtime -eq 'csharp') { $exe=$Dotnet; $arguments=@("`"$dll`"")+$common }
                else {
                    $exe=$java
                    $arguments=@("-XX:ActiveProcessorCount=$Cpu",'-Xms64m','-Xmx2g',"-Dlwppocr.vectorFma=$($runtime -eq 'java-fma')",
                        "-Dlwppocr.smallFmaPointwise=$($SmallChannelFma -and $runtime -eq 'java-fma')",'--add-modules','jdk.incubator.vector','-cp',"`"$cp`"",$main,'run')+$common
                }
                Write-Output "Running $name"
                $process=Start-Process -FilePath $exe -ArgumentList $arguments -WindowStyle Hidden -PassThru -RedirectStandardOutput $stdout -RedirectStandardError $stderr
                $peak=0L; $watch=[Diagnostics.Stopwatch]::StartNew(); $next=15
                while(!$process.HasExited) {
                    $process.Refresh()
                    if(!$process.HasExited) { $peak=[Math]::Max($peak,$process.PeakWorkingSet64) }
                    if($watch.Elapsed.TotalSeconds -ge $next) { Write-Output "${name}: running $([Math]::Round($watch.Elapsed.TotalSeconds))s"; $next+=15 }
                    Start-Sleep -Milliseconds 50
                }
                $process.WaitForExit()
                if($process.ExitCode -ne 0) { throw (Get-Content $stderr -Raw) }
                $result=Get-Content $stdout -Raw | ConvertFrom-Json
                $result | Add-Member process_os_peak_bytes $peak
                $result | Add-Member replica $replica
                $result | ConvertTo-Json -Depth 10 | Set-Content $stdout -Encoding utf8NoBOM
                Write-Output "$name mean=$($result.mean_ms)ms lines=$($result.lines.Count) peak=$([Math]::Round($peak/1MB,1))MiB"
            }
        }
    }
    & python scripts/summarize-csharp-comparison.py $output
    if($LASTEXITCODE -ne 0) { throw 'Comparison validation failed' }
    Write-Output "Reports: $output"
} finally {
    $env:JAVA_HOME=$savedJavaHome; $env:DOTNET_PROCESSOR_COUNT=$savedCpu; $env:DOTNET_GCHeapHardLimit=$savedHeap
    Pop-Location
}
