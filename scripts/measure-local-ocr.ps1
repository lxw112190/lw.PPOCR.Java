param(
    [string]$JdkPath = 'C:\Program Files\Java\jdk-25.0.4.1',
    [string]$ImagePath = 'E:\My-Code\PPOCR\lw.PPOCR.C\models\ppocrv6-tiny\sample.jpg',
    [string]$Revision = 'HEAD',
    [int]$Replicas = 3,
    [switch]$WorkingTree,
    [int]$Warmup = 10,
    [int]$TimedIterations = 20,
    [int[]]$ProcessorCounts = @(1,4,8),
    [switch]$SkipComparable
)

$ErrorActionPreference = 'Stop'
if ($Replicas -lt 1) { throw 'Replicas must be positive' }
if ($Warmup -lt 1 -or $TimedIterations -lt 1) { throw 'Warmup and timed iterations must be positive' }
foreach ($count in $ProcessorCounts) { if ($count -lt 1) { throw 'Processor counts must be positive' } }
$repoRoot = Split-Path -Parent $PSScriptRoot
$java = Join-Path $JdkPath 'bin\java.exe'
$javac = Join-Path $JdkPath 'bin\javac.exe'
if (!(Test-Path -LiteralPath $ImagePath)) { throw "Missing image: $ImagePath" }
$image = (Resolve-Path -LiteralPath $ImagePath).Path
Push-Location $repoRoot
try {
    $sha = (& git rev-parse $Revision).Trim()
    if ($LASTEXITCODE -ne 0) { throw 'Cannot resolve revision' }
    $stamp = Get-Date -Format 'yyyyMMdd-HHmmss'
    $runRoot = Join-Path $repoRoot "benchmark-results\local-$stamp-$($sha.Substring(0,7))"
    $snapshot = Join-Path $runRoot 'source'
    $classes = Join-Path $runRoot 'classes'
    New-Item -ItemType Directory -Path $snapshot,$classes | Out-Null
    $archive = Join-Path $runRoot 'source.tar'
    & git archive --format=tar "--output=$archive" $sha
    if ($LASTEXITCODE -ne 0) { throw 'git archive failed' }
    & tar.exe -xf $archive -C $snapshot
    if ($LASTEXITCODE -ne 0) { throw 'Source extraction failed' }

    if ($WorkingTree) {
        # Overlay current source files into the archived snapshot, never compile
        # stale classes and never modify the working tree during a measurement.
        foreach ($module in @('lw-ppocr-core','lw-ppocr-imageio','lw-ppocr-vector','lw-ppocr-benchmark')) {
            $moduleRoot = Join-Path $repoRoot $module
            foreach ($file in (Get-ChildItem (Join-Path $moduleRoot 'src') -Filter '*.java' -Recurse)) {
                $relative = $file.FullName.Substring($repoRoot.Length + 1)
                $destination = Join-Path $snapshot $relative
                New-Item -ItemType Directory -Force (Split-Path -Parent $destination) | Out-Null
                Copy-Item -LiteralPath $file.FullName -Destination $destination -Force
            }
        }
        & git diff --binary | Set-Content -LiteralPath (Join-Path $runRoot 'working-tree.patch') -Encoding UTF8
    }

    # Compile the committed snapshot, never stale target/classes or worktree edits.
    $portableSources = Get-ChildItem (Join-Path $snapshot 'lw-ppocr-core/src/main/java'),
        (Join-Path $snapshot 'lw-ppocr-imageio/src/main/java'),
        (Join-Path $snapshot 'lw-ppocr-benchmark/src/main/java') -Filter '*.java' -Recurse |
        ForEach-Object FullName
    & $javac --release 8 -encoding UTF-8 -d $classes $portableSources
    if ($LASTEXITCODE -ne 0) { throw 'Portable source compilation failed' }
    $vectorSources = Get-ChildItem (Join-Path $snapshot 'lw-ppocr-vector/src/main/java') -Filter '*.java' -Recurse |
        ForEach-Object FullName
    & $javac --release 25 --add-modules jdk.incubator.vector -encoding UTF-8 -cp $classes -d $classes $vectorSources
    if ($LASTEXITCODE -ne 0) { throw 'Vector source compilation failed' }
    $resources = Join-Path $snapshot 'lw-ppocr-core/src/test/resources'
    $classpath = "$classes;$resources"
    $cpu = Get-CimInstance Win32_Processor | Select-Object Name,NumberOfCores,NumberOfLogicalProcessors
    $os = Get-CimInstance Win32_OperatingSystem | Select-Object Caption,Version,BuildNumber,TotalVisibleMemorySize
    $hardware = Get-CimInstance Win32_ComputerSystem | Select-Object TotalPhysicalMemory
    $vector = & $java --add-modules jdk.incubator.vector -cp $classes io.github.lxw112190.ppocr.benchmark.VectorCapabilitiesMain
    if ($LASTEXITCODE -ne 0) { throw 'Vector capability query failed' }
    $hashes = @{}
    foreach ($asset in @('det/det.lwm','cls/cls.lwm','rec/rec.lwm','rec/ppocr_keys.txt')) {
        $hashes[$asset] = (Get-FileHash -LiteralPath (Join-Path $resources "golden/$asset") -Algorithm SHA256).Hash
    }
    $environment = [ordered]@{
        measured_at = (Get-Date).ToString('o'); git_sha = $sha
        source = $(if ($WorkingTree) { 'git archive plus working-tree Java source overlay' } else { 'committed git archive' })
        worktree_status = @(& git status --short)
        cpu = $cpu; os = $os; hardware = $hardware
        power_scheme = @(& powercfg /getactivescheme)
        jdk_path = $JdkPath; java_version = @(& $java -version 2>&1 | ForEach-Object { "$_" })
        vector = ($vector | ConvertFrom-Json); image_path = $image
        image_sha256 = (Get-FileHash -LiteralPath $image -Algorithm SHA256).Hash
        golden_image_sha256 = (Get-FileHash -LiteralPath (Join-Path $resources 'golden/ocr/sample.jpg') -Algorithm SHA256).Hash
        model_sha256 = $hashes; heap_flags = @('-Xms64m','-Xmx512m')
        detector_limit = 960; backend = 'vector'; parallelism = 'auto'
        vector_bits = 'preferred'; pointwise_block = 'auto'
        memory_method = 'Windows Process.WorkingSet64 / PeakWorkingSet64; 50ms polling; entire JVM lifetime'
        replicas = $Replicas
        profiles = @("steady: warmup$Warmup timed$TimedIterations, CPU visible $ProcessorCounts, replicas set by Replicas",
            "comparable: warmup1 timed99, CPU visible 4, one replica; skipped=$SkipComparable")
    }
    $environment | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath (Join-Path $runRoot 'environment.json') -Encoding UTF8
    $results = @()
    $profiles = @()
    # Alternate processor counts between fresh JVM runs to expose drift.
    for ($replica = 1; $replica -le $Replicas; $replica++) {
        foreach ($processors in $ProcessorCounts) {
            $profiles += @{name='steady'; warmup=$Warmup; iterations=$TimedIterations; processors=$processors; replica=$replica}
        }
    }
    if (!$SkipComparable) { $profiles += @{name='comparable'; warmup=1; iterations=99; processors=4; replica=1} }
    foreach ($profile in $profiles) {
        $label = "$($profile.name)-cpu$($profile.processors)-r$($profile.replica)"
        $stdout = Join-Path $runRoot "$label-raw.json"
        $stderr = Join-Path $runRoot "$label-stderr.log"
        $javaArgs = @("-XX:ActiveProcessorCount=$($profile.processors)", '-Xms64m', '-Xmx512m',
            '-Dlwppocr.vectorBits=preferred', '-Dlwppocr.vectorPointwiseBlock=auto',
            "`"-Dlwppocr.benchmark.image=$image`"", '--add-modules', 'jdk.incubator.vector',
            '-cp', "`"$classpath`"", 'io.github.lxw112190.ppocr.benchmark.FullOcrPerformanceMain',
            "$($profile.warmup)", "$($profile.iterations)", '960', 'vector', 'auto')
        Write-Output "Running $label (warmup=$($profile.warmup), timed=$($profile.iterations))"
        $process = Start-Process -FilePath $java -ArgumentList $javaArgs -WorkingDirectory $repoRoot -WindowStyle Hidden -PassThru -RedirectStandardOutput $stdout -RedirectStandardError $stderr
        $workingSetPeak = 0L
        $osPeak = 0L
        $samples = 0
        while (!$process.HasExited) {
            $process.Refresh()
            if (!$process.HasExited) {
                $workingSetPeak = [Math]::Max($workingSetPeak, $process.WorkingSet64)
                $osPeak = [Math]::Max($osPeak, $process.PeakWorkingSet64)
                $samples++
            }
            Start-Sleep -Milliseconds 50
        }
        $process.WaitForExit()
        if ($process.ExitCode -ne 0) { throw "$label failed: $(Get-Content -LiteralPath $stderr -Raw)" }
        $json = Get-Content -LiteralPath $stdout | Where-Object { $_.StartsWith('{') } | Select-Object -Last 1 | ConvertFrom-Json
        if ($json.lines -ne 16 -or $json.image_width -ne 500 -or $json.image_height -ne 500) {
            throw "$label has unexpected sample image or line count"
        }
        $record = [ordered]@{
            profile=$profile.name; processors=$profile.processors; replica=$profile.replica
            mean_ms=$json.mean_ms; median_ms=$json.median_ms; p95_ms=$json.p95_ms
            detection_mean_ms=$json.detection_mean_ms; classification_mean_ms=$json.classification_mean_ms
            recognition_mean_ms=$json.recognition_mean_ms; lines=$json.lines
            warmup=$json.warmup; iterations=$json.iterations
            classification_parallelism=$json.classification_parallelism
            recognition_parallelism=$json.recognition_parallelism
            allocated_bytes_per_ocr=$json.allocated_bytes_per_ocr
            gc_count=$json.gc_count_delta; gc_time_ms=$json.gc_time_ms_delta
            retained_heap_bytes=$json.retained_heap_delta_bytes; workspace_bytes=$json.workspace_bytes
            packed_weight_bytes=$json.packed_weight_bytes
            projection_scratch_bytes=$json.projection_scratch_bytes
            spatial_packed_weight_bytes=$json.spatial_packed_weight_bytes
            spatial_scratch_bytes=$json.spatial_scratch_bytes
            model_load_ms=$json.model_load_ms; cold_ms=$json.cold_ms
            process_sampled_peak_bytes=$workingSetPeak; process_os_peak_bytes=$osPeak
            memory_samples=$samples; raw_file=(Split-Path -Leaf $stdout)
            command="$java $($javaArgs -join ' ')"
        }
        $results += $record
        $results | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath (Join-Path $runRoot 'results.json') -Encoding UTF8
        Write-Output "$label median=$($json.median_ms)ms p95=$($json.p95_ms)ms lines=$($json.lines) peak=$([Math]::Round($osPeak/1MB,1))MiB"
    }
    Write-Output "RESULT_DIRECTORY=$runRoot"
} finally {
    Pop-Location
}
