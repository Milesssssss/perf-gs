# Copyright (c) 2026 Electronic Arts Inc. All rights reserved.
param([Parameter(Mandatory)][string]$ExpectedGpu, [switch]$QualityPlusOnly)

$ErrorActionPreference = 'Stop'
function Invoke-Checked($program, [string[]]$arguments) {
    & $program @arguments
    if ($LASTEXITCODE -ne 0) { throw "$program $($arguments[0]) failed (exit $LASTEXITCODE)" }
}

$stage = $null
$lock = $null
Push-Location (Join-Path $PSScriptRoot '../..')
try {
    $probe = 'from trainer.backend.compute.device import create_device; d = create_device(); print(d.info.adapter_name); d.close()'
    $gpu = (Invoke-Checked python @('-c', $probe) | Select-Object -Last 1).Trim()
    if ($gpu -cne $ExpectedGpu) { throw "Expected $ExpectedGpu, found $gpu on the training Vulkan device." }
    $gpuTag = $gpu -replace '[^A-Za-z0-9]+', '_'
    $revision = (Invoke-Checked git @('rev-parse', 'HEAD')).Trim()
    # The next workstation may receive the trainer as an uncommitted patch.
    $fingerprint = @'
import hashlib, pathlib, subprocess
import trainer
if pathlib.Path(trainer.__file__).resolve() != pathlib.Path('src/trainer/__init__.py').resolve():
    raise SystemExit('Run python -m pip install -e . in this checkout before benchmarking.')
files = subprocess.check_output(['git', 'ls-files', '-co', '--exclude-standard', '-z', '--', 'src/trainer', 'shaders'])
digest = hashlib.sha256()
for name in sorted(filter(None, files.split(bytes(1)))):
    path = pathlib.Path(name.decode('utf-8'))
    if path.is_file():
        digest.update(name + bytes(1) + hashlib.sha256(path.read_bytes()).digest())
print(digest.hexdigest())
'@
    $sourceHash = (Invoke-Checked python @('-c', $fingerprint)).Trim()
    $short = $revision.Substring(0, 12) + '-' + $sourceHash.Substring(0, 12)
    $dest = (New-Item -ItemType Directory -Force "outputs/benchmarks/$gpuTag/$short").FullName
    $suite = if ($QualityPlusOnly) { 'qualityplus' } else { 'presets' }
    $csv = Join-Path $dest "${gpuTag}_${suite}.csv"
    $system = Join-Path $dest "${gpuTag}_system.txt"
    $lock = [IO.File]::Open((Join-Path $dest 'benchmark.lock'), 'OpenOrCreate', 'ReadWrite', 'None')
    if (Test-Path -LiteralPath $csv) {
        if (!(Test-Path -LiteralPath $system)) { throw "Missing system record for $csv" }
        Write-Host "Benchmark results:`n$csv`n$system"
        return
    }

    $computer = Get-CimInstance Win32_ComputerSystem
    $cpu = Get-CimInstance Win32_Processor
    $os = Get-CimInstance Win32_OperatingSystem
    $pythonInfo = @'
import sys, slangpy, torch, torchvision
print('Python', sys.version)
for package in (slangpy, torch, torchvision):
    print(package.__name__, package.__version__)
if not torch.cuda.is_available():
    print('Graphdeco scoring device: CPU')
'@
    # Keep the NVIDIA system record compatible with earlier resumable runs.
    if ($gpu -like 'NVIDIA*' -and (Get-Command nvidia-smi -ErrorAction SilentlyContinue)) {
        $gpus = Invoke-Checked nvidia-smi @('--query-gpu=name,driver_version,memory.total', '--format=csv')
        $gpuRecord = $gpus | Where-Object { $_ -eq $gpus[0] -or ($_ -split ',')[0] -ceq $gpu }
    } else {
        $adapter = Get-CimInstance Win32_VideoController | Where-Object Name -eq $gpu | Select-Object -First 1
        if (!$adapter) { throw "Cannot find the Windows display driver for Vulkan device $gpu." }
        $gpuRecord = @("GPU: $gpu", "Windows display driver: $($adapter.DriverVersion)")
        # Win32_VideoController.AdapterRAM is only 32 bits, even on discrete cards over 4 GiB.
        $driver = (Get-ItemProperty -LiteralPath "HKLM:\SYSTEM\CurrentControlSet\Enum\$($adapter.PNPDeviceID)").Driver
        $bytes = Get-ItemPropertyValue -LiteralPath "HKLM:\SYSTEM\CurrentControlSet\Control\Class\$driver" `
            -Name 'HardwareInformation.qwMemorySize' -ErrorAction SilentlyContinue
        if ($bytes) { $gpuRecord += "Dedicated video memory (MiB): $([int64]($bytes / 1MB))" }
    }
    $record = @(
        "Trainer commit: $revision"
        "Trainer source files SHA-256: $sourceHash"
        "System: $($computer.Manufacturer) $($computer.Model)"
        "CPU: $($cpu.Name) ($($cpu.NumberOfCores) cores)"
        "OS: $($os.Caption) $($os.Version)"
        "Visible system RAM (KiB): $($os.TotalVisibleMemorySize)"
        $gpuRecord
        Invoke-Checked python @('-c', $pythonInfo)
    ) -join "`n"
    if (Test-Path -LiteralPath $system) {
        if ([IO.File]::ReadAllText($system) -cne $record) { throw "This partial benchmark has a different system record: $system" }
    } else {
        [IO.File]::WriteAllText($system, $record)
    }

    $reference = 'outputs/graphdeco_reference'
    if (!(Test-Path -LiteralPath $reference)) {
        Invoke-Checked git @('clone', 'https://github.com/graphdeco-inria/gaussian-splatting.git', $reference)
    }
    Invoke-Checked git @('-C', $reference, 'checkout', '--detach', '54c035f7834b564019656c3e3fcc3646292f727d')
    $scenes = 'bicycle', 'flowers', 'garden', 'stump', 'treehill', 'bonsai', 'counter', 'kitchen', 'room'
    $presets = if ($QualityPlusOnly) { 'quality+' } else { 'ultrafast', 'balanced', 'quality' }
    $models = foreach ($preset in $presets) {
        $steps = [int](Invoke-Checked python @('-c', 'import sys; from trainer.frontend.cli import parse_args; print(parse_args(sys.argv[1:]).iters)', '--preset', $preset))
        $name = "mip360_${gpuTag}_${short}_$preset" -replace '-', '_'
        Invoke-Checked python @('scripts/running/eval_sweep.py', '--axis', 'seed', '--values', '42', '--iters', "$steps",
            '--name', $name, '--no-plot', '--extra', '--preset', $preset, '--backend', 'vulkan', '--eval-every', '0',
            '--preview-every', '0', '--no-write-ground-truths', '--run-report', '--save-test-renders') | Out-Host
        $runs = Get-Content -Raw "outputs/eval_sweep/$name/results.json" | ConvertFrom-Json
        if ($runs.Count -ne 9 -or (Compare-Object $scenes $runs.scene) -or
            (!$QualityPlusOnly -and ($runs | Where-Object { $_.iters -ne $steps }))) {
            throw "Incomplete $preset sweep. Fix the failed scenes and rerun this command."
        }
        foreach ($scene in $scenes) {
            $run = "sweep_${name}_${scene}_42"
            $model = "outputs/graphdeco_input/$run"
            $report = Get-Content -Raw "outputs/train/$run/run_report.json" | ConvertFrom-Json
            if ($report.device.adapter_name -cne $gpu) { throw "$run used $($report.device.adapter_name), expected $gpu" }
            $actual = [int]$report.iters_completed
            if ($report.iters_requested -ne $steps -or $actual -le 0 -or
                ($QualityPlusOnly -and $report.stop_reason -cne 'plateau') -or
                (!$QualityPlusOnly -and $actual -ne $steps)) { throw "Incomplete $preset run: $run" }
            if (!(Test-Path -LiteralPath "$model/test/ours_$actual" -PathType Container)) {
                Invoke-Checked python @('scripts/running/prepare_graphdeco.py', "outputs/train/$run", $model) | Out-Host
            }
            $model
        }
    }
    $stage = Join-Path $dest ("$suite-" + [guid]::NewGuid() + '.tmp')
    $scoreArgs = @('scripts/running/score_graphdeco.py', '--graphdeco', $reference,
        '--expect-gpu', $gpu, '--preset-csv', $stage)
    if ($QualityPlusOnly) { $scoreArgs += '--quality-plus-only' }
    Invoke-Checked python ($scoreArgs + $models)
    [IO.File]::Move($stage, $csv)
    Write-Host "Benchmark results:`n$csv`n$system"
} finally {
    if ($stage -and (Test-Path -LiteralPath $stage)) { Remove-Item -LiteralPath $stage }
    if ($lock) { $lock.Dispose() }
    Pop-Location
}
