# Copyright (c) 2026 Electronic Arts Inc. All rights reserved.
param([Parameter(Mandatory)][string]$ExpectedGpu)

$ErrorActionPreference = 'Stop'
function Invoke-Checked($program, [string[]]$arguments) {
    & $program @arguments
    if ($LASTEXITCODE -ne 0) { throw "$program $($arguments[0]) failed (exit $LASTEXITCODE)" }
}

$gpu = (Invoke-Checked nvidia-smi @('-i', '0', '--query-gpu=name', '--format=csv,noheader')).Trim()
if ($gpu -cne $ExpectedGpu) { throw "Expected $ExpectedGpu, found $gpu on adapter 0." }
$gpuTag = $gpu -replace '[^A-Za-z0-9]+', '_'
$stage = $null
$lock = $null
Push-Location (Join-Path $PSScriptRoot '../..')
try {
    $revision = (Invoke-Checked git @('rev-parse', 'HEAD')).Trim()
    $short = $revision.Substring(0, 12)
    $dest = (New-Item -ItemType Directory -Force "outputs/benchmarks/$gpuTag/$short").FullName
    $csv = Join-Path $dest "${gpuTag}_presets.csv"
    $system = Join-Path $dest "${gpuTag}_system.txt"
    $lock = [IO.File]::Open((Join-Path $dest 'benchmark.lock'), 'OpenOrCreate', 'ReadWrite', 'None')
    if (Test-Path -LiteralPath $csv) {
        if (!(Test-Path -LiteralPath $system)) { throw "Missing system record for $csv" }
        Write-Host "Already complete. Return these files:`n$csv`n$system"
        return
    }

    $computer = Get-CimInstance Win32_ComputerSystem
    $cpu = Get-CimInstance Win32_Processor
    $os = Get-CimInstance Win32_OperatingSystem
    $pythonInfo = @'
import sys, slangpy, torch, torchvision
if not torch.cuda.is_available():
    raise SystemExit('Install CUDA PyTorch before benchmarking. The reference scorer requires CUDA.')
print('Python', sys.version)
for package in (slangpy, torch, torchvision):
    print(package.__name__, package.__version__)
'@
    $record = @(
        "Trainer commit: $revision"
        "System: $($computer.Manufacturer) $($computer.Model)"
        "CPU: $($cpu.Name) ($($cpu.NumberOfCores) cores)"
        "OS: $($os.Caption) $($os.Version)"
        "Visible system RAM (KiB): $($os.TotalVisibleMemorySize)"
        Invoke-Checked nvidia-smi @('-i', '0', '--query-gpu=name,driver_version,memory.total', '--format=csv')
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
    $models = foreach ($preset in 'ultrafast', 'balanced', 'quality') {
        $name = "mip360_3k_${gpuTag}_${short}_$preset"
        Invoke-Checked python @('scripts/running/eval_sweep.py', '--axis', 'seed', '--values', '42', '--iters', '3000',
            '--name', $name, '--no-plot', '--extra', '--preset', $preset, '--backend', 'vulkan', '--eval-every', '0',
            '--preview-every', '0', '--no-write-ground-truths', '--run-report', '--save-test-renders') | Out-Host
        $runs = Get-Content -Raw "outputs/eval_sweep/$name/results.json" | ConvertFrom-Json
        if ($runs.Count -ne 9 -or (Compare-Object $scenes $runs.scene) -or ($runs | Where-Object { $_.iters -ne 3000 })) {
            throw "Incomplete $preset sweep. Fix the failed scenes and rerun this command."
        }
        foreach ($scene in $scenes) {
            $run = "sweep_${name}_${scene}_42"
            $model = "outputs/graphdeco_input/$run"
            $report = Get-Content -Raw "outputs/train/$run/run_report.json" | ConvertFrom-Json
            if ($report.device.adapter_name -cne $gpu) { throw "$run used $($report.device.adapter_name), expected $gpu" }
            if (!(Test-Path -LiteralPath "$model/test/ours_3000" -PathType Container)) {
                Invoke-Checked python @('scripts/running/prepare_graphdeco.py', "outputs/train/$run", $model) | Out-Host
            }
            $model
        }
    }
    $stage = Join-Path $dest ("presets-" + [guid]::NewGuid() + '.tmp')
    Invoke-Checked python (@('scripts/running/score_graphdeco.py', '--graphdeco', $reference,
        '--expect-gpu', $gpu, '--preset-csv', $stage) + $models)
    [IO.File]::Move($stage, $csv)
    Write-Host "Complete. Return these files:`n$csv`n$system"
} finally {
    if ($stage -and (Test-Path -LiteralPath $stage)) { Remove-Item -LiteralPath $stage }
    if ($lock) { $lock.Dispose() }
    Pop-Location
}
