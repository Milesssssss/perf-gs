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
    $csv = Join-Path $dest "${gpuTag}_presets.csv"
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
if not torch.cuda.is_available():
    raise SystemExit('Install CUDA PyTorch before benchmarking. The reference scorer requires CUDA.')
print('Python', sys.version)
for package in (slangpy, torch, torchvision):
    print(package.__name__, package.__version__)
'@
    $record = @(
        "Trainer commit: $revision"
        "Trainer source files SHA-256: $sourceHash"
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
        $steps = [int](Invoke-Checked python @('-c', 'import sys; from trainer.frontend.cli import parse_args; print(parse_args(sys.argv[1:]).iters)', '--preset', $preset))
        $name = "mip360_${gpuTag}_${short}_$preset" -replace '-', '_'
        Invoke-Checked python @('scripts/running/eval_sweep.py', '--axis', 'seed', '--values', '42', '--iters', "$steps",
            '--name', $name, '--no-plot', '--extra', '--preset', $preset, '--backend', 'vulkan', '--eval-every', '0',
            '--preview-every', '0', '--no-write-ground-truths', '--run-report', '--save-test-renders') | Out-Host
        $runs = Get-Content -Raw "outputs/eval_sweep/$name/results.json" | ConvertFrom-Json
        if ($runs.Count -ne 9 -or (Compare-Object $scenes $runs.scene) -or ($runs | Where-Object { $_.iters -ne $steps })) {
            throw "Incomplete $preset sweep. Fix the failed scenes and rerun this command."
        }
        foreach ($scene in $scenes) {
            $run = "sweep_${name}_${scene}_42"
            $model = "outputs/graphdeco_input/$run"
            $report = Get-Content -Raw "outputs/train/$run/run_report.json" | ConvertFrom-Json
            if ($report.device.adapter_name -cne $gpu) { throw "$run used $($report.device.adapter_name), expected $gpu" }
            if (!(Test-Path -LiteralPath "$model/test/ours_$steps" -PathType Container)) {
                Invoke-Checked python @('scripts/running/prepare_graphdeco.py', "outputs/train/$run", $model) | Out-Host
            }
            $model
        }
    }
    $stage = Join-Path $dest ("presets-" + [guid]::NewGuid() + '.tmp')
    Invoke-Checked python (@('scripts/running/score_graphdeco.py', '--graphdeco', $reference,
        '--expect-gpu', $gpu, '--preset-csv', $stage) + $models)
    [IO.File]::Move($stage, $csv)
    Write-Host "Benchmark results:`n$csv`n$system"
} finally {
    if ($stage -and (Test-Path -LiteralPath $stage)) { Remove-Item -LiteralPath $stage }
    if ($lock) { $lock.Dispose() }
    Pop-Location
}
