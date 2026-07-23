# A/B timing: per-tile vs flat warp-per-bucket backward, on all 9 Mip-NeRF 360
# scenes. Pure training-loop wall time (iter 0 -> 3000), preview/eval off, so the
# ONLY difference between the two runs of a scene is the backward kernel.
# Idempotent: skips a run whose log.jsonl already has a train_complete event.
$ErrorActionPreference = "Stop"
$py = ".venv\Scripts\python.exe"

$scenes = @(
    @{ name = "bicycle";  path = "datasets/360_v2/bicycle" },
    @{ name = "bonsai";   path = "datasets/360_v2/bonsai" },
    @{ name = "counter";  path = "datasets/360_v2/counter" },
    @{ name = "garden";   path = "datasets/360_v2/garden" },
    @{ name = "kitchen";  path = "datasets/360_v2/kitchen" },
    @{ name = "room";     path = "datasets/360_v2/room" },
    @{ name = "stump";    path = "datasets/360_v2/stump" },
    @{ name = "flowers";  path = "datasets/360_extra_scenes/flowers" },
    @{ name = "treehill"; path = "datasets/360_extra_scenes/treehill" }
)

function Has-Complete($runName) {
    # Require test_psnr.json so runs that finished WITHOUT test-PSNR re-run.
    return (Test-Path "outputs/train/$runName/test_psnr.json")
}

foreach ($s in $scenes) {
    foreach ($mode in @("pertile", "flat")) {
        $runName = "timing_$($s.name)_$mode"
        if (Has-Complete $runName) {
            Write-Host "==== SKIP $runName (already complete) ===="
            continue
        }
        # Force the backward mode EXPLICITLY (the trainer flag defaults to
        # flat=on, so pertile must pass --no-flat-bucket-backward).
        if ($mode -eq "flat") { $bwd = "--flat-bucket-backward" }
        else { $bwd = "--no-flat-bucket-backward" }
        Write-Host "==================== [$(Get-Date -Format HH:mm:ss)] $runName ===================="
        & $py scripts/running/v2_train.py --scene $s.path --iters 3000 `
            --structure-densify --mip-splatting --mip-kernel-size 0.1 `
            --sample-mode sadgs --preview-every 0 --eval-every 0 `
            --final-test-psnr `
            --no-viewer --seed 0 --run-name $runName $bwd 2>&1 |
            Select-String -Pattern "loop done|final-test-psnr|Error|error|Traceback" |
            ForEach-Object { $_.Line }
    }
}
Write-Host "==================== ALL TIMING A/B RUNS COMPLETE ===================="
