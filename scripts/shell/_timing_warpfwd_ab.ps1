# Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

# A/B timing: baseline thread-per-pixel forward vs warp-per-pixel forward, on
# all 9 Mip-NeRF 360 scenes. Pure training-loop wall time (iter 0 -> 3000),
# preview/eval off, identical flags + identical backward (flat, the default),
# so the ONLY difference between the two runs of a scene is the FORWARD
# rasterizer kernel (--warp-per-pixel-forward).
# Idempotent: skips a run whose test_psnr.json already exists.
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
    return (Test-Path "outputs/train/$runName/test_psnr.json")
}

foreach ($s in $scenes) {
    foreach ($mode in @("threadfwd", "warpfwd")) {
        $runName = "timing_$($s.name)_$mode"
        if (Has-Complete $runName) {
            Write-Host "==== SKIP $runName (already complete) ===="
            continue
        }
        # threadfwd = baseline (no flag); warpfwd = --warp-per-pixel-forward.
        if ($mode -eq "warpfwd") { $fwd = "--warp-per-pixel-forward" }
        else { $fwd = "" }
        Write-Host "==================== [$(Get-Date -Format HH:mm:ss)] $runName ===================="
        & $py scripts/running/v2_train.py --scene $s.path --iters 3000 `
            --densify-mode structure_aware --mip-splatting --mip-kernel-size 0.1 `
            --sample-mode sadgs --preview-every 0 --eval-every 0 `
            --final-test-psnr `
            --no-viewer --seed 0 --run-name $runName $fwd 2>&1 |
            Select-String -Pattern "loop done|final-test-psnr|Error|error|Traceback" |
            ForEach-Object { $_.Line }
    }
}
Write-Host "==================== ALL WARP-FWD A/B RUNS COMPLETE ===================="
