# BACKWARD_PIXEL_STRIDE sweep: 1 / 2 / 4 / 8 / 16, on all 9 Mip-NeRF 360 scenes.
# Pure training-loop wall time (iter 0 -> 3000) + held-out test PSNR, identical
# flags except the backward pixel-subsampling stride (a shader define). Uses the
# production flat warp-per-bucket backward. Idempotent on test_psnr.json.
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
$strides = @(1, 2, 4, 8, 16)

foreach ($s in $scenes) {
    foreach ($st in $strides) {
        $runName = "timing_$($s.name)_s$st"
        if (Test-Path "outputs/train/$runName/test_psnr.json") {
            Write-Host "==== SKIP $runName (already complete) ===="
            continue
        }
        Write-Host "==================== [$(Get-Date -Format HH:mm:ss)] $runName ===================="
        & $py scripts/running/v2_train.py --scene $s.path --iters 3000 `
            --backward-pixel-stride $st `
            --structure-densify --mip-splatting --mip-kernel-size 0.1 `
            --sample-mode sadgs --preview-every 0 --eval-every 0 `
            --final-test-psnr --no-viewer --seed 0 --run-name $runName 2>&1 |
            Select-String -Pattern "loop done|final-test-psnr|Error|error|Traceback" |
            ForEach-Object { $_.Line }
    }
}
Write-Host "==================== STRIDE SWEEP COMPLETE ===================="
