# Fresh timing + test-PSNR pass of the CURRENT state (blend_data forwardRasterize,
# default CLI: warp OFF, flat backward on). Same flags as every other timing
# batch. Run dirs: timing_<scene>_rerun. Idempotent on test_psnr.json.
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

foreach ($s in $scenes) {
    $runName = "timing_$($s.name)_rerun"
    if (Test-Path "outputs/train/$runName/test_psnr.json") {
        Write-Host "==== SKIP $runName (already complete) ===="
        continue
    }
    Write-Host "==================== [$(Get-Date -Format HH:mm:ss)] $runName ===================="
    & $py scripts/v2_train.py --scene $s.path --iters 3000 `
        --structure-densify --mip-splatting --mip-kernel-size 0.1 `
        --sample-mode sadgs --preview-every 0 --eval-every 0 `
        --final-test-psnr `
        --no-viewer --seed 0 --run-name $runName 2>&1 |
        Select-String -Pattern "loop done|final-test-psnr|Error|error|Traceback" |
        ForEach-Object { $_.Line }
}
Write-Host "==================== ALL RERUN TIMING RUNS COMPLETE ===================="
