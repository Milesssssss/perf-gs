# Timing for the NEW blend_data forwardRasterize (thread-per-pixel, warp OFF),
# across all 9 Mip-NeRF 360 scenes. Run dirs: timing_<scene>_blendfwd.
# Compare against timing_<scene>_threadfwd (the SAME kernel BEFORE the change,
# when it gathered the fat Gaussian3D + Projected2DGaussian) via
# plot_blendfwd_timing_compare.py. Identical flags, seed 0, flat backward, so
# the ONLY difference vs threadfwd is the forwardRasterize gather (fat -> 36 B
# BlendData). Idempotent: skips runs whose test_psnr.json already exists.
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
    $runName = "timing_$($s.name)_blendfwd"
    if (Test-Path "outputs/train/$runName/test_psnr.json") {
        Write-Host "==== SKIP $runName (already complete) ===="
        continue
    }
    Write-Host "==================== [$(Get-Date -Format HH:mm:ss)] $runName ===================="
    # No --warp-per-pixel-forward: exercises the (now blend_data) forwardRasterize.
    & $py scripts/v2_train.py --scene $s.path --iters 3000 `
        --structure-densify --mip-splatting --mip-kernel-size 0.1 `
        --sample-mode sadgs --preview-every 0 --eval-every 0 `
        --final-test-psnr `
        --no-viewer --seed 0 --run-name $runName 2>&1 |
        Select-String -Pattern "loop done|final-test-psnr|Error|error|Traceback" |
        ForEach-Object { $_.Line }
}
Write-Host "==================== ALL BLEND-FWD RUNS COMPLETE ===================="
