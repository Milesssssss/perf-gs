# Timing + test-PSNR of the RESTORED flat-CSV runtime state (fat forwardRasterize,
# commit 68a5d7f, blend/warp WIP stashed). Default CLI = flat backward on, warp
# off — matches the `flat` column of timing_flat_vs_baseline.csv. Run dirs:
# timing_<scene>_restored. Idempotent on test_psnr.json.
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
    $runName = "timing_$($s.name)_restored"
    if (Test-Path "outputs/train/$runName/test_psnr.json") {
        Write-Host "==== SKIP $runName (already complete) ===="
        continue
    }
    Write-Host "==================== [$(Get-Date -Format HH:mm:ss)] $runName ===================="
    & $py scripts/running/v2_train.py --scene $s.path --iters 3000 `
        --densify-mode structure_aware --mip-splatting --mip-kernel-size 0.1 `
        --sample-mode sadgs --preview-every 0 --eval-every 0 `
        --final-test-psnr `
        --no-viewer --seed 0 --run-name $runName 2>&1 |
        Select-String -Pattern "loop done|final-test-psnr|Error|error|Traceback" |
        ForEach-Object { $_.Line }
}
Write-Host "==================== ALL RESTORED TIMING RUNS COMPLETE ===================="
