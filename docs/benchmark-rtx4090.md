# Reproduce the RTX 4090 benchmark

The original 3,000-step sweep averaged **11.85 s** per scene for Ultrafast, **17.31 s** for Balanced and **23.24 s** for the previous Quality recipe. The command below measures the current presets. Keep the original CSV alongside the new sweep so the two Quality versions can be compared. [Original RTX 4090 results](benchmarks.md#rtx-4090-presets) · [Per-scene CSV](NVIDIA_GeForce_RTX_4090_presets.csv) · [System record](NVIDIA_GeForce_RTX_4090_system.txt)

Run the commands from the repository root with the hub closed and the GPU idle. The published sweep used an RTX 4090 (24 GB), a 32-core Threadripper PRO 9975WX and Windows 11. [Hardware and software versions](NVIDIA_GeForce_RTX_4090_system.txt)

## Set up once

Use PowerShell with Git and Python 3.11. The CUDA PyTorch package runs the independent Graphdeco evaluator. PERF-GS training uses Vulkan. Ultrafast requires NVIDIA’s separately licensed research masks. [PyTorch builds](https://pytorch.org/get-started/previous-versions/#v2100) · [Mask terms](stbn_pixel_sampling.md)

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e . slangpy==0.42.0 tqdm
python -m pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu128
python scripts/data/download_mipnerf360.py --set all
python scripts/data/download_stbn.py
```

## Run the benchmark

```powershell
.\scripts\running\benchmark_presets.ps1 -ExpectedGpu 'NVIDIA GeForce RTX 4090'
```

The sweep runs **27 fresh training processes** on a first pass. It reads the default step count from each preset. Expect **3,000** for Ultrafast and Balanced and **3,500** for Quality. Each preset uses seed 42 on all nine scenes, including flowers and treehill. It uses the stored quarter-resolution images outdoors and half-resolution images indoors. Every eighth camera is held out. Each preset must score all 246 held-out views.

Report **mean training-loop seconds per scene** and **Graphdeco RGB8 PSNR, SSIM and VGG LPIPS**. The clock excludes setup, evaluation and export. Average views within each scene, then give each of the nine scenes equal weight. An interrupted run can resume with the same command on the same checkout. Record any contention or reruns alongside the results. [Full protocol](mipnerf360_3k.md#reproduce-the-benchmark)

The command writes two files under `outputs/benchmarks/NVIDIA_GeForce_RTX_4090/<revision>-<source hash>/`. The source hash includes local trainer and shader edits, so a patched checkout gets a separate run directory.

- `NVIDIA_GeForce_RTX_4090_presets.csv` — one measured row per scene and preset.
- `NVIDIA_GeForce_RTX_4090_system.txt` — workstation, driver, software versions and trainer revision.

With Matplotlib installed, `python scripts/plots/plot_preset_results.py --presets PATH_TO_CSV` prints the table and writes a 4090-specific chart.

For the handoff, keep the new CSV and system record together. Compare Quality against its earlier 4090 row on training-loop time, PSNR, SSIM, VGG LPIPS, final splats and loop-end VRAM. The Stump demo is a separate interactive Ultrafast run. [RTX 4080 comparison](benchmarks.md#current-presets)

## Interactive Stump demo

To try the recorded settings, launch the hub with the following command, open `datasets/360_v2/stump`, then select **Start Training**.

```powershell
.\.venv\Scripts\python.exe -m trainer.frontend.app --hub --preset ultrafast --iters 3000 --downsample 4 --seed 42 --backend vulkan
```

The recording used a 1152 × 648 viewer and the default snapshot interval of 10 steps. The published clip starts at the beginning of optimization. Its training timer stops at 9.21 s. The timing strip also accounts for the 3.86 s before playback (0.38 s loading Stump + 0.53 s waiting for Start + 2.95 s preparing training). The hub reported the trained result 13.08 s after scene open. A separate run without capture took 8.96 s of optimization. Both runs used the viewer’s 15 Hz limit during optimization. The final 42° orbit was rendered from the exported model at 50 fps and takes 5.8 seconds.
