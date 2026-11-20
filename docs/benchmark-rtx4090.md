# RTX 4090 benchmark handoff

The completed seed-42 sweep averaged **11.85 s** per scene for Ultrafast, **17.31 s** for Balanced and **23.24 s** for Quality. All three presets scored 246 held-out views. [RTX 4090 results](benchmarks.md#rtx-4090-presets) · [Per-scene CSV](NVIDIA_GeForce_RTX_4090_presets.csv) · [System record](NVIDIA_GeForce_RTX_4090_system.txt)

Measure Ultrafast, Balanced and Quality on the Lenovo 30HJSAME00. It has an RTX 4090 (24 GB), a 32-core Threadripper PRO 9975WX, 127.2 GB of reported RAM and Windows 11 Enterprise 10.0.26300. Use the PERF-GS checkout containing this guide. Run from the repository root on an idle GPU with the hub closed.

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

## Run and return the results

```powershell
.\scripts\running\benchmark_presets.ps1 -ExpectedGpu 'NVIDIA GeForce RTX 4090'
```

The sweep runs **27 fresh training processes** on a first pass. Each preset gets 3,000 steps at seed 42 on all nine scenes, including flowers and treehill. It uses the stored quarter-resolution images outdoors and half-resolution images indoors. Every eighth camera is held out. Each preset must score all 246 held-out views.

Report **mean training-loop seconds per scene** and **Graphdeco RGB8 PSNR, SSIM and VGG LPIPS**. The clock excludes setup, evaluation and export. Average views within each scene, then give each of the nine scenes equal weight. An interrupted run can resume with the same command on the same checkout. Record any contention or reruns alongside the results. [Full protocol](mipnerf360_3k.md#reproduce-the-benchmark)

Return these two files from the folder the command prints, `outputs/benchmarks/NVIDIA_GeForce_RTX_4090/<revision>/`. Retain the raw run reports and scored model directories on the Lenovo.

- `NVIDIA_GeForce_RTX_4090_presets.csv` — one measured row per scene and preset.
- `NVIDIA_GeForce_RTX_4090_system.txt` — workstation, driver, software versions and trainer revision.

Include the three preset means in the handoff. With Matplotlib installed, `python scripts/plots/plot_preset_results.py --presets PATH_TO_CSV` prints the table and writes a 4090-specific chart.

The README uses the verified 4090 means for its headline and three-preset table. Its Stump demo comes from a separate interactive 4090 run with the viewer and frame capture active. The 16 GB RTX 4080 Balanced result (25.97 s, 27.39 dB, 2.72–2.89 GiB of process VRAM at loop end) and the same-card peer comparison come from the earlier workstation. [RTX 4080 results](benchmarks.md#current-presets)

## Interactive Stump demo

To try the recorded settings, launch the hub with the following command, open `datasets/360_v2/stump`, then select **Start Training**.

```powershell
.\.venv\Scripts\python.exe -m trainer.frontend.app --hub --preset ultrafast --iters 3000 --downsample 4 --seed 42 --backend vulkan
```

The recording used a 1152 × 648 viewer and the default snapshot interval of 10 steps. The 12.71 s clock starts when the hub opens the dataset and stops when it reports the trained result. The optimization loop took 8.95 s with frame capture and 8.96 s in a separate run without capture. Both runs used the viewer’s 15 Hz limit during optimization.
