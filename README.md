<!-- Images in assets/ are real outputs of this repo: teaser.jpg and
     examples_strip.jpg are renders from a trained bicycle scene (Mip-NeRF 360,
     30k iters); the Results figures are generated from the benchmark runs by
     scripts/plots/plot_readme_results.py. -->

# SCATTO-GS
## Slang-Compiled Accelerated Training-Time Optimization for Gaussian Splatting
A fast and efficient Slang-based 3DGS trainer.

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![Shaders](https://img.shields.io/badge/Shaders-Slang-DE3163)
![Runtime](https://img.shields.io/badge/Runtime-SlangPy-6E44FF)
![Backend](https://img.shields.io/badge/Backend-Vulkan-A41E22?logo=vulkan&logoColor=white)
![Backend](https://img.shields.io/badge/Backend-D3D12-107C10)
![No CUDA required](https://img.shields.io/badge/CUDA-not%20required-2ea44f)
![Status](https://img.shields.io/badge/status-research-F5A623)

A 3D Gaussian Splatting trainer written in [Slang](https://shader-slang.org) and driven from Python through [SlangPy](https://slangpy.shader-slang.org).<br>
It converges in ~31s per scene at SH0 and ~42s at SH3, averaged over the full Mip-NeRF 360 dataset on an RTX 4080. The codebase has no CUDA dependencies, is directly integratable with pytorch frontend and all shaders are dispatched via SlangPy.

<p align="center">
  <img src="assets/teaser.jpg" alt="Mip-NeRF 360 bicycle scene trained by this pipeline, rendered from a held-out view" width="80%">
  <br><em>Mip-NeRF 360 bicycle, trained end to end by this pipeline and rendered from a held-out camera.</em>
</p>

By writing the kernels in Slang and having the whole system driven by slangpy, we aim to allow distributing this trainer on any system and any backend. 

**Portability first.** The rasterizer, the gradients, the optimizer, and the densification kernels are all Slang. The same source compiles to SPIR-V for Vulkan and DXIL for D3D12 today, with room for more backends as slang-rhi grows. There are no `.cu` files anywhere and nothing that assumes a specific vendor.

**Modular codebase.** Most 3DGS trainer usually live on a performance-readability spectrum, we aimed to have both, while also having flexible portability.

**Built to be extended.** Primitives, losses, and densification strategies are separate shader units behind small interfaces, so adding a new primitive kind or a new loss should be a rather local change rather than surgery across the whole pipeline.


## Features

- Full training from COLMAP scenes: tile-binned forward rasterization, a hand-written backward pass, Adam, and a held-out evaluation split.
- Densification you choose per run: a gradient (AbsGS-style) path, and a structure-aware (SAD-GS-style) path that grid-splits gaussians which under-resolve local image structure.
- Mip-Splatting 3D smoothing filter for scale-correct, anti-aliased rendering.
- Deterministic Nsight GPU Trace capture hooks for profiling the real, optimized kernels (this is Nvidia specific, but we want to allow integrating profiling also on AMD Radeon GPU Profiler (RGP)).

<p align="center">
  <img src="assets/examples_strip.jpg" alt="Three novel views rendered from a scene trained by this pipeline" width="95%">
  <br><em>Three novel views of the same reconstruction. Every frame is rendered from the trained gaussians, not the input photos.</em>
</p>

## Results

Measured on one **RTX 4080 (16 GB)** under Windows with the Vulkan backend. Every run trains 3000 iterations with structure-aware densification and Mip-Splatting, holds out every 8th view for testing, uses seed 42, and is run once; the seed alone moves a scene by up to about 0.2 dB (kitchen SH0: 29.88 dB at seed 42, 30.06 at seed 0). Mip-NeRF 360 (9 scenes) trains at quarter resolution, on the same images as the SADGS runs below; Tanks & Temples (train, truck) at 980×545 and Deep Blending (drjohnson, playroom) at native resolution.

PSNR is held-out sRGB PSNR averaged over the test views. Time is the training loop alone, without data loading or the final metrics pass. VRAM is the whole process at the default 7M-splat capacity. Every configuration, scene by scene, is in [docs/benchmark_results.csv](docs/benchmark_results.csv); [scripts/plots/plot_readme_results.py](scripts/plots/plot_readme_results.py) lists the sweep commands and regenerates every figure below.

### Default configuration

Dense supervision, fp32 optimizer state. SADGS is the reference CUDA/PyTorch implementation of structure-aware densification, with its 3000-iteration Mip-NeRF 360 recipe on the same images and holdout. Its SH0 column is the earlier reference run of all nine scenes; its SH3 column was rerun on this GPU, which only the indoor scenes fit (see [Speed and memory](#speed-and-memory-against-sadgs)).

| Scene | PSNR SH0 | SADGS SH0 | PSNR SH3 | SADGS SH3 | Time SH0 (s) | Time SH3 (s) | Splats (M) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| bicycle | 24.42 | 24.46 | 24.80 |  | 33.9 | 45.3 | 5.30 |
| flowers | 21.12 | 21.26 | 21.34 |  | 40.4 | 54.8 | 5.40 |
| garden | 26.43 | 26.25 | 26.74 |  | 48.7 | 66.1 | 5.32 |
| stump | 26.02 | 25.91 | 26.37 |  | 35.7 | 48.5 | 5.72 |
| treehill | 22.16 | 22.11 | 22.21 |  | 39.8 | 53.5 | 5.47 |
| bonsai | 29.98 | 29.72 | 30.56 | 30.94 | 19.2 | 24.8 | 2.08 |
| counter | 27.45 | 27.31 | 27.97 | 28.55 | 22.9 | 30.3 | 1.86 |
| kitchen | 29.88 | 29.98 | 30.52 | 31.03 | 23.2 | 30.3 | 1.67 |
| room | 30.71 | 30.83 | 30.88 | 31.24 | 17.9 | 22.5 | 1.70 |
| train | 20.28 |  | 20.52 |  | 35.2 | 48.8 | 3.45 |
| truck | 24.84 |  | 25.27 |  | 28.8 | 39.2 | 2.90 |
| drjohnson | 28.65 |  | 28.71 |  | 31.0 | 39.4 | 5.25 |
| playroom | 29.71 |  | 30.25 |  | 28.6 | 35.6 | 4.37 |
| **Mip-NeRF 360** | **26.46** | **26.43** | **26.82** |  | **31.3** | **41.8** | **3.83** |
| **Tanks & Temples** | **22.56** |  | **22.89** |  | **32.0** | **44.0** | **3.17** |
| **Deep Blending** | **29.18** |  | **29.48** |  | **29.8** | **37.5** | **4.81** |

<p align="center">
  <img src="assets/psnr_sweep.png" alt="Held-out PSNR per scene for this trainer and SADGS, SH0 and SH3" width="95%">
  <br>
  <img src="assets/timing_sweep.png" alt="Training-loop time per scene for this trainer and SADGS, SH0 and SH3" width="95%">
</p>

At SH0 the trainer matches SADGS on Mip-NeRF 360 (26.46 against 26.43 dB) with a similar number of splats (3.83M against 3.70M). SH3 adds 0.3–0.4 dB on every dataset for about a third more training time. On the indoor scenes SADGS's SH3 is 0.46 dB higher. It turns every SH band on from the first iterations, while this trainer adds one band every 500 iterations by default (`--sh-band-interval`, derived from `--iters`); switching every band on from the start (`--sh-band-interval 1`) gains 0.07 dB on Mip-NeRF 360 (0.11 dB indoors), so the schedule is only part of the difference.

| Dataset | Method | PSNR | SSIM | LPIPS | Splats (M) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Mip-NeRF 360 | Ours SH0 | 26.46 | 0.809 | 0.216 | 3.83 |
| Mip-NeRF 360 | SADGS SH0 | 26.43 | 0.811 |  | 3.70 |
| Mip-NeRF 360 | Ours SH3 | 26.82 | 0.817 | 0.207 | 3.82 |
| Tanks & Temples | Ours SH0 | 22.56 | 0.841 | 0.220 | 3.17 |
| Tanks & Temples | Ours SH3 | 22.89 | 0.847 | 0.210 | 3.16 |
| Deep Blending | Ours SH0 | 29.18 | 0.902 | 0.328 | 4.81 |
| Deep Blending | Ours SH3 | 29.48 | 0.905 | 0.322 | 4.81 |

LPIPS is shown for this trainer only: SADGS's own LPIPS code scores identical images about 7% lower than the `lpips` package used here, while PSNR and SSIM agree.

### Speed and memory against SADGS

Same GPU, same images, on the four indoor scenes only: on bicycle, SADGS's densification at iteration 1000 fills the 16 GB card and training drops to a few iterations per second, and the other outdoor and Deep Blending scenes grow as many splats. SADGS keeps its training images in host memory here (`--data_device cpu`) so that it fits; its memory is PyTorch's peak reservation, ours the whole process.

| Scene | Ours SH0 (s) | SADGS SH0 (s) | Ours SH3 (s) | SADGS SH3 (s) | Ours SH3 VRAM (GiB) | SADGS SH3 VRAM (GiB) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| bonsai | 19.2 | 44.4 | 24.8 | 46.7 | 8.8 | 9.1 |
| counter | 22.9 | 48.4 | 30.3 | 52.1 | 8.8 | 9.8 |
| kitchen | 23.2 | 51.3 | 30.3 | 50.7 | 8.8 | 8.1 |
| room | 17.9 | 41.8 | 22.5 | 42.5 | 8.8 | 8.4 |
| **Average** | **20.8** | **46.5** | **27.0** | **48.0** | **8.8** | **8.8** |

On these scenes the trainer is 2.2× faster than SADGS at SH0 and 1.8× faster at SH3. At SH3 both use about 8.8 GiB, although ours reserves room for 7M splats where these scenes need under 2.1M (`--max-splats` trades that headroom for memory); bf16 optimizer moments take ours to 7.2 GiB. At SH0 ours needs 5.3 GiB against 9.0 GiB, partly because the SADGS SH0 run keeps every SH coefficient allocated and only pins the active degree to 0.

<p align="center">
  <img src="assets/vram.png" alt="GPU memory while training on the Mip-NeRF 360 indoor scenes" width="60%">
</p>

### Sparse supervision (STBN)

`--stbn-pixel-fraction ρ` supervises only a fraction ρ of the pixels in each backward pass, chosen by NVIDIA's spatiotemporal blue-noise masks, and scales them by 1/ρ so the gradient stays an unbiased estimate of the dense one.

SH0:

| Pixels / step | M360 PSNR | M360 time (s) | T&T PSNR | T&T time (s) | DB PSNR | DB time (s) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 100% | 26.46 | 31.3 | 22.56 | 32.0 | 29.18 | 29.8 |
| 50% | 26.38 | 28.8 | 22.50 | 30.6 | 29.20 | 27.5 |
| 25% | 26.29 | 26.3 | 22.34 | 29.0 | 29.16 | 24.8 |
| 10% | 25.81 | 25.4 | 22.07 | 28.7 | 28.98 | 23.5 |
| 2% | 23.41 | 27.1 | 19.76 | 31.9 | 27.17 | 22.8 |

SH3:

| Pixels / step | M360 PSNR | M360 time (s) | T&T PSNR | T&T time (s) | DB PSNR | DB time (s) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 100% | 26.82 | 41.8 | 22.89 | 44.0 | 29.48 | 37.5 |
| 50% | 26.79 | 39.5 | 22.87 | 42.7 | 29.23 | 35.3 |
| 25% | 26.66 | 36.9 | 22.69 | 41.2 | 29.31 | 32.4 |
| 10% | 26.23 | 35.9 | 22.37 | 41.2 | 29.25 | 31.3 |
| 2% | 23.68 | 39.8 | 20.11 | 48.5 | 27.44 | 30.5 |

<p align="center">
  <img src="assets/stbn_tradeoff.png" alt="Held-out PSNR against training time for each supervised-pixel fraction" width="95%">
  <br>
  <img src="assets/stbn_convergence.png" alt="Held-out PSNR against training time at 100, 25, 10 and 2 percent of pixels" width="95%">
  <br>
  <img src="assets/stbn_qualitative.jpg" alt="Held-out bicycle view: ground truth and renders trained with 100, 25, 10 and 2 percent of pixels" width="95%">
</p>

Down to 25% of the pixels PSNR drops by about 0.2 dB or less, for a 6–17% shorter loop; 10% costs 0.2–0.65 dB. The speedup is bounded because sparse supervision only shortens the backward pixel loop: projection, sorting, the forward pass, the optimizer and densification still run in full. At 2% quality collapses, and the noisier gradients densify more splats (4.7M against 3.9M on average), which makes it slower than 10% on Mip-NeRF 360 and Tanks & Temples.

### Optimizer precision

`--adam-moments bf16` stores the optimizer's first and second moments in bf16 with stochastic rounding instead of fp32 (dense supervision below).

| SH | Adam moments | M360 PSNR | T&T PSNR | DB PSNR | Time (s) | VRAM (GiB) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| SH0 | fp32 | 26.46 | 22.56 | 29.18 | 31.2 | 5.37 |
| SH0 | bf16 | 26.47 | 22.54 | 29.18 | 30.2 | 4.64 |
| SH3 | fp32 | 26.82 | 22.89 | 29.48 | 41.5 | 8.92 |
| SH3 | bf16 | 26.84 | 22.84 | 29.47 | 39.0 | 7.25 |

Time and VRAM are averages over all 13 scenes. Over the 130 paired fp32/bf16 runs of the whole grid (every scene, pixel fraction and SH degree), held-out PSNR differs by 0.07 dB on average, which is run-to-run noise at one run per cell. bf16 saves 0.7 GiB at SH0 and 1.7 GiB at SH3 at the default capacity, enough to bring SH3 with room for 7M splats under 8 GiB, and the loop is 3–6% faster. It is off by default and fixed at launch, because it changes the compiled shaders.

### Convergence

<p align="center">
  <img src="assets/convergence.png" alt="Held-out PSNR every 100 iterations for all 13 scenes" width="70%">
</p>

Held-out PSNR every 100 iterations (SH0, dense supervision). The dip at iteration 1200 is the scheduled opacity reset; the trainer recovers within about 200 iterations.

## How it trains

```
COLMAP scene  ->  init gaussians  ->  per-iteration loop:

      sample a training camera
      forward rasterize    (tile bin, depth sort, front-to-back blend)
      backward rasterize   (analytic gradients into fixed-point buffers)
      Adam step            (per-parameter LRs, extent-scaled means LR)
      densify on schedule  (gradient or structure-aware) then Morton re-sort
      refresh the mip 3D filter

  every N iters:  held-out eval  ->  PSNR / SSIM  ->  export PLY
```

The loop keeps the GPU queue shallow with fence-based backpressure instead of draining it every step, so the CPU never races minutes ahead of the device.

## Architecture

`TrainingHub` handles interactive commands and reconstruction. It delegates each
run to `TrainingApplication`, whose loop owns camera sampling, schedules,
evaluation, and reporting. The backend consumes cameras, learning rates, and
per-step render parameters; it does not depend on COLMAP entries or UI events.

`TrainingPipeline.train_step(camera, lrs=..., contextual=...)` records a complete
optimizer step. Experiments can instead call `record_forward`, `record_loss`,
and `record_backward` inside `pipeline.recording(...)` to inspect intermediate
results without applying Adam. Forward and backward must use the same per-step
parameters; call `clear_gradients()` before each independent gradient sample.

`GpuRuntime` owns command submission and queue waits. The viewer shares that
device while keeping its own rendering buffers; snapshots carry primitive data,
active SH degree, and extinction hardness. Generated contracts keep Python
storage layouts and Slang structures aligned.

Structure-aware buffers are allocated only for that densification mode and
initialized before use. Training targets stay in host memory by default and are
uploaded inside each step's command buffer (`--image-residency gpu` keeps one
texture per training camera instead). At startup the trainer plans its GPU
allocations against the VRAM budget Windows grants the process (or
`--vram-budget-gib`) and lowers `--max-splats`, with a warning, when they would
not fit.

## Backends and portability

| Backend | Status | Shader IR |
|---|---|---|
| Vulkan | primary | SPIR-V |
| D3D12 | cross-checked | DXIL |
| others | as slang-rhi adds them | Slang recompiles, no rewrite |

Every kernel is written once in Slang. Switching backends is a single flag (`--backend vulkan` or `--backend d3d12`), and the smoke tests compile the full shader set on both so a change that breaks one is caught early. Nothing in the training path depends on CUDA, cuBLAS, or a specific driver runtime.

## Install

You need a Vulkan or D3D12 capable GPU with recent drivers, and Python 3.10 or newer.

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux / macOS
source .venv/bin/activate

pip install -e ".[dev]"
```

For a regular installation, use `pip install .` or install a built wheel, then
run `python -m trainer.frontend.app`. Wheels and source distributions include
the Slang shaders. Installed runs resolve shaders from package resources and
keep their STBN cache in the working directory; editable checkouts retain the
repository's `.stbn` cache.

Scenes use the Mip-NeRF 360 layout (a COLMAP `sparse/` folder plus `images/`). Point `--scene` at any such folder.

## Datasets

The download helpers link to **three datasets** that can be obtained separately by users:

| Dataset | Project / attribution | Included scene selection |
| --- | --- | --- |
| Mip-NeRF 360 | [Barron et al., CVPR 2022](https://jonbarron.info/mipnerf360/) | bicycle, bonsai, counter, garden, kitchen, room, stump; optional flowers and treehill |
| Tanks & Temples | [Knapitsch et al., ACM TOG 2017](https://www.tanksandtemples.org/) | train, truck |
| Deep Blending | [Hedman et al., ACM TOG 2018](https://visual.cs.ucl.ac.uk/pubs/deepblending/) | drjohnson, playroom |

Fetch the [Mip-NeRF 360](https://jonbarron.info/mipnerf360/) dataset with the helper script (standard library only, no extra dependencies):

```bash
# The seven public scenes (bicycle, bonsai, counter, garden, kitchen, room, stump)
# -> datasets/360_v2/
python scripts/data/download_mipnerf360.py

# Also fetch the extra scenes (flowers, treehill) -> datasets/360_extra_scenes/
python scripts/data/download_mipnerf360.py --set all
```

The archives are large (~12 GB for the public scenes, ~4.5 GB more for the extras). The download streams to disk with a progress readout, is idempotent (re-running skips scenes already present), and extracts into place. Useful flags:

- `--set {v2,extra,all}` — which scenes to fetch (default `v2`).
- `--dest <dir>` — download somewhere other than the repo `datasets/` folder.
- `--force` — re-download even if the scenes already exist.
- `--keep-zip` — keep the downloaded `.zip` archives after extraction.

Fetch the **Tanks & Temples** and **Deep Blending** scenes (the other datasets the 3DGS papers report on) with a second helper. Both ship in one ~650 MB archive from the INRIA 3DGS authors:

```bash
# Both -> datasets/tandt/{train,truck} and datasets/db/{drjohnson,playroom}
python scripts/data/download_tandt_db.py

# Only one dataset (still downloads the combined archive)
python scripts/data/download_tandt_db.py --set tandt   # or: --set db
```

Same flags as above (`--dest`, `--force`, `--keep-zip`).

## Quickstart

```bash
# Structure-aware densification with anti-aliasing, headless
python scripts/running/v2_train.py --scene datasets/360_v2/garden \
    --densify-mode structure_aware --mip-splatting --no-viewer

# Watch it train in the live viewer
python scripts/running/v2_train.py --scene datasets/360_v2/room --densify-mode structure_aware

# Cross-check the same run on D3D12
python scripts/running/v2_train.py --scene datasets/360_v2/room --backend d3d12 --no-viewer
```

Each run writes to `outputs/train/<run-name>/`: a `log.jsonl`, a `final_psnr.json` summary, an `output.ply`, and per-camera renders under `cams/`. The plot scripts in `scripts/plots/` turn those into convergence and comparison figures.

The trainer also opens `errors.log` in the initial run directory **before GPU setup** and prints its full path. Python tracebacks and native stderr (including SlangPy/Vulkan errors) go there and are echoed to the terminal and the viewer's **DIAGNOSTICS** log. In hub mode this file covers the entire session, including subsequent runs. Native aborts can close the window before it refreshes; already-written diagnostics remain in the file. This records errors, not crash recovery, and cannot recover messages a driver never emits.

## Testing

```bash
pytest -q
```

The suite covers Slang math against known inputs, gradient checks against finite differences, the CLI contract, the training schedule, and a device-creation smoke test on each backend. GPU tests are skipped automatically where no device is available.

## License

This project is provided under the BSD 3 Clause License, see [LICENSE](LICENSE) for details. See [NOTICE](NOTICE.md) for licensing details on other software and data.

## CONTRIBUTING

Before you can contribute, EA must have a Contributor License Agreement (CLA) on file that has been signed by each contributor. You can sign [here](https://electronicarts.na1.echosign.com/public/esignWidget?wid=CBFCIBAA3AAABLblqZhByHRvZqmltGtliuExmuV-WNzlaJGPhbSRg2ufuPsM3P0QmILZjLpkGslg24-UJtek*).
