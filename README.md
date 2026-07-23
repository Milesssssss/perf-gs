<!-- Images in assets/ are real outputs of this repo: teaser.jpg and
     examples_strip.jpg are renders from a trained bicycle scene (Mip-NeRF 360,
     30k iters); convergence.png is plotted from the per-scene psnr_curve.json
     runs. Swap in your own renders by replacing the files under assets/. -->

# Slang 3DGS Trainer

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![Shaders](https://img.shields.io/badge/Shaders-Slang-DE3163)
![Runtime](https://img.shields.io/badge/Runtime-SlangPy-6E44FF)
![Backend](https://img.shields.io/badge/Backend-Vulkan-A41E22?logo=vulkan&logoColor=white)
![Backend](https://img.shields.io/badge/Backend-D3D12-107C10)
![No CUDA required](https://img.shields.io/badge/CUDA-not%20required-2ea44f)
![Status](https://img.shields.io/badge/status-research-F5A623)

A 3D Gaussian Splatting trainer written in [Slang](https://shader-slang.org) and driven from Python through [SlangPy](https://slangpy.shader-slang.org). It trains real scenes end to end (forward, backward, Adam, densification) and runs on Vulkan and D3D12 rather than a CUDA-only path, so it is not tied to one vendor's GPU stack.

<p align="center">
  <img src="assets/teaser.jpg" alt="Mip-NeRF 360 bicycle scene trained by this pipeline, rendered from a held-out view" width="80%">
  <br><em>Mip-NeRF 360 bicycle, trained end to end by this pipeline and rendered from a held-out camera.</em>
</p>

## Why this exists

Most open 3DGS trainers are welded to CUDA. That is great if you have an NVIDIA card and never plan to leave it, and awkward for everyone else. This project takes the other bet: write the math once in Slang, compile it to whatever backend the runtime supports, and keep the training loop in plain, readable Python.

Three ideas shape the design.

**Portability first.** The rasterizer, the gradients, the optimizer, and the densification kernels are all Slang. The same source compiles to SPIR-V for Vulkan and DXIL for D3D12 today, with room for more backends as slang-rhi grows. There are no `.cu` files anywhere and nothing that assumes a specific vendor.

**Readable by a human.** The Python frontend is organized in layers you can follow top to bottom: setup, the training loop, evaluation, post-training. Each concern (learning-rate schedules, densification, output files, the live viewer) sits in its own small module with a clear API instead of one enormous `main()`.

**Built to be extended.** Primitives, losses, and densification strategies are separate shader units behind small interfaces, so adding a new primitive kind or a new loss is a local change rather than surgery across the whole pipeline.

It is research code and it is honest about that. It also trains Mip-NeRF 360 scenes to quality that matches a solid reference, so it is a genuinely fun place to try ideas.

## What it does

- Full training from COLMAP scenes: tile-binned forward rasterization, a hand-written backward pass, Adam, and a held-out evaluation split.
- Densification you choose per run: a gradient (AbsGS-style) path, and a structure-aware (SAD-GS-style) path that grid-splits gaussians which under-resolve local image structure.
- Mip-Splatting 3D smoothing filter for scale-correct, anti-aliased rendering.
- A live, in-process viewer that shares the training device, so you watch a scene converge with a zero-copy GPU-to-GPU blit and no second process.
- Deterministic Nsight GPU Trace capture hooks for profiling the real, optimized kernels.
- Reproducible metrics: per-image PSNR and SSIM on a held-out split, written as JSON per run so a plot script can pick them up later.

<p align="center">
  <img src="assets/examples_strip.jpg" alt="Three novel views rendered from a scene trained by this pipeline" width="95%">
  <br><em>Three novel views of the same reconstruction. Every frame is rendered from the trained gaussians, not the input photos.</em>
</p>

## Results

Mip-NeRF 360, quarter resolution, spherical-harmonic degree 0, 3000 iterations, every 8th view held out for test. The table compares this trainer against a reference structure-aware densification run on the identical scenes and settings.

| Scene | Slang PSNR (dB) | SAD-GS reference (dB) | Delta |
|---|---:|---:|---:|
| bicycle | 24.64 | 24.46 | +0.18 |
| flowers | 21.23 | 21.26 | -0.03 |
| garden | 26.60 | 26.26 | +0.34 |
| stump | 26.10 | 25.91 | +0.19 |
| treehill | 22.29 | 22.11 | +0.18 |
| bonsai | 30.16 | 29.72 | +0.44 |
| counter | 27.60 | 27.31 | +0.29 |
| kitchen | 30.22 | 29.98 | +0.24 |
| room | 30.66 | 30.83 | -0.18 |
| **Average (9)** | **26.61** | **26.43** | **+0.18** |

Reference quality on the same runs, for context: average SSIM 0.811, LPIPS 0.178 (the trainer tracks these within noise). This is not a claim of a new state of the art. It is a check that a from-scratch Slang implementation reproduces a known-good pipeline faithfully, which is the bar a research base needs to clear before it is worth building on.

<p align="center">
  <img src="assets/convergence.png" alt="Test PSNR versus iteration" width="70%">
</p>

## Performance

Same runs, single GPU, wall-clock for the full 3000-iteration loop (the final metrics pass is measured separately and excluded).

| Scene | Train time (s) |
|---|---:|
| bicycle | 81.4 
| flowers | 90.7 
| garden | 109.4 
| stump | 82.0 
| treehill | 83.1 
| bonsai | 48.4 
| counter | 56.5 
| kitchen | 58.8
| room | 43.5
| **Average** | **72.6**

Evaluated on an RTX 3090.

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

Two halves: Slang shaders that do the GPU work, and a Python package that orchestrates them.

**Shaders (`shaders/`)**, organized by responsibility so each stage is easy to find and swap:

| Folder | Role |
|---|---|
| `common/`, `core/` | shared types, math, interfaces |
| `primitives/` | the gaussian primitive (project, evaluate, covariance) |
| `rasterize/` | tile-binned forward and backward passes |
| `sort/` | depth sort and global radix / Morton ordering |
| `densify/` | clone, split, and prune kernels |
| `structure/` | structure-tensor analysis for structure-aware densification |
| `mip/` | the Mip-Splatting 3D filter |
| `losses/` | L2 and SSIM image losses |
| `optimizers/` | Adam |

**Python (`src/trainer/`)**. The device and pipeline wrap slang-rhi; the training frontend lives in `trainer/training/` split by layer:

| Module | Responsibility |
|---|---|
| `training/app.py` | wires setup, loop, and post-training together |
| `training/cli.py` | the single argparse boundary |
| `training/resources.py` | scene load, eval split, config resolution |
| `training/densification.py` | the densification subsystem controller |
| `training/evaluation.py` | held-out renders and metrics |
| `training/outputs.py` | per-run file lifecycle |
| `training/post_training.py` | final metrics, summary, PLY export |
| `training/viewer_control.py` | the live viewer lifecycle |
| `training/schedule.py`, `lr_schedule.py`, `queue_limiter.py` | timing, LR, GPU backpressure |

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

Scenes use the Mip-NeRF 360 layout (a COLMAP `sparse/` folder plus `images/`). Point `--scene` at any such folder.

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

## Extending it

- **A new primitive.** Add a folder under `shaders/primitives/`, implement the small project / evaluate / covariance interface, and register a specialization. The rasterizer talks to primitives through that interface, not concrete types.
- **A new loss.** Drop a kernel in `shaders/losses/` and add it to the image-loss pass; the Python side reads it back like the existing L2 and SSIM terms.
- **A new densification strategy.** Extend the controller in `training/densification.py`; the four existing modes (structure-aware, AbsGS filler, simple, importance-score) show the shape.
- **A new backend.** When slang-rhi gains one, add it to `create_device` and the backend smoke test. The shaders do not change.

## Testing

```bash
pytest -q
```

The suite covers Slang math against known inputs, gradient checks against finite differences, the CLI contract, the training schedule, and a device-creation smoke test on each backend. GPU tests are skipped automatically where no device is available.

## Repository layout

```
src/trainer/           device, buffers, pipeline, and the training/ frontend package
shaders/               all Slang kernels, grouped by stage
scripts/
  running/             entry points (v2_train.py) + trainer-imported helpers
  plots/               metrics and plotting tools
  debug/               diagnostics, viewers, measurement
  shell/               batch sweeps and reference runs
tests/                 math, gradient, contract, and smoke tests
assets/                images used by this README
```

## Status and honesty

This is a research trainer, not a product. The happy path (Mip-NeRF 360, SH0, structure-aware densification, Vulkan) is well travelled and gives the numbers above. Corners are rougher: higher SH degrees, exotic camera models, and unusual scenes have had less mileage. The code favours being clear and correct over being maximally fast, and it fails fast at boundaries rather than clamping quietly. If something looks wrong, it probably is, and the diagnostics under `scripts/debug/` are there to help you see why.

## Acknowledgements

Built on the ideas in 3D Gaussian Splatting (Kerbl et al.), Mip-Splatting, AbsGS, and structure-aware densification (SAD-GS), and on the [Slang](https://shader-slang.org) and SlangPy toolchain from the shader-slang project. The Mip-NeRF 360 dataset is used for evaluation.
