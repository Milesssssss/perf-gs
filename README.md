<!-- Images in assets/ are real outputs of this repo: teaser.jpg and
     examples_strip.jpg are renders from a trained bicycle scene (Mip-NeRF 360,
     30k iters); convergence.png is plotted from the per-scene psnr_curve.json
     runs. Swap in your own renders by replacing the files under assets/. -->

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
It converges in ~47s per scene at SH0 and ~69s at SH3, averaged over the full Mip-NeRF 360 dataset on an RTX 3090. The codebase has no CUDA dependencies, is directly integratable with pytorch frontend and all shaders are dispatched via SlangPy.

<p align="center">
  <img src="assets/teaser.jpg" alt="Mip-NeRF 360 bicycle scene trained by this pipeline, rendered from a held-out view" width="80%">
  <br><em>Mip-NeRF 360 bicycle, trained end to end by this pipeline and rendered from a held-out camera.</em>
</p>

## Why this exists

Most open 3DGS trainers are welded to CUDA.<br>
By writing the kernels in Slang and having the whole system driven by slangpy, we aim to allow distributing this trainer on any system and any backend. 

**Portability first.** The rasterizer, the gradients, the optimizer, and the densification kernels are all Slang. The same source compiles to SPIR-V for Vulkan and DXIL for D3D12 today, with room for more backends as slang-rhi grows. There are no `.cu` files anywhere and nothing that assumes a specific vendor.

**Modular codebase.** Most 3DGS trainer usually live on a performance-readability spectrum, we aimed to have both, while also having flexible portability.

**Built to be extended.** Primitives, losses, and densification strategies are separate shader units behind small interfaces, so adding a new primitive kind or a new loss should be a rather local change rather than surgery across the whole pipeline.


## What it does

- Full training from COLMAP scenes: tile-binned forward rasterization, a hand-written backward pass, Adam, and a held-out evaluation split.
- Densification you choose per run: a gradient (AbsGS-style) path, and a structure-aware (SAD-GS-style) path that grid-splits gaussians which under-resolve local image structure.
- Mip-Splatting 3D smoothing filter for scale-correct, anti-aliased rendering.
- Deterministic Nsight GPU Trace capture hooks for profiling the real, optimized kernels (this is Nvidia specific, but I want to allow integrating profiling also on AMD Radeon GPU Profiler (RGP)).

<p align="center">
  <img src="assets/examples_strip.jpg" alt="Three novel views rendered from a scene trained by this pipeline" width="95%">
  <br><em>Three novel views of the same reconstruction. Every frame is rendered from the trained gaussians, not the input photos.</em>
</p>

## Results

Full Mip-NeRF 360 dataset (9 scenes), quarter resolution, 3000 iterations, structure-aware densification with Mip-Splatting, every 8th view held out for test. Held-out sRGB PSNR and full-loop train time (final metrics pass excluded), for view-independent color (SH0) and the full view-dependent spherical-harmonic model (SH3), on an RTX 3090.

| Scene | PSNR SH0 (dB) | PSNR SH3 (dB) | Time SH0 (seconds) | Time SH3 (seconds) |
|---|---:|---:|---:|---:|
| bicycle | 24.62 | 24.61 | 53.2 | 77.2 |
| flowers | 21.22 | 21.22 | 58.6 | 90.4 |
| garden | 26.56 | 26.54 | 71.4 | 108.4 |
| stump | 26.15 | 26.19 | 54.5 | 82.0 |
| treehill | 22.19 | 22.21 | 58.4 | 81.4 |
| bonsai | 30.17 | 30.16 | 28.9 | 39.9 |
| counter | 27.55 | 27.51 | 36.0 | 53.6 |
| kitchen | 30.06 | 30.10 | 36.2 | 51.4 |
| room | 30.85 | 30.91 | 27.7 | 38.4 |
| **Average** | **26.60** | **26.61** | **47.2** | **69.2** |

<p align="center">
  <img src="assets/psnr_sweep.png" alt="Held-out PSNR per scene, SH0 vs SH3" width="90%">
  <br>
  <img src="assets/timing_sweep.png" alt="Training time per scene, SH0 vs SH3" width="90%">
</p>

The higher-order SH bands leave PSNR essentially flat on this dataset (+0.01 dB average) while adding ~47% to train time, so SH0 is the better default here; SH3 is available for scenes with strong view-dependent shading. This is not a claim of a new state of the art — it is a check that a from-scratch Slang implementation reproduces a known-good pipeline faithfully, the bar a research base needs to clear before it is worth building on.

<p align="center">
  <img src="assets/convergence.png" alt="Test PSNR versus iteration" width="70%">
</p>

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

## Datasets

The download helpers cover **three datasets**, obtained separately by users:

| Dataset | Project / attribution | Included scene selection |
| --- | --- | --- |
| Mip-NeRF 360 | [Barron et al., CVPR 2022](https://jonbarron.info/mipnerf360/) | bicycle, bonsai, counter, garden, kitchen, room, stump; optional flowers and treehill |
| Tanks & Temples | [Knapitsch et al., ACM TOG 2017](https://www.tanksandtemples.org/) | train, truck |
| Deep Blending | [Hedman et al., ACM TOG 2018](https://visual.cs.ucl.ac.uk/pubs/deepblending/) | drjohnson, playroom |

Raw datasets are not part of the current tracked source tree. Dataset and
derived-asset terms are separate from the source-code license; see
[NOTICE.md](NOTICE.md#research-datasets-downloaded-separately) before downloading
or redistributing them. Public availability does not imply unrestricted rights.

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

The extra scenes (flowers, treehill) carry a redistribution caveat from the dataset authors; the script prints a notice before fetching them. See <https://jonbarron.info/mipnerf360/> for details.

Fetch the **Tanks & Temples** and **Deep Blending** scenes (the other datasets the 3DGS papers report on) with a second helper. Both ship in one ~650 MB archive from the INRIA 3DGS authors:

```bash
# Both -> datasets/tandt/{train,truck} and datasets/db/{drjohnson,playroom}
python scripts/data/download_tandt_db.py

# Only one dataset (still downloads the combined archive)
python scripts/data/download_tandt_db.py --set tandt   # or: --set db
```

Same flags as above (`--dest`, `--force`, `--keep-zip`). The script prints a terms notice before fetching. Review the terms for the specific archive and intended use; see [Tanks & Temples](https://www.tanksandtemples.org/), [Deep Blending](https://visual.cs.ucl.ac.uk/pubs/deepblending/), and the dataset qualifications in [NOTICE.md](NOTICE.md#research-datasets-downloaded-separately).

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

## Testing

```bash
pytest -q
```

The suite covers Slang math against known inputs, gradient checks against finite differences, the CLI contract, the training schedule, and a device-creation smoke test on each backend. GPU tests are skipped automatically where no device is available.

## Status

This is a research trainer. The happy path (Mip-NeRF 360, SH0, structure-aware densification, Vulkan) is well travelled and gives the numbers above. 

### TODOs/next steps

- [ ] Experiment with view dependant reconstruction (Spherical harmonics / Spherical gaussians)
- [ ] Harden the primitive generalization
- [ ] Improve Adam's generalization and refactor optimizer state, slightly confusing rn
- [ ] Implement HYPERPARAM-free densification strategy. SADGS is great, but required a lot of manual tuning. MCMC is nice for that, but wont converge as fast. I need to find a middle ground between speed (hopefully maintaining the 3k/4k step range for convergence) while avoiding overparameterized training setup.
- [ ] Include Glomap/Colmap preprocessing stage
- [ ] Improve viewer, likely should move to separate queues on separate devices and have shared memory mapped that viewer reads from and trainer pushes too.
- [ ] Include PPISP to improve training quality
- [ ] Refactor to allow easier integration of neural backbone for primitive prediction
- [ ] Ideally this becomes a fast and efficient differentiable blended-primitive renderer that allows rapid experimentation of NVS and/or generative 3D experiments

## License and third-party notices

EA-owned source code is provided under the [EA license](LICENSE). Included
third-party code retains its original copyrights and license terms; the EA
license does not relicense those portions. See [NOTICE.md](NOTICE.md) for copied
or ported code, separately installed dependencies, optional downloaded assets,
and dataset attribution. Research references remain documented alongside the
relevant algorithms.

## CONTRIBUTING

Before you can contribute, EA must have a Contributor License Agreement (CLA) on file that has been signed by each contributor. You can sign [here](https://electronicarts.na1.echosign.com/public/esignWidget?wid=CBFCIBAA3AAABLblqZhByHRvZqmltGtliuExmuV-WNzlaJGPhbSRg2ufuPsM3P0QmILZjLpkGslg24-UJtek*).
