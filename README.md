# PERF-GS

**3D Gaussian Splatting. Trained in seconds.**

Turn a COLMAP scene into a 3D model you can explore and export as a PLY. At 3,000 steps on an RTX 4080, Ultrafast averages **17.62 seconds of optimization** across nine Mip-NeRF 360 scenes. Quality averages **27.78 dB PSNR in 33.85 seconds**.

[Quickstart](#quickstart) · [Benchmarks](#mip-nerf-360-at-3000-steps) · [Gallery](#reconstructions)

<p align="center">
  <a href="assets/perfgs-demo.webp"><img src="assets/perfgs-demo.webp" width="896" alt="Stump in real time, from opening the COLMAP scene through 3,000 Ultrafast steps. The timer freezes at training completion as the camera moves through the result."></a>
  <br><em>Stump · Ultrafast · RTX 4080 · real-time playback</em> · <a href="assets/perfgs-demo.gif">GIF version</a>
</p>

**18.14 s from opening the scene to training completion** = 0.19 s to load COLMAP + 0.52 s before pressing Start + 3.41 s for setup + 14.02 s of optimization. The clock stays at 18.14 s during the final camera move.

## Mip-NeRF 360 at 3,000 steps

| Preset | Training loop | Launch to saved PLY | PSNR ↑ | SSIM ↑ | VGG LPIPS ↓ |
| --- | ---: | ---: | ---: | ---: | ---: |
| Ultrafast | 17.62 s | 31.28 s | 27.13 dB | 0.7936 | 0.2567 |
| Balanced (default) | 25.97 s | 41.30 s | 27.39 dB | 0.8143 | 0.2229 |
| Quality | 33.85 s | 49.56 s | 27.78 dB | 0.8275 | 0.2011 |

Headless runs on Windows, RTX 4080 16 GB, Vulkan, seed 42. The training loop excludes loading, initialization and export. Launch to saved PLY includes all three and a disk sync. Each scene contributes equally. Graphdeco’s RGB8 evaluator scores 246 held-out views at quarter resolution outdoors and half resolution indoors. [Reproduce the benchmark](docs/mipnerf360_3k.md#reproduce-the-benchmark) · [Per-scene results and published comparisons](docs/benchmarks.md)

Ultrafast requires NVIDIA’s separately licensed non-commercial research masks. Download them once with `python scripts/data/download_stbn.py`. Balanced and Quality work with the base install. [Mask terms](docs/stbn_pixel_sampling.md)

<details>
<summary>Speed versus quality</summary>

![Mean training-loop seconds versus held-out PSNR for the three presets on the same nine scenes](assets/perfgs-speed-quality.png)

</details>

## Quickstart

With Python 3.10+ and a Vulkan or D3D12 GPU, launch the hub from the repository root.

```sh
python -m pip install -e .
python -m trainer.frontend.app --hub
```

Drop the folder containing `images/` and `sparse/0/` into the hub. Click **Start Training**, explore the result, then choose **Export PLY...**. If you have photos only, choose **Reconstruct to COLMAP** first. [Hub walkthrough](docs/usage.md#interactive-hub)

![PERF-GS hub showing the trained kitchen scene with Export PLY and Set up another run controls](assets/perfgs-hub.webp)

**Try a sample from the command line.** Download the Tanks & Temples sample archive (about 650 MB) and train the truck scene.

```sh
python scripts/data/download_tandt_db.py --set tandt
python -m trainer.frontend.app --scene datasets/tandt/truck --no-viewer --save-ply
```

The model is saved in `outputs/train/<run-name>/output.ply`. Evaluation is off by default. Add `--run-report` for held-out metrics. [Datasets](docs/usage.md#datasets) · [Compatibility and first-run help](docs/usage.md#first-run-help)

## Reconstructions

Each row shows one held-out photograph and all three 3,000-step presets rendered from that camera. Beneath each full view is the same boxed detail, cropped without color or exposure adjustments. Select a tile to see the crop at native resolution.

| Scene | Reference photo | Ultrafast | Balanced | Quality |
| --- | --- | --- | --- | --- |
| **Kitchen** | [<img src="assets/reconstructions/kitchen-reference.webp" width="170" alt="Kitchen reference photograph and detail crop">][kitchen-reference] | [<img src="assets/reconstructions/kitchen-ultrafast.webp" width="170" alt="Kitchen Ultrafast render and detail crop">][kitchen-ultrafast] | [<img src="assets/reconstructions/kitchen-balanced.webp" width="170" alt="Kitchen Balanced render and detail crop">][kitchen-balanced] | [<img src="assets/reconstructions/kitchen-quality.webp" width="170" alt="Kitchen Quality render and detail crop">][kitchen-quality] |
| **Bicycle** | [<img src="assets/reconstructions/bicycle-reference.webp" width="170" alt="Bicycle reference photograph and detail crop">][bicycle-reference] | [<img src="assets/reconstructions/bicycle-ultrafast.webp" width="170" alt="Bicycle Ultrafast render and detail crop">][bicycle-ultrafast] | [<img src="assets/reconstructions/bicycle-balanced.webp" width="170" alt="Bicycle Balanced render and detail crop">][bicycle-balanced] | [<img src="assets/reconstructions/bicycle-quality.webp" width="170" alt="Bicycle Quality render and detail crop">][bicycle-quality] |
| **Stump** | [<img src="assets/reconstructions/stump-reference.webp" width="170" alt="Stump reference photograph and detail crop">][stump-reference] | [<img src="assets/reconstructions/stump-ultrafast.webp" width="170" alt="Stump Ultrafast render and detail crop">][stump-ultrafast] | [<img src="assets/reconstructions/stump-balanced.webp" width="170" alt="Stump Balanced render and detail crop">][stump-balanced] | [<img src="assets/reconstructions/stump-quality.webp" width="170" alt="Stump Quality render and detail crop">][stump-quality] |
| **Room** | [<img src="assets/reconstructions/room-reference.webp" width="170" alt="Room reference photograph and detail crop">][room-reference] | [<img src="assets/reconstructions/room-ultrafast.webp" width="170" alt="Room Ultrafast render and detail crop">][room-ultrafast] | [<img src="assets/reconstructions/room-balanced.webp" width="170" alt="Room Balanced render and detail crop">][room-balanced] | [<img src="assets/reconstructions/room-quality.webp" width="170" alt="Room Quality render and detail crop">][room-quality] |
| **Bonsai** | [<img src="assets/reconstructions/bonsai-reference.webp" width="170" alt="Bonsai reference photograph and detail crop">][bonsai-reference] | [<img src="assets/reconstructions/bonsai-ultrafast.webp" width="170" alt="Bonsai Ultrafast render and detail crop">][bonsai-ultrafast] | [<img src="assets/reconstructions/bonsai-balanced.webp" width="170" alt="Bonsai Balanced render and detail crop">][bonsai-balanced] | [<img src="assets/reconstructions/bonsai-quality.webp" width="170" alt="Bonsai Quality render and detail crop">][bonsai-quality] |

[kitchen-reference]: assets/reconstructions/kitchen-reference.webp
[kitchen-ultrafast]: assets/reconstructions/kitchen-ultrafast.webp
[kitchen-balanced]: assets/reconstructions/kitchen-balanced.webp
[kitchen-quality]: assets/reconstructions/kitchen-quality.webp
[bicycle-reference]: assets/reconstructions/bicycle-reference.webp
[bicycle-ultrafast]: assets/reconstructions/bicycle-ultrafast.webp
[bicycle-balanced]: assets/reconstructions/bicycle-balanced.webp
[bicycle-quality]: assets/reconstructions/bicycle-quality.webp
[stump-reference]: assets/reconstructions/stump-reference.webp
[stump-ultrafast]: assets/reconstructions/stump-ultrafast.webp
[stump-balanced]: assets/reconstructions/stump-balanced.webp
[stump-quality]: assets/reconstructions/stump-quality.webp
[room-reference]: assets/reconstructions/room-reference.webp
[room-ultrafast]: assets/reconstructions/room-ultrafast.webp
[room-balanced]: assets/reconstructions/room-balanced.webp
[room-quality]: assets/reconstructions/room-quality.webp
[bonsai-reference]: assets/reconstructions/bonsai-reference.webp
[bonsai-ultrafast]: assets/reconstructions/bonsai-ultrafast.webp
[bonsai-balanced]: assets/reconstructions/bonsai-balanced.webp
[bonsai-quality]: assets/reconstructions/bonsai-quality.webp

Scenes and reference photography from Barron et al., CVPR 2022. [Mip-NeRF 360 dataset](https://jonbarron.info/mipnerf360/)

## How it works

Python and SlangPy drive the same Slang rendering, optimization and densification shaders on Vulkan and D3D12. The trainer starts with COLMAP points, adds stereo matches from training views and fits a lens response from those views. Vulkan is the default. On Windows, select D3D12 with `--backend d3d12`. [Shaders](shaders/) · [Preset settings](docs/usage.md#training)

## License

This project is provided under the [BSD 3-Clause License](LICENSE), see LICENSE for details. See [NOTICE](NOTICE.md) for licensing details on other software and data.

## Contributing

Before you can contribute, EA must have a Contributor License Agreement (CLA) on file that has been signed by each contributor. You can sign [here](https://electronicarts.na1.echosign.com/public/esignWidget?wid=CBFCIBAA3AAABLblqZhByHRvZqmltGtliuExmuV-WNzlaJGPhbSRg2ufuPsM3P0QmILZjLpkGslg24-UJtek*).
