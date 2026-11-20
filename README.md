# PERF-GS

**3DGS in seconds.**

On a **16 GB RTX 4080**, Ultrafast averages **27.13 dB PSNR in 17.62 seconds** of training-loop time across the nine Mip-NeRF 360 scenes. Quality reaches **27.78 dB in 33.85 seconds**. Both run for 3,000 steps.

For GPUs with less VRAM, start with **Balanced**. It streams training photos from system RAM and reached **27.39 dB in 25.97 seconds** on the same 4080. The trainer used **2.72–2.89 GiB of VRAM** at the end of each headless run. Peak and viewer usage can be higher. [VRAM by preset](docs/benchmarks.md#current-presets)

[Quickstart](#quickstart) · [Benchmarks](#mip-nerf-360) · [Gallery](#reconstructions)

<p align="center">
  <a href="assets/perfgs-demo.webp?raw=true"><img src="assets/perfgs-demo.webp" width="896" alt="Stump in real time, from opening the COLMAP scene through 3,000 Ultrafast steps. The timer freezes at training completion as the camera moves through the result."></a>
  <br><em>Stump · Ultrafast · RTX 4080 · real-time playback</em> · <a href="assets/perfgs-demo.gif?raw=true">GIF (61 MiB)</a>
</p>

**18.14 s from opening the scene to training completion** = 0.19 s to load COLMAP + 0.52 s before pressing Start + 3.41 s for setup + 14.02 s of optimization.

## Mip-NeRF 360

| Preset (RTX 4080 16 GB) | Training loop | PSNR ↑ | SSIM ↑ | VGG LPIPS ↓ |
| --- | ---: | ---: | ---: | ---: |
| Ultrafast | 17.62 s | 27.13 dB | 0.7936 | 0.2567 |
| Balanced (default) | 25.97 s | 27.39 dB | 0.8143 | 0.2229 |
| Quality | 33.85 s | 27.78 dB | 0.8275 | 0.2011 |

Headless runs on Windows and Vulkan, seed 42. Training-loop time excludes loading, initialization and export. Each scene contributes equally. Graphdeco’s RGB8 evaluator scores 246 held-out views at quarter resolution outdoors and half resolution indoors. [Full RTX 4080 results](docs/benchmarks.md) · [Run the benchmark on RTX 4090](docs/benchmark-rtx4090.md)

Ultrafast requires NVIDIA’s separately licensed non-commercial research masks. Download them once with `python scripts/data/download_stbn.py`. Balanced and Quality work with the base install. [Mask terms](docs/stbn_pixel_sampling.md)

![Training-loop seconds versus held-out PSNR for PERF-GS and other Gaussian splatting trainers on the same RTX 4080.](assets/mipnerf360-rtx4080-frontier.png)

On the same RTX 4080, PERF-GS Quality averaged **27.78 dB in 33.85 s**. FastGS-Big reached **27.83 dB in 325.86 s** on the same nine scenes and held-out photos. [Method settings and per-scene results](docs/benchmarks.md#rtx-4080-comparison)

## Quickstart

With Python 3.10+ and a Vulkan or D3D12 GPU, launch the hub from the repository root.

```sh
python -m pip install -e .
python -m trainer.frontend.app --hub
```

Drop the folder containing `images/` and `sparse/0/` into the hub and click **Start Training**. Explore the result, export a PLY, or train again from the original point cloud. If you have photos only, choose **Reconstruct to COLMAP** first. [Hub walkthrough](docs/usage.md#interactive-hub)

![Trained kitchen scene in the PERF-GS hub, with controls to export the model or start another run](assets/perfgs-hub.webp)

**Try a sample from the command line.** Download the Tanks & Temples sample archive (about 650 MB) and train the truck scene.

```sh
python scripts/data/download_tandt_db.py --set tandt
python -m trainer.frontend.app --scene datasets/tandt/truck --no-viewer --save-ply
```

The model is saved in `outputs/train/<run-name>/output.ply`. Evaluation is off by default. Add `--run-report` for held-out metrics. [Datasets](docs/usage.md#datasets) · [Compatibility and first-run help](docs/usage.md#first-run-help)

## Reconstructions

Each row shows one held-out photograph and all three 3,000-step presets rendered from that camera. Beneath each full view is the same boxed detail, cropped without color or exposure adjustments. Select a tile to see the crop at native resolution.

<table>
  <thead>
    <tr>
      <th scope="col" width="25%">Reference photo</th>
      <th scope="col" width="25%">Ultrafast</th>
      <th scope="col" width="25%">Balanced</th>
      <th scope="col" width="25%">Quality</th>
    </tr>
  </thead>
  <tbody>
    <tr><th scope="rowgroup" colspan="4" align="left">Kitchen</th></tr>
    <tr>
      <td><a href="assets/reconstructions/kitchen-reference.webp"><img src="assets/reconstructions/kitchen-reference.webp" width="216" alt="Kitchen reference photograph and detail crop"></a></td>
      <td><a href="assets/reconstructions/kitchen-ultrafast.webp"><img src="assets/reconstructions/kitchen-ultrafast.webp" width="216" alt="Kitchen Ultrafast render and detail crop"></a></td>
      <td><a href="assets/reconstructions/kitchen-balanced.webp"><img src="assets/reconstructions/kitchen-balanced.webp" width="216" alt="Kitchen Balanced render and detail crop"></a></td>
      <td><a href="assets/reconstructions/kitchen-quality.webp"><img src="assets/reconstructions/kitchen-quality.webp" width="216" alt="Kitchen Quality render and detail crop"></a></td>
    </tr>
  </tbody>
  <tbody>
    <tr><th scope="rowgroup" colspan="4" align="left">Bicycle</th></tr>
    <tr>
      <td><a href="assets/reconstructions/bicycle-reference.webp"><img src="assets/reconstructions/bicycle-reference.webp" width="216" alt="Bicycle reference photograph and detail crop"></a></td>
      <td><a href="assets/reconstructions/bicycle-ultrafast.webp"><img src="assets/reconstructions/bicycle-ultrafast.webp" width="216" alt="Bicycle Ultrafast render and detail crop"></a></td>
      <td><a href="assets/reconstructions/bicycle-balanced.webp"><img src="assets/reconstructions/bicycle-balanced.webp" width="216" alt="Bicycle Balanced render and detail crop"></a></td>
      <td><a href="assets/reconstructions/bicycle-quality.webp"><img src="assets/reconstructions/bicycle-quality.webp" width="216" alt="Bicycle Quality render and detail crop"></a></td>
    </tr>
  </tbody>
  <tbody>
    <tr><th scope="rowgroup" colspan="4" align="left">Stump</th></tr>
    <tr>
      <td><a href="assets/reconstructions/stump-reference.webp"><img src="assets/reconstructions/stump-reference.webp" width="216" alt="Stump reference photograph and detail crop"></a></td>
      <td><a href="assets/reconstructions/stump-ultrafast.webp"><img src="assets/reconstructions/stump-ultrafast.webp" width="216" alt="Stump Ultrafast render and detail crop"></a></td>
      <td><a href="assets/reconstructions/stump-balanced.webp"><img src="assets/reconstructions/stump-balanced.webp" width="216" alt="Stump Balanced render and detail crop"></a></td>
      <td><a href="assets/reconstructions/stump-quality.webp"><img src="assets/reconstructions/stump-quality.webp" width="216" alt="Stump Quality render and detail crop"></a></td>
    </tr>
  </tbody>
  <tbody>
    <tr><th scope="rowgroup" colspan="4" align="left">Room</th></tr>
    <tr>
      <td><a href="assets/reconstructions/room-reference.webp"><img src="assets/reconstructions/room-reference.webp" width="216" alt="Room reference photograph and detail crop"></a></td>
      <td><a href="assets/reconstructions/room-ultrafast.webp"><img src="assets/reconstructions/room-ultrafast.webp" width="216" alt="Room Ultrafast render and detail crop"></a></td>
      <td><a href="assets/reconstructions/room-balanced.webp"><img src="assets/reconstructions/room-balanced.webp" width="216" alt="Room Balanced render and detail crop"></a></td>
      <td><a href="assets/reconstructions/room-quality.webp"><img src="assets/reconstructions/room-quality.webp" width="216" alt="Room Quality render and detail crop"></a></td>
    </tr>
  </tbody>
  <tbody>
    <tr><th scope="rowgroup" colspan="4" align="left">Bonsai</th></tr>
    <tr>
      <td><a href="assets/reconstructions/bonsai-reference.webp"><img src="assets/reconstructions/bonsai-reference.webp" width="216" alt="Bonsai reference photograph and detail crop"></a></td>
      <td><a href="assets/reconstructions/bonsai-ultrafast.webp"><img src="assets/reconstructions/bonsai-ultrafast.webp" width="216" alt="Bonsai Ultrafast render and detail crop"></a></td>
      <td><a href="assets/reconstructions/bonsai-balanced.webp"><img src="assets/reconstructions/bonsai-balanced.webp" width="216" alt="Bonsai Balanced render and detail crop"></a></td>
      <td><a href="assets/reconstructions/bonsai-quality.webp"><img src="assets/reconstructions/bonsai-quality.webp" width="216" alt="Bonsai Quality render and detail crop"></a></td>
    </tr>
  </tbody>
</table>

Scenes and reference photography from Barron et al., CVPR 2022. [Mip-NeRF 360 dataset](https://jonbarron.info/mipnerf360/)

## How it works

Python and SlangPy drive the same Slang rendering, optimization and densification shaders on Vulkan and D3D12. The trainer starts with COLMAP points, adds stereo matches from training views and fits a lens response from those views. Vulkan is the default. On Windows, select D3D12 with `--backend d3d12`. [Shaders](shaders/) · [Preset settings](docs/usage.md#training)

## Datasets

The download helpers link to three datasets that can be obtained separately by users · [Download instructions](docs/usage.md#datasets)

| Dataset | Project / attribution | Included scene selection |
| --- | --- | --- |
| Mip-NeRF 360 | [Barron et al., CVPR 2022](https://jonbarron.info/mipnerf360/) | bicycle, bonsai, counter, garden, kitchen, room, stump (plus optional flowers and treehill) |
| Tanks & Temples | [Knapitsch et al., ACM TOG 2017](https://www.tanksandtemples.org/) | train, truck |
| Deep Blending | [Hedman et al., ACM TOG 2018](https://repo-sam.inria.fr/fungraph/deep-blending/) | drjohnson, playroom |

## License

This project is provided under the BSD 3-Clause License, see [LICENSE](LICENSE) for details. See [NOTICE](NOTICE.md) for licensing details on other software and data.

## Contributing

Before you can contribute, EA must have a Contributor License Agreement (CLA) on file that has been signed by each contributor. You can sign [here](https://electronicarts.na1.echosign.com/public/esignWidget?wid=CBFCIBAA3AAABLblqZhByHRvZqmltGtliuExmuV-WNzlaJGPhbSRg2ufuPsM3P0QmILZjLpkGslg24-UJtek*).
