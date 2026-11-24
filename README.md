# PERF-GS

**Efficient, extensible & low memory differentiable training.**

Train a 3D Gaussian Splatting model from a COLMAP scene with Slang on Vulkan or Direct3D 12. On an **RTX 4090**, Ultrafast averages **27.10 dB PSNR in 11.85 seconds** across all nine Mip-NeRF 360 scenes. On a **16 GB RTX 4080**, the updated Quality preset reaches **27.90 dB in 32.44 seconds** with **2.11M final splats**. Times cover the training loop. [Same-GPU comparison with LiteGS, SAD-GS and FastGS](docs/benchmarks.md#rtx-4080-comparison)

For GPUs with less VRAM, Balanced streams training images from system RAM. It used **2.72–2.90 GiB of VRAM at loop end** across the two workstations. [VRAM and system RAM breakdown](docs/benchmarks.md#where-memory-goes)

[Quickstart](#quickstart) · [Benchmarks](#mip-nerf-360) · [Gallery](#reconstructions)

<div align="center">

![The Stump training timer reaches 9.21 seconds at step 3,000, then the camera orbits the reconstruction.](assets/stump-demo.webp)

*Stump · Ultrafast · RTX 4090 · real-time training* · [Open GIF](assets/stump-preview.gif) · [Full-resolution WebP](assets/stump-rtx4090-orbit.webp)

</div>

**9.21 s for 3,000 optimization steps.** Playback starts with the training loop. Before the clip, the hub spent 0.38 s loading Stump, 0.53 s waiting for Start, and 2.95 s preparing training. The result appeared **13.08 s after scene open**. The same loop took 8.96 s in a separate unrecorded hub run and 8.95 s in the headless benchmark.

## Mip-NeRF 360

| Preset | GPU | Steps | Training loop | PSNR ↑ | SSIM ↑ | VGG LPIPS ↓ | VRAM at loop end |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Ultrafast | RTX 4090 | 3,000 | 11.85 s | 27.10 dB | 0.7930 | 0.2567 | 5.29 GiB |
| Balanced (default) | RTX 4090 | 3,000 | 17.31 s | 27.42 dB | 0.8143 | 0.2229 | 2.81 GiB |
| **Quality** | **RTX 4080** | **3,500** | **32.44 s** | **27.90 dB** | **0.8286** | **0.1997** | **4.80 GiB** |
| Quality (previous) | RTX 4090 | 3,000 | 23.24 s | 27.79 dB | 0.8274 | 0.2011 | 6.14 GiB |

On the same 4080, the previous Quality preset reached 27.78 dB in 33.85 s with 3.08M splats. The updated preset improved PSNR on all nine scenes and used 32% fewer splats on average. Its 4090 rerun is pending. Each row averages nine headless Vulkan runs at seed 42. Graphdeco scored 246 held-out views at quarter resolution outdoors and half resolution indoors. Timing excludes loading, initialization, evaluation and export. [4080 CSV](docs/NVIDIA_GeForce_RTX_4080_presets.csv) · [Original 4090 CSV](docs/NVIDIA_GeForce_RTX_4090_presets.csv) · [Reproduce the benchmark](docs/benchmark-rtx4090.md)

![RTX 4080 training-loop time, PSNR, SSIM and VGG LPIPS by scene for Ultrafast, Balanced and the updated Quality preset](assets/mipnerf360-rtx4080-presets.png)

## Requirements

- **OS and Python.** Python 3.10+. Tested on Windows 11. Linux and macOS are unverified.
- **GPU.** Tested on the NVIDIA RTX 4080 (16 GB) and RTX 4090 (24 GB). The default shaders use 32-lane waves and 64-bit integer buffer atomics through Vulkan (default) or Direct3D 12 (Windows).
- **VRAM.** *Balanced* preset used 2.72-2.90 GiB at the end of headless Mip-NeRF 360 training. Peak and viewer usage can be higher. Minimum card capacity is unverified. [Memory measurements](docs/benchmarks.md#where-memory-goes)

## Quickstart

1. Get **perf-gs.** You can:

   - **Clone with Git.**:
   ```powershell
   git clone https://github.com/electronicarts/perf-gs.git
   ```
   - Or **download a release.**: Open **Releases**, choose a version, then under **Assets** click **Source code (zip)**.
   - Or **download the latest source.**: Click **Code -> Download ZIP**.

2. **Open PowerShell in the project folder.**
3. **Setup venv and install PERF-GS (once)**: In PowerShell, run `py -3.11 --version`.

   - If you see `Python 3.11` followed by a patch number (for example, `Python 3.11.9`), Python is ready.
   - If you get an error, download and open the **Python install manager** from the [Python for Windows page](https://www.python.org/downloads/windows/). After installing it, reopen PowerShell in the project folder and run `pymanager install 3.11` to download and install Python 3.11.

   Once Python is ready, run these two commands, one at a time, to setup a virtual environemnt and install PERF-GS.

   ```powershell
   py -3.11 -m venv .venv
   .\.venv\Scripts\python.exe -m pip install -e .
   ```

4. **Open the training window.**  

   ```powershell
   .\.venv\Scripts\python.exe -m trainer.frontend.app --hub
   ```

5. **Train and save.** Click **Open dataset folder...** and select your scene folder. If it contains only photos, click **Install reconstruction tools** if prompted, then **Reconstruct to COLMAP**. An existing COLMAP scene opens directly. When the point cloud appears, click **Start Training**. When training finishes, click **Export PLY...** to save your model.

![Kitchen in the RTX 4090 hub after 3,000 Balanced steps, with controls to export the trained model or start another run](assets/perfgs-hub.webp)

### Train from the command line

After step 3, you can train directly from PowerShell. From the project folder, run these commands to download the Truck sample (about 650 MB) and train it without opening a window.

```powershell
.\.venv\Scripts\python.exe scripts/data/download_tandt_db.py --set tandt
.\.venv\Scripts\python.exe -m trainer.frontend.app --scene "datasets/tandt/truck" --no-viewer --save-ply
```

For your own COLMAP scene, skip the download and replace `"datasets/tandt/truck"` with your folder path in quotes. The default is Balanced at 3,000 steps. With `--save-ply`, the model is saved in a timestamped folder under `outputs/train/` as `output.ply`.

Run `.\.venv\Scripts\python.exe -m trainer.frontend.app --help` to see every flag. If you already installed PERF-GS in an active Python environment, use `python` in place of `.\.venv\Scripts\python.exe`. [Presets and training settings](docs/usage.md#training) · [Evaluation and export flags](docs/usage.md#evaluation-and-ply-export)

[Hub walkthrough](docs/usage.md#interactive-hub) · [Datasets](docs/usage.md#datasets) · [First-run help](docs/usage.md#first-run-help)

## Reconstructions

Each row pairs a held-out photo with the three renders scored in the original RTX 4090 benchmark. All gallery models used 3,000 steps, including the previous Quality preset.

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

## Architecture from a high level

Python and SlangPy drive the same Slang rendering, optimization and densification shaders on Vulkan and D3D12. The trainer starts with COLMAP points, adds stereo matches from training views and fits a lens response from those views. Vulkan is the default. On Windows, select D3D12 with `--backend d3d12`. [Shaders](shaders/) · [Preset settings](docs/usage.md#training).<br>

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

## How to Cite

Scolari, S. (2026). *PERF-GS*. Electronic Arts. <https://github.com/electronicarts/perf-gs>

```bibtex
@software{scolari2026perfgs,
  author    = {Scolari, Stefano},
  title     = {{PERF-GS}},
  year      = {2026},
  publisher = {Electronic Arts},
  url       = {https://github.com/electronicarts/perf-gs}
}
```

## SEED

<p align="center"><a href="https://seed.ea.com"><img src="assets/SEED.jpg" width="150px"></a><br>
<b>Search for Extraordinary Experiences Division (SEED) - Electronic Arts <br> http://seed.ea.com</b><br>
We are a cross-disciplinary team within EA Worldwide Studios.<br>
Our mission is to explore, build and help define the future of interactive entertainment.<br>
PERF-GS was created by <a href="https://stefanoscolapasta.github.io/nozbufferallowed/about.html">Stefano Scolari</a>.</p>
