# PERF-GS

PERF-GS trains 3D Gaussian Splatting models from COLMAP scenes. Python drives the training loop through SlangPy, and the same Slang shaders handle rendering, optimization, and densification on Vulkan and D3D12.

## Mip-NeRF 360 at 3,000 steps

| Preset | Train loop | Launch to saved PLY | PSNR ↑ | SSIM ↑ | VGG LPIPS ↓ |
| --- | ---: | ---: | ---: | ---: | ---: |
| Ultrafast | 17.62 s | 31.28 s | 27.13 dB | 0.7936 | 0.2567 |
| Balanced | 25.97 s | 41.30 s | 27.39 dB | 0.8143 | 0.2229 |
| Quality | 33.85 s | 49.56 s | 27.78 dB | 0.8275 | 0.2011 |

Nine-scene means on an RTX 4080 using Vulkan, seed 42. Training and evaluation use the stored `images_4` photos outdoors and `images_2` indoors. Graphdeco’s RGB8 evaluator scores all 246 held-out views. Saved-PLY time includes a disk sync. [Benchmark recipe](docs/mipnerf360_3k.md#reproduce-the-benchmark) · [Full results and timing details](docs/benchmarks.md)

<p align="center">
  <img src="assets/teaser.jpg" alt="A held-out view of the Mip-NeRF 360 bicycle scene" width="80%">
  <br><em>Bicycle, rendered from a held-out camera after 30,000 training steps.</em>
</p>

## Quickstart

With Python 3.10+ and a Vulkan or D3D12 GPU, launch the interactive hub from the repository root.

```sh
python -m pip install -e .
python -m trainer.frontend.app --hub
```

Click **Open dataset folder...** or drop your COLMAP scene into the window. Choose a preset, click **Start Training**, then **Export PLY...** to choose where to save the model. [Using the hub](docs/usage.md#interactive-hub)

**Headless example** — train garden with the default `balanced` preset.

```sh
python scripts/data/download_mipnerf360.py
python -m trainer.frontend.app --scene datasets/360_v2/garden --no-viewer --save-ply
```

The model is saved in `outputs/train/<run-name>/output.ply`. Evaluation is off by default. Add `--run-report` to score the held-out views. For ultrafast, first download NVIDIA’s non-commercial research masks with `python scripts/data/download_stbn.py`. [Preset settings](docs/usage.md#training)

## Datasets

| Dataset | Project | Scenes used |
| --- | --- | --- |
| Mip-NeRF 360 | [Barron et al., CVPR 2022](https://jonbarron.info/mipnerf360/) | Seven public scenes, plus flowers and treehill |
| Tanks & Temples | [Knapitsch et al., ACM TOG 2017](https://www.tanksandtemples.org/) | train, truck |
| Deep Blending | [Hedman et al., ACM TOG 2018](https://repo-sam.inria.fr/fungraph/deep-blending/) | drjohnson, playroom |

To download all nine Mip-NeRF 360 scenes, use `python scripts/data/download_mipnerf360.py --set all`. [Dataset paths and download options](docs/usage.md#datasets)

## License

This project is provided under the [BSD 3-Clause License](LICENSE), see LICENSE for details. See [NOTICE](NOTICE.md) for licensing details on other software and data.

## Contributing

Before you can contribute, EA must have a Contributor License Agreement (CLA) on file that has been signed by each contributor. You can sign [here](https://electronicarts.na1.echosign.com/public/esignWidget?wid=CBFCIBAA3AAABLblqZhByHRvZqmltGtliuExmuV-WNzlaJGPhbSRg2ufuPsM3P0QmILZjLpkGslg24-UJtek*).
