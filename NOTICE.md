# Third-party notices

SCATTO-GS uses the third-party code, packages, tools, and research datasets
identified below. This notice distinguishes material included in the source
tree from material installed or downloaded separately. It is not an additional
license grant. The [EA license](LICENSE) applies to EA-owned portions only;
third-party portions retain their respective copyrights and terms.

## Third-party code included in the repository

### INRIA / GRAPHDECO 3D Gaussian Splatting evaluation code

- Copyright (C) 2023, Inria. GRAPHDECO research group. All rights reserved.
- Project: [3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting).
- Included evaluator: [src/trainer/frontend/evaluation/metrics.py](src/trainer/frontend/evaluation/metrics.py).
  The original INRIA copyright and license header is retained. Local changes
  include ordering evaluation methods by their iteration suffix.
- Included metric functions: [scripts/plots/image_metrics.py](scripts/plots/image_metrics.py).
  Its source identifies `psnr`, `gaussian`, `create_window`, `ssim`, and `_ssim`
  as verbatim copies of INRIA utility functions obtained through SAD-GS. The
  surrounding run-loading, selection, reporting, and command-line code is local.
- Upstream sources: [evaluator](https://github.com/graphdeco-inria/gaussian-splatting/blob/main/metrics.py),
  [image utilities](https://github.com/graphdeco-inria/gaussian-splatting/blob/main/utils/image_utils.py),
  and [loss utilities](https://github.com/graphdeco-inria/gaussian-splatting/blob/main/utils/loss_utils.py).
- License text: [Gaussian-Splatting License](licenses/INRIA-Gaussian-Splatting-LICENSE.md),
  reproduced from upstream license revision
  [d9fad7b3450bf4bd29316315032d57157e23a515](https://github.com/graphdeco-inria/gaussian-splatting/blob/d9fad7b3450bf4bd29316315032d57157e23a515/LICENSE.md).
  It contains research/non-commercial restrictions and redistribution conditions.
  A notice or an EA copyright header does not remove those restrictions.

Section 6 of the reproduced upstream license separately identifies the
MIT-licensed origin of `utils/loss_utils.py`: **pytorch-ssim, Copyright Evan Su,
2017** ([project](https://github.com/Po-Hsun-Su/pytorch-ssim),
[license](https://github.com/Po-Hsun-Su/pytorch-ssim/blob/master/LICENSE.txt)).
That exception must not be generalized to the whole evaluator or to unrelated
metric functions. The exact imported source revisions, intermediary SAD-GS
provenance, and applicable function-level terms require confirmation before
release; the reproduced license is a reviewed upstream snapshot, not evidence
of the original import revision.

### Thomas Smith's GPUSorting

- Copyright (c) 2024 Thomas Smith.
- Project: [GPUSorting](https://github.com/b0nes164/GPUSorting).
- Included port: [shaders/sort/device_radix_sort.slang](shaders/sort/device_radix_sort.slang).
- Upstream implementation: [DeviceRadixSort.hlsl](https://github.com/b0nes164/GPUSorting/blob/main/GPUSortingD3D12/Shaders/DeviceRadixSort.hlsl)
  and [SortCommon.hlsl](https://github.com/b0nes164/GPUSorting/blob/main/GPUSortingD3D12/Shaders/SortCommon.hlsl),
  both identified upstream as MIT-licensed.
- Local changes: Slang port of the reduce-then-scan WGE16 ascending path;
  64-bit keys stored as two 32-bit words with a Gaussian-index payload;
  eight radix passes; trainer buffer/dispatch contracts and GPU-resident counts.
- License: [full Thomas Smith MIT notice](licenses/GPUSorting-MIT.txt),
  reproduced from the GPUSorting section of upstream license revision
  [6f946265e7a392a12a8422bd8a74f4a629ba94f4](https://github.com/b0nes164/GPUSorting/blob/6f946265e7a392a12a8422bd8a74f4a629ba94f4/LICENSE).
  The original implementation revision used for the port is not recorded.

The upstream repository also contains notices for other implementations and
benchmarks. Their presence in that repository does not establish that those
components are included in this port.

## Packages installed separately

The package manifest is [pyproject.toml](pyproject.toml). These packages are
installed by the user's package manager; their source distributions and binaries
are not vendored here. Each installed version and its bundled dependencies are
subject to their own accompanying licenses. This list does not claim to be a
version-resolved inventory of transitive dependencies.

| Use | Package / source |
| --- | --- |
| GPU runtime and shader dispatch | [SlangPy](https://pypi.org/project/slangpy/) / [shader-slang/slangpy](https://github.com/shader-slang/slangpy) |
| Numerical arrays | [NumPy](https://pypi.org/project/numpy/) |
| Image decoding and resizing | [opencv-python-headless](https://pypi.org/project/opencv-python-headless/) |
| Image input/output | [Pillow](https://pypi.org/project/Pillow/) |
| PLY input/output | [plyfile](https://pypi.org/project/plyfile/) |
| Scientific utilities | [SciPy](https://pypi.org/project/scipy/) |
| Build backend | [setuptools](https://pypi.org/project/setuptools/) |
| Development and tests | [pytest](https://pypi.org/project/pytest/), [Ruff](https://pypi.org/project/ruff/) |
| Optional evaluation and perceptual metrics | [PyTorch](https://pytorch.org/), [torchvision](https://pypi.org/project/torchvision/), [LPIPS](https://github.com/richzhang/PerceptualSimilarity), [lpipsPyTorch](https://github.com/S-aiueo32/lpips-pytorch), [tqdm](https://pypi.org/project/tqdm/) |
| Optional plots | [Matplotlib](https://matplotlib.org/) |
| Optional download certificate bundle | [certifi](https://pypi.org/project/certifi/) |

SlangPy supplies the Slang / slang-rhi toolchain used by the trainer. Its package
and bundled native components retain their own notices; the
[Slang project](https://shader-slang.org/) is not EA-authored material.

## Optional tools, textures, and model weights obtained separately

- **COLMAP and GLOMAP.** Reconstruction setup can download Windows release
  archives from [COLMAP](https://github.com/colmap/colmap/releases) and
  [GLOMAP](https://github.com/colmap/glomap/releases), or use user-installed
  executables. See [src/trainer/frontend/reconstruction/tools.py](src/trainer/frontend/reconstruction/tools.py).
  The default local tool cache is excluded from version control. Consult
  [COLMAP's license](https://colmap.github.io/license.html),
  [GLOMAP's license](https://github.com/colmap/glomap/blob/main/LICENSE), and the
  notices accompanying each binary's separately licensed dependencies.
- **NVIDIA spatiotemporal blue-noise textures.**
  [scripts/data/download_stbn.py](scripts/data/download_stbn.py) is an opt-in
  downloader for the pregenerated scalar texture set in
  [NVIDIA-RTX/STBN](https://github.com/NVIDIA-RTX/STBN). It caches a repackaged
  scalar volume with the downloaded license and a provenance notice outside
  version control; the texture assets and SDK source are not vendored here.
  Citation: Wolfe, Morrical, Akenine-Moller, and Ramamoorthi,
  *Scalar Spatiotemporal Blue Noise Masks*, EGSR 2022,
  [arXiv:2112.09629](https://arxiv.org/abs/2112.09629).
  The current upstream [license document](https://github.com/NVIDIA-RTX/STBN/blob/main/License.txt)
  contains both non-commercial and commercial-use sections. The applicable
  terms must be established for the particular assets and intended use; do not
  treat the downloader's conservative non-commercial warning as a complete
  interpretation of every upstream licensing option. Its download follows
  `main`, not an immutable asset revision.
- **LPIPS and pretrained backbones.** Optional perceptual evaluation in
  [src/trainer/common/perceptual.py](src/trainer/common/perceptual.py) loads
  LPIPS with a pretrained VGG backbone by default; AlexNet is also selectable.
  The supporting library may download backbone weights on first use.
  No pretrained model weights are included in this source tree. Package-source
  licensing does not by itself establish the terms for pretrained weights or
  their training data.

## Research datasets downloaded separately

The dataset download helpers are invoked explicitly by the user. They do not
make the datasets EA-owned or place them under the EA source-code license.
No raw datasets are tracked in the current source tree. Users must review the
terms accompanying the specific dataset archives they obtain; public download
availability is not a statement of unrestricted use or redistribution rights.

| Dataset | Attribution and project | Scenes / download helper |
| --- | --- | --- |
| Mip-NeRF 360 | Barron, Mildenhall, Verbin, Srinivasan, and Hedman, *Mip-NeRF 360: Unbounded Anti-Aliased Neural Radiance Fields*, CVPR 2022. [Project and dataset links](https://jonbarron.info/mipnerf360/) | bicycle, bonsai, counter, garden, kitchen, room, stump; optional flowers and treehill. [scripts/data/download_mipnerf360.py](scripts/data/download_mipnerf360.py) |
| Tanks & Temples | Knapitsch, Park, Zhou, and Koltun, *Tanks and Temples: Benchmarking Large-Scale Scene Reconstruction*, ACM TOG 2017. [Project](https://www.tanksandtemples.org/), [published terms](https://www.tanksandtemples.org/license/) | train, truck, from the INRIA-hosted COLMAP archive. [scripts/data/download_tandt_db.py](scripts/data/download_tandt_db.py) |
| Deep Blending | Hedman, Philip, Price, Frahm, Drettakis, and Brostow, *Deep Blending for Free-Viewpoint Image-Based Rendering*, ACM TOG 2018. [Project](https://visual.cs.ucl.ac.uk/pubs/deepblending/) | drjohnson, playroom, from the same INRIA-hosted COLMAP archive. [scripts/data/download_tandt_db.py](scripts/data/download_tandt_db.py) |

The helpers obtain Mip-NeRF 360 archives from the Google Research links on its
project page, and the Tanks & Temples / Deep Blending archive from the
[INRIA 3DGS dataset host](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/datasets/input/tandt_db.zip).
The Mip-NeRF helper warns about redistribution of the extra scenes. The current
Tanks & Temples terms page contains both CC BY 4.0 language and a restrictive
research-only license grant; the applicable terms for the downloaded archive
require clarification rather than selecting one statement by assumption.
Deep Blending archive terms also require confirmation before redistribution.

### Dataset-derived examples

[assets/teaser.jpg](assets/teaser.jpg) and
[assets/examples_strip.jpg](assets/examples_strip.jpg) are renders of a
Mip-NeRF 360 bicycle reconstruction, as documented in the README. The tracked
convergence, PSNR, and timing plots report dataset-based experiments. These
examples are separate from the raw datasets; excluding raw downloads does not
by itself establish permission to redistribute every image, reconstruction, or
other derived artifact.