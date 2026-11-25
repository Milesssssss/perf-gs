# Usage

## Installation

PERF-GS requires Python 3.10 or newer and a GPU with a Vulkan or D3D12 driver. Run the commands below from the checkout with your Python environment active. The README has the editable install and first training command. For a regular installation, use `python -m pip install .` or install a built wheel. The package includes the Slang shaders. [Quickstart](../README.md#quickstart)

### First-run help

The benchmark workstations ran Windows, Python 3.11.9, SlangPy 0.42.0 and Vulkan. Balanced used 2.72–2.90 GiB of process-local GPU memory at loop end across the 16 GB RTX 4080 and 24 GB RTX 4090 runs. Peak usage and the interactive viewer’s memory requirements can be higher. [RTX 4090 measurements](benchmarks.md#rtx-4090-presets) · [RTX 4080 measurements by preset](benchmarks.md#current-presets)

- If PowerShell opens the Microsoft Store or reports “Python was not found,” run `.\.venv\Scripts\python.exe -m trainer.frontend.app --hub` from the checkout. This uses the installed benchmark environment directly. For a fresh checkout, follow the [Windows setup](../README.md#quickstart).
- If Python cannot find `trainer`, run `python -m pip install -e .` from the repository root using the same Python that launches the hub.
- If the window closes during GPU setup, open the `errors.log` path printed in the terminal. On Windows you can also start the hub with `python -m trainer.frontend.app --hub --backend d3d12` to use Direct3D 12.

## Datasets

Pass `--scene` the directory containing the registered photos in `images/` and the COLMAP reconstruction in `sparse/0/`. The loader accepts binary or text cameras, images and points. For `--downsample 2` or `4` it uses `images_2/` or `images_4/` when available and otherwise resizes on load. Use `--downsample 1` for the original resolution. [COLMAP loader](../src/trainer/data/colmap.py)

The optional download helpers use the dataset authors’ archives. Paths below are relative to the checkout.

| Dataset | Command | Download and destination |
| --- | --- | --- |
| Mip-NeRF 360 | `python scripts/data/download_mipnerf360.py` | About 12 GB for bicycle, bonsai, counter, garden, kitchen, room and stump in `datasets/360_v2/`. [Dataset](#ref-mipnerf360) |
| Mip-NeRF 360 extra scenes | `python scripts/data/download_mipnerf360.py --set extra` | About 4.5 GB for flowers and treehill in `datasets/360_extra_scenes/`. Use `--set all` for all nine. [Dataset and terms](https://jonbarron.info/mipnerf360/) |
| Tanks & Temples and Deep Blending | `python scripts/data/download_tandt_db.py` | One archive of about 650 MB. Use `--set tandt` to keep train and truck in `datasets/tandt/`, or `--set db` to keep drjohnson and playroom in `datasets/db/`. [Tanks & Temples](#ref-tandt) · [Deep Blending](#ref-deepblending) |

Both helpers skip complete datasets. Use `--dest DIR` to choose the parent directory, `--keep-zip` to retain the archive, or `--force` to download again.

Sparse pixel supervision with NVIDIA’s spatiotemporal blue-noise masks requires a separate download. From a checkout, `python scripts/data/download_stbn.py` puts the masks in `.stbn/`. A wheel installation looks for the cache in the working directory. [Mask sources and asset terms](stbn_pixel_sampling.md)

## Training

The trainer opens its viewer by default. Use `--no-viewer` for headless training, or `--wait-for-viewer` to inspect the initial Gaussians before starting. Vulkan is the default backend. On Windows, select D3D12 with `--backend d3d12`.

During optimization the viewer presents at up to 15 frames per second so training can use the same GPU. Pausing the run or viewing the completed model restores the viewer’s normal refresh rate.

Choose a preset with `--preset` or in the viewer. All presets use stereo matches from training views and a lens response fitted from training landmarks. [RTX 4090 benchmarks](benchmarks.md#rtx-4090-presets) · [RTX 4080 benchmarks](benchmarks.md#current-presets)

| Preset | Steps | SH degree | Pixels supervised per step | Growth capacity | Fraction kept after growth | Training images |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `ultrafast` | 3,000 | 2 | 25% STBN | 1.85M | 100% | GPU-resident |
| `balanced` (default) | 3,000 | 2 | 100% | 2.60M | 100% | Streamed |
| `quality` | 3,500 | 3 | 100% | 4.00M | 75% target | Streamed |
| `quality+` | Until convergence | 3 | 100% | 4.00M | 75% target | Streamed |

Ultrafast renders at the full selected image dimensions and samples the pixel gradients. Download NVIDIA’s masks once with `python scripts/data/download_stbn.py`. Balanced, Quality and Quality+ use every pixel and need no mask download. [STBN asset terms](stbn_pixel_sampling.md)

All presets use structure-aware densification. It compares a Gaussian’s projected axes with local image detail across training views, then splits it along axes that cover too much detail. The CLI also supports threshold-based 3DGS splitting with `--densify-mode simple` and a fixed set of Gaussians with `--densify-mode none`. [SAD-GS](#ref-sadgs) · [Split and prune decisions](../shaders/densify/decide_actions_structure.slang)

When Quality’s proposed splits exceed `--max-splats`, it ranks complete expansions by the parent’s average screen-position loss gradient divided by the proposed family size. A parent can split into two, then four, then eight children if each expansion earns enough slots. The strongest image-detail axis goes first. The GPU groups similar scores into buckets. Within the last admitted bucket, an expansion receives slots only if its whole family fits. Slots freed by pruning become available at the next growth update. [Split allocator](../shaders/densify/rank_families.slang)

Quality finishes growth at step 1,900. Over the next 200 steps it samples one pixel per 4×4 block and estimates how much removing each splat would change the rendered RGB, including occlusion by the splats in front. It then drops the lowest-scoring splats to retain 75% of the live population and refines the survivors for the rest of the run. Splats with no observations are kept. `--final-splat-fraction` controls the fraction, `--iters` sets the step count, and both also appear in the hub. [Removal cost](../shaders/densify/removal_cost.slang)

The published Mip-NeRF 360 comparison uses the selected image resolution for training and final evaluation. Explicit settings override the preset. To control the lens response, pass `--lens-strength 1` for fixed cos⁴ falloff or `--lens-strength 0` for a flat response. [Camera model and PLY metadata](mipnerf360_3k.md#camera-response-and-ply-playback)

### Quality+ convergence

Quality+ starts with the same 3,500 updates as Quality. It then keeps the splats and optimizer state and continues refinement at Quality’s final learning rates and loss weights. The first convergence check is at step 3,500 with default settings. Each check renders up to eight fixed training cameras and averages their PSNR. Checks are separated by four passes through the training images (at least 100 updates). A cumulative gain of 0.05 dB restarts the count. Three checks without that gain end the run. The official test cameras are reserved for reporting.

The default `--iters 0` lets the scene determine the duration. Pass a positive `--iters` to set an upper limit. The run report distinguishes reaching the plateau from hitting this limit or being interrupted. This measures the plateau of the 4M-capacity Quality model.

### Interactive hub

Launch the hub from the repository root with your trainer environment active. It starts empty. To open a known scene immediately, add `--scene PATH`.

```sh
python -m trainer.frontend.app --hub
```

1. Click **Open dataset folder...** and choose a folder, or drop it into the window. A COLMAP scene with `images/` and `sparse/` is ready to train. With photos only, click **Install reconstruction tools** if prompted, then **Reconstruct to COLMAP** to estimate the camera poses and initial points.
2. Choose a **Preset** and set **Iterations**, then click **Start Training**. Open **Advanced training settings** to adjust individual controls. The viewer shows the reconstruction and training progress.
3. When training finishes, choose **Export PLY...** to save the model. The dialog lets you pick the filename and destination. On platforms without a save dialog, the hub uses the run directory.
4. To repeat the run with the same settings, choose **Train again**. If the model is unsaved, the button reads **Discard model and train again**. Training starts from the original point cloud.
5. To try a different preset or iteration count, choose **Change settings...**. You can preview the original point cloud and edit settings before starting, or choose **Back to trained model** to return to the result. **Open another dataset...** switches scenes and offers to export an unsaved model first.

The hub reuses the cameras and original point cloud from the preview. Each run builds its training split and stereo seeds, and the diagnostics log records setup time.

## Evaluation and PLY export

All presets reserve every eighth camera in filename order, starting with the first. Evaluation, metric logging and copies of the reference photos are off by default in the CLI and hub. Enable **Evaluate image quality** under **Advanced training settings** in the hub, or choose a CLI option below. Change the split with `--eval-holdout-stride`, or use `--eval 0` to train on every camera.

Each run writes its resolved `config.json`, training `log.jsonl` and `densify_log.jsonl` to `outputs/train/<run-name>/`. Use `--run-name` to choose that directory.

| Flags to add to a training run | Output |
| --- | --- |
| `--evaluation` | Evaluate once on the last iteration and write `eval_curve.jsonl`. Use `--eval-every N` for periodic sweeps. |
| `--final-test-psnr` | `test_psnr.json`, including mean per-image sRGB PSNR on the held-out set. |
| `--run-report` | `run_report.json` with per-view PSNR and SSIM, timings, memory and split details. Add `--save-test-renders` for PNGs in `test_renders/`. |
| `--run-report --eval-lpips` | VGG LPIPS at evaluation resolution. Requires `torch`, `torchvision`, `lpips` and VGG weights. If the model is unavailable, the report records null LPIPS. |
| `--compute-final-metrics` | `output.ply`, `final_psnr.json` and per-camera renders under `cams/`. |
| `--write-ground-truths` | Save the corresponding reference photos as `cams/cam_*/gt.png`. |
| `--save-ply` | Export `output.ply` after the last step. Works independently of evaluation. |

To train room headlessly, export the result and open it in the PLY viewer, run

```sh
python -m trainer.frontend.app --scene datasets/360_v2/room --run-name room_export --downsample 2 --no-viewer --save-ply
python scripts/debug/view_ply.py outputs/train/room_export/output.ply
```

The viewer restores the lens response recorded in the PLY. External renderers need the same camera intrinsics and radial gain to match playback. [PLY metadata and rendering formula](mipnerf360_3k.md#camera-response-and-ply-playback)

## Testing and diagnostics

The source distribution contains the trainer and its shaders. Use a repository checkout for the download helpers and test suite.

```sh
python -m pip install -e ".[dev]"
python -m pytest -q
```

The suite checks CLI settings, schedules, shader math, gradients against finite differences, and Vulkan and D3D12 device creation. Backend-specific GPU tests skip when their device is unavailable.

At startup the trainer prints the path to `errors.log` in the initial run directory. It opens the file before GPU setup and captures Python tracebacks and native stderr. In hub mode the same log covers the entire session. Check it if the viewer closes or a shader fails to compile. The viewer also shows messages in its **DIAGNOSTICS** panel.

Use `python -m trainer.frontend.app --help` for the full CLI. The repository includes plotting scripts for training histories and an Nsight GPU Trace helper for NVIDIA GPUs. Pass your Nsight Graphics project with `--project PATH.ngfx-proj`. [Plotting scripts](../scripts/plots/) · [GPU trace helper](../scripts/running/capture_gpu_trace.py)

## References

1. <a id="ref-3dgs"></a>Kerbl et al. *3D Gaussian Splatting for Real-Time Radiance Field Rendering.* TOG, 2023. [Paper](https://arxiv.org/abs/2308.04079)
2. <a id="ref-mipnerf360"></a>Barron et al. *Mip-NeRF 360: Unbounded Anti-Aliased Neural Radiance Fields.* CVPR, 2022. [Project and data](https://jonbarron.info/mipnerf360/)
3. <a id="ref-colmap"></a>Schönberger and Frahm. *Structure-from-Motion Revisited.* CVPR, 2016. [COLMAP](https://colmap.github.io/)
4. <a id="ref-adam"></a>Kingma and Ba. *Adam: A Method for Stochastic Optimization.* ICLR, 2015. [Paper](https://arxiv.org/abs/1412.6980)
5. <a id="ref-absgs"></a>Ye et al. *AbsGS: Recovering Fine Details for 3D Gaussian Splatting.* 2024. [Paper](https://arxiv.org/abs/2404.10484)
6. <a id="ref-sadgs"></a>Lyu et al. *Faster 3D Gaussian Splatting Convergence via Structure-Aware Densification.* 2026. [Paper](https://arxiv.org/abs/2604.28016)
7. <a id="ref-mipsplatting"></a>Yu et al. *Mip-Splatting: Alias-free 3D Gaussian Splatting.* CVPR, 2024. [Paper](https://arxiv.org/abs/2311.16493)
8. <a id="ref-tandt"></a>Knapitsch et al. *Tanks and Temples: Benchmarking Large-Scale Scene Reconstruction.* TOG, 2017. [Project and data](https://www.tanksandtemples.org/)
9. <a id="ref-deepblending"></a>Hedman et al. *Deep Blending for Free-Viewpoint Image-Based Rendering.* TOG, 2018. [Project and data](https://repo-sam.inria.fr/fungraph/deep-blending/)
10. <a id="ref-ssim"></a>Wang et al. *Image Quality Assessment: From Error Visibility to Structural Similarity.* IEEE Transactions on Image Processing, 2004. [Project](https://www.cns.nyu.edu/~lcv/ssim/)
11. <a id="ref-lpips"></a>Zhang et al. *The Unreasonable Effectiveness of Deep Features as a Perceptual Metric.* CVPR, 2018. [Paper](https://arxiv.org/abs/1801.03924)
12. <a id="ref-stbn"></a>Wolfe et al. *Scalar Spatiotemporal Blue Noise Masks.* EGSR, 2022. [Paper](https://arxiv.org/abs/2112.09629)
13. <a id="ref-ppisp"></a>Deutsch et al. *PPISP: Physically-Plausible Compensation and Control of Photometric Variations in Radiance Field Reconstruction.* 2026. [Paper](https://arxiv.org/abs/2601.18336)
