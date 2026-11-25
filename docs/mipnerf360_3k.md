# Mip-NeRF 360 benchmark protocol

## Reproduce the benchmark

Both workstations have nine-scene measurements for the current presets and the original 3,000-step recipes. Each per-scene row records the step count, training-loop time, Graphdeco image metrics, final splats and loop-end GPU memory. [Benchmark tables](benchmarks.md) · [Current RTX 4090 CSV](NVIDIA_GeForce_RTX_4090_presets_current.csv) · [Current RTX 4080 CSV](NVIDIA_GeForce_RTX_4080_presets.csv) · [Original RTX 4090 CSV](NVIDIA_GeForce_RTX_4090_presets.csv) · [Original RTX 4080 CSV](mipnerf360_3k_presets.csv)

On Windows with an NVIDIA GPU, use the benchmark driver from the repository root in the trainer’s Python environment. It records the hardware and software, runs all three presets and scores the saved renders with the pinned Graphdeco evaluator. The RTX 4090 guide covers environment and dataset setup. [Reproduce the RTX 4090 benchmark](benchmark-rtx4090.md)

```powershell
.\scripts\running\benchmark_presets.ps1 -ExpectedGpu 'NVIDIA GeForce RTX 4090'
```

For another NVIDIA card, pass its actual adapter name. Each completed sweep produces a CSV under `outputs/benchmarks/<GPU>/<revision>-<source hash>/`. Each preset runs its default step count in a fresh headless process per scene at seed 42, with mid-loop previews, metrics and logging off. Training and held-out rendering use the stored `images_4` outdoors and `images_2` indoors. Stereo initialization uses a 672-pixel-wide working image for up to 28 training-camera anchors and can add up to 256,000 points during setup.

The manual example below runs Balanced. For other presets, use their default step count and a distinct run name per card and preset. Prepare each run with `python scripts/running/prepare_graphdeco.py RUN MODEL`. The driver shows how to score the resulting directories together and collect all 27 rows. [Benchmark driver](../scripts/running/benchmark_presets.ps1) · [Graphdeco reference evaluation](https://github.com/graphdeco-inria/gaussian-splatting/blob/54c035f7834b564019656c3e3fcc3646292f727d/metrics.py)

```sh
python scripts/running/eval_sweep.py --axis seed --values 42 --iters 3000 --name my_gpu_balanced --no-plot --extra --preset balanced --backend vulkan --eval-every 0 --preview-every 0 --no-write-ground-truths --run-report --save-test-renders
```

To score the model from a timed, training-only run, use `--no-evaluation --save-ply` as shown below. Then run `python scripts/running/prepare_graphdeco.py RUN MODEL --ply RUN/output.ply`, substituting the training directory for `RUN` and a fresh output directory for `MODEL`. The converter uses the training log and configuration, restores the PLY’s 3D filter, and renders the registered held-out cameras. Score it with `python scripts/running/score_graphdeco.py --graphdeco outputs/graphdeco_reference MODEL`. Replaying the original Quality preset’s 4,102,802-splat bicycle PLY at seed 42 reproduced all 76,261,050 RGB8 channel values across its 25 held-out views.

Held-out cameras are indices 0, 8, 16, and so on in filename order. Training and stereo initialization use the other views. The benchmark scenes informed preset selection. Each model directory contains `results.json` and `per_view.json`. Aggregate the nine scene means with equal weight for each seed. The reference evaluates saved RGB8 images and passes values in [0, 1] to its VGG LPIPS implementation. The wrapper caches the reference VGG network between images. On one indoor and one outdoor held-out view, its PSNR, SSIM and LPIPS matched the standalone upstream script exactly. [Per-scene preset measurements](NVIDIA_GeForce_RTX_4080_presets.csv)

For elapsed training time including process startup and final PLY export, run each scene in a fresh process and time it externally, using the same `--downsample` setting. Specify the scene path, preset and seed for each measurement. The example shows PowerShell timing for outdoor garden.

```powershell
Measure-Command {
  python -m trainer.frontend.app --scene datasets/360_v2/garden --downsample 4 --preset quality --seed 42 `
    --no-viewer --no-evaluation --save-ply --no-write-ground-truths --preview-every 0 --eval-every 0
}
```

`Measure-Command` stops at process exit. The training log records the optimizer loop under `train_complete.train_wall_s`, and the benchmark CSV distinguishes model readiness, PLY closure, process exit and a completed file sync. In the clean training-only CLI, the writer closes the PLY before exit and the timing harness syncs it immediately afterward. The original RTX 4080 Quality run synced its PLY before scoring held-out views. Fresh processes may reuse filesystem and shader disk caches. Graphdeco’s `full_eval.py` times its training commands including model saves. [Per-scene timing boundaries](mipnerf360_3k_presets.csv) · [Upstream timing script](https://github.com/graphdeco-inria/gaussian-splatting/blob/54c035f7834b564019656c3e3fcc3646292f727d/full_eval.py)

### Compare LiteGS on the same GPU

Use the pinned LiteGS checkout and its CUDA environment. From the PERF-GS repository root, the Bash commands below time all nine scenes and then score their final models. Set `LITEGS` to the path to that checkout. The `--reference` directories come from the PERF-GS sweep above. [LiteGS installation and source](https://github.com/MooreThreads/LiteGS/tree/004b95215c90c36cdaf4b354301132b700ac287b) · [Benchmark helper](../scripts/running/benchmark_litegs.py)

```bash
export PYTHONHASHSEED=42
gpu_tag=$(nvidia-smi -i 0 --query-gpu=name --format=csv,noheader | tr ' ' '_')
read -rp 'Revision-source hash printed in the completed PERF-GS benchmark path: ' sweep_id
sweep_id=${sweep_id//-/_}
for phase in train score; do
  for scene in bicycle flowers garden stump treehill bonsai counter kitchen room; do
    dataset=360_v2
    case "$scene" in flowers|treehill) dataset=360_extra_scenes ;; esac
    extra=()
    if [[ "$phase" == score ]]; then
      extra=(--graphdeco outputs/graphdeco_reference
             --reference "outputs/graphdeco_input/sweep_mip360_${gpu_tag}_${sweep_id}_quality_${scene}_42")
    fi
    python scripts/running/benchmark_litegs.py "$phase" --litegs "$LITEGS" \
      --scene "datasets/$dataset/$scene" --model "outputs/litegs_${gpu_tag}/$scene" "${extra[@]}" || exit 1
  done
done
```

Each LiteGS scene writes its loop time to `time.json` and RGB8 PSNR to `heldout_rgb8.json`. On the RTX 4080, all 246 reference images matched the PERF-GS targets byte for byte. Replaying the first held-out bicycle view through upstream LiteGS reproduced the helper’s PSNR within 0.000001 dB.

### Reproduce the additional trainers

Use a separate upstream checkout and CUDA environment for each trainer. Apply its benchmark patch from the PERF-GS repository root with `git -C "$CHECKOUT" apply "$PWD/scripts/running/benchmark_patches/$PATCH"`. Run the nine Mip-NeRF 360 entries in the linked upstream recipe, retaining its per-scene settings. Set `PYTHONHASHSEED=42`, use one GPU, and keep each model in its own output directory. Use `images_4` outdoors and `images_2` indoors at their stored sizes. Each patch records the synchronized optimization loop in `time.json`.

| Trainer | Upstream training and rendering | Benchmark patch |
| --- | --- | --- |
| SAD-GS | Use the authors’ 3,000-step Mip-NeRF 360 options, including the indoor batch size of two. Pass `--data_device cpu -r 1` and the image folder to training. Render with `render.py --skip_train --iteration 3000`. The patch caches float32 structure maps in system RAM and charges transfers during optimization to the loop. [Pinned recipe](https://github.com/LinjieLyu/SADGS/blob/029814fb5ed773e4a794cec5115951a64f610ab8/run_train.sh) | [sadgs.patch](../scripts/running/benchmark_patches/sadgs.patch) |
| FastGS-Big | Use each scene’s 30,000-step Big options with `-r 1` and the image folder. Render with `render.py --skip_train --iteration 30000`. [Pinned recipe](https://github.com/fastgs/FastGS/blob/44e02a5c1d5e9ed64d2ecd4af1cbba14ac92150f/train_big.sh) | [fastgs.patch](../scripts/running/benchmark_patches/fastgs.patch) |
| SkipGS on FastGS Standard | Install the pinned SkipGS package into the FastGS environment. Use the 30,000-step Standard per-scene options with the same image and render flags as Big. The controller begins at step 15,000. [FastGS Standard recipe](https://github.com/fastgs/FastGS/blob/44e02a5c1d5e9ed64d2ecd4af1cbba14ac92150f/train_base.sh) · [Pinned SkipGS integration](https://github.com/ASU-ESIC-FAN-Lab/SkipGS/blob/d69d9605a18d0f6e5e493ea3d25dfd158849c63e/examples/integrate_fastgs.md) | Apply [fastgs.patch](../scripts/running/benchmark_patches/fastgs.patch) first, then [skipgs-fastgs.patch](../scripts/running/benchmark_patches/skipgs-fastgs.patch). |

The RTX 4080 runs used PyTorch 2.5.1+cu121 on WSL2 with `OMP_NUM_THREADS=4`. SAD-GS used `OPENBLAS_NUM_THREADS=4`. Both FastGS-based runs used `MKL_NUM_THREADS=4` and `OPENBLAS_NUM_THREADS=2`. SkipGS trained from the supplied COLMAP points for all 30,000 iterations and recorded its skipped backward passes in `skip_stats.json`.

Each native renderer writes a `native_views.json` manifest alongside its output. Use the Quality reference directory for the same scene from the sweep above. The scorer checks the camera names and image dimensions before applying Graphdeco’s PSNR to the saved RGB8 renders and the same reference photos.

```bash
python scripts/running/score_benchmark_renders.py --graphdeco outputs/graphdeco_reference \
  --reference "$REFERENCE" --views "$MODEL/native_views.json" --output "$MODEL/heldout_rgb8.json"
```

To plot another card, record its per-scene means in CSVs with the same fields as the RTX 4080 files. Pass the preset CSV with `--presets` and the external-method CSVs with `--baselines` to `python scripts/plots/plot_method_frontiers.py`. The script checks the GPU, image dimensions and held-out counts before creating a card-specific image. [PERF-GS data](NVIDIA_GeForce_RTX_4080_presets.csv) · [LiteGS data](mipnerf360_litegs_4080.csv) · [Additional trainers](mipnerf360_peers_4080.csv)

## Camera response and PLY playback

`--natural-vignetting` applies a fixed cos⁴ falloff with a gamma-2.2 approximation in encoded RGB. For the normalized camera ray `(x, y, 1)` through the pixel center, the RGB gain is `(1 + x² + y²)^(-2 / 2.2)`. Camera intrinsics determine the ray. The same law applies to every scene and view.

For the fixed model, exported PLYs carry the comment `trainer natural_vignetting_v1 cos4_srgb_2.2`. A fitted strength other than zero or one is stored as `trainer natural_vignetting_strength_v2 <strength>`. A flat model needs no comment. The PLY loader and viewer restore the response. To match playback in another renderer, use the gain with the same camera intrinsics and resolution. Ignoring the falloff metadata produces brighter image edges.

## Adapting the falloff to other cameras

All current presets fit one shared strength `b` per single-camera reconstruction with `--auto-lens`. The fitter samples up to 28 decoded training images, finds landmarks seen in two views, and compares pairs of landmarks at different radii. Taking differences in log brightness cancels a shared exposure offset within each image and the intrinsic brightness of each landmark across images. A robust fit reduces the influence of occlusions and view-dependent highlights. The renderer uses `(1 + x² + y²)^(-2 b / 2.2)` during training, evaluation and PLY playback. To override the fit, use `--lens-strength 0` for a flat response, `--lens-strength 1` for the fixed model above, or supply another known strength.

For landmark $p$ in view $i$, write its log brightness as $L_{ip}=R_p+E_i+bG_{ip}$, where $R_p$ is landmark brightness, $E_i$ is exposure and $G_{ip}=(-2/2.2)\log(1+x^2+y^2)$. Two views $i,j$ of two shared landmarks $p,q$ give

$$
(L_{ip}-L_{jp})-(L_{iq}-L_{jq})=b\big[(G_{ip}-G_{jp})-(G_{iq}-G_{jq})\big].
$$

Both unknown brightness terms cancel. The fit uses Huber weights and restricts $b$ to $[0,2]$.

The fit estimates the radial falloff of the processed images. Within supported reconstructions it requires binary COLMAP tracks, one camera record and enough matched points at different radii. Otherwise it uses a flat response. It runs once during setup and reuses the existing per-pixel shader operation. The fitted value is recorded in `config.json` under `cli_args.lens_strength`. [Calibration code](../src/trainer/data/lens_calibration.py) · [Renderer](../shaders/common/camera.slang)

PPISP also models the response curve, exposure and white balance. [PPISP, Sections 4 and 5](https://arxiv.org/html/2601.18336v2)

## Regenerate the benchmark figures

From the repository root, run the following commands with Matplotlib installed. They read the checked-in per-scene CSVs.

```sh
python scripts/plots/plot_preset_results.py --presets docs/NVIDIA_GeForce_RTX_4090_presets_current.csv
python scripts/plots/plot_preset_results.py --presets docs/NVIDIA_GeForce_RTX_4080_presets.csv
python scripts/plots/plot_method_frontiers.py
```

The preset plots print the nine-scene means and write `assets/mipnerf360-rtx4090-presets.png` and `assets/mipnerf360-rtx4080-presets.png`. The method-comparison plot uses the RTX 4080, where all five trainers were measured on the same card.
