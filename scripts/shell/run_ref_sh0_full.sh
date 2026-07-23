#!/usr/bin/env bash
# Full reference (SADGS) pipeline on bicycle @ our config (images_4, 3000 iters),
# SH0 (REF_PIN_SH0=1: active_sh_degree pinned to 0 = DC-only color, tensors full-
# sized so no zero-grid kernel crash). Runs THEIR OWN code end to end:
#   1. train.py       -> saves point_cloud/iteration_3000/point_cloud.ply + cfg_args
#   2. render.py      -> dumps test/ours_3000/{renders,gt}/*.png (GT + renders)
#   3. metrics.py     -> their canonical results.json + per_view.json (PSNR/SSIM/LPIPS)
# Output persists to a Windows path so it survives WSL shutdown.
#
# NOTE: --mult 0.25 is passed to BOTH train and render (render.py defaults --mult
# to 0.5; must match training's 0.25 or renders differ).
set -euo pipefail

REPO=/mnt/c/Users/sscolari/Desktop/slang_3dgs_trainer
OUT=$REPO/outputs/SADGS_ref_sh0_bicycle          # persistent Windows path

export MAMBA_ROOT_PREFIX=$HOME/mm CUDA_HOME=$HOME/mm/envs/sadgs
export PATH=$HOME/mm/envs/sadgs/bin:$PATH
export LD_LIBRARY_PATH=$HOME/mm/envs/sadgs/lib/python3.11/site-packages/torch/lib:$HOME/mm/envs/sadgs/lib:${LD_LIBRARY_PATH:-}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export REF_ST_BATCH=16
export REF_PIN_SH0=1
export REF_POP_OUT=$OUT/ref_popstat.jsonl        # trajectory alongside the model
export REF_EVAL_OUT=$OUT/ref_eval_inmem.json     # my in-memory eval (cross-check)

PY=$HOME/mm/envs/sadgs/bin/python
mkdir -p "$OUT"
cd "$REPO/SADGS/SADGS"

echo "==================== [1/3] TRAIN ===================="
"$PY" train.py \
  -s "$REPO/datasets/360_v2/bicycle" -i images_4 --eval \
  --optimizer_type hybrid --split_ratio_threshold 0.4 --sample_bbox_faces \
  --warmup_densification --mult 0.25 \
  --iterations 3000 --position_lr_max_steps 3000 \
  --opacity_reset_interval 1200 --densification_interval 500 \
  --densify_until_iter 1900 --densify_from_iter 100 \
  --save_iterations 3000 --model_path "$OUT"

echo "==================== [2/3] RENDER (GT + renders @ iter 3000) ===================="
"$PY" render.py -m "$OUT" --iteration 3000 --mult 0.25

echo "==================== [3/3] METRICS (their canonical results.json) ===================="
"$PY" metrics.py -m "$OUT"

echo "==================== DONE ===================="
echo "Output folder: $OUT"
ls -R "$OUT" | head -40
