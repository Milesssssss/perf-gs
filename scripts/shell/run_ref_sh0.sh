#!/usr/bin/env bash
# Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

# Launch the ACTUAL SADGS reference trainer on bicycle at OUR config
# (images_4 / downsample-4, 3000 iters) with a faithful SH0 color model.
# Produces the missing it3000 test PSNR + full per-densification-event
# population trajectory for the parity comparison.
#
# SH0 approach (no rebuild): keep sh_degree=3 so all SH tensors are FULL-SIZED
# (N,15,3) and non-empty -> no zero-grid CUDA kernel launch inside the compiled
# rasterizer -> avoids the deterministic "invalid configuration argument" that
# --sh_degree 0 triggers via empty _features_rest=(N,0,3). REF_PIN_SH0=1 pins
# active_sh_degree=0 for the whole run (oneupSHdegree patch), so the rasterizer
# evaluates ONLY the DC term (= SH0 color); higher coeffs get zero gradient and
# stay at their zero init. Extra VRAM for unused features_rest + Adam state is
# fine on the 24GB 4090.
#
# Run ONCE in the background. Do NOT poll. Wait for the task notification.
set -euo pipefail

REPO=/mnt/c/Users/sscolari/Desktop/slang_3dgs_trainer
ITERS="${ITERS:-3000}"

export MAMBA_ROOT_PREFIX=$HOME/mm CUDA_HOME=$HOME/mm/envs/sadgs
export PATH=$HOME/mm/envs/sadgs/bin:$PATH
export LD_LIBRARY_PATH=$HOME/mm/envs/sadgs/lib/python3.11/site-packages/torch/lib:$HOME/mm/envs/sadgs/lib:${LD_LIBRARY_PATH:-}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export REF_ST_BATCH=16
export REF_PIN_SH0=1

# Output into a dedicated per-run SADGS folder that mirrors our trainer's
# outputs/train/<run-name>/ layout (final_psnr.json + logs), so the reference
# result sits next to ours for a like-for-like PSNR@3k comparison.
RUN_NAME="${RUN_NAME:-bicycle_sh0}"
OUT="$REPO/outputs/sadgs/$RUN_NAME"
mkdir -p "$OUT"
export REF_EVAL_OUT=$OUT/ref_eval.json

PY="$HOME/mm/envs/sadgs/bin/python"
cd "$REPO/SADGS/SADGS"
"$PY" train.py \
  -s "$REPO/datasets/360_v2/bicycle" -i images_4 --eval \
  --optimizer_type hybrid --split_ratio_threshold 0.4 --sample_bbox_faces \
  --warmup_densification --mult 0.25 \
  --iterations "$ITERS" --position_lr_max_steps 3000 \
  --opacity_reset_interval 1200 --densification_interval 500 \
  --densify_until_iter 1900 --densify_from_iter 100 \
  --save_iterations 3000 --model_path "$OUT"

# Re-emit the reference in-memory eval in our final_psnr.json schema so it can be
# diffed directly against outputs/train/<run>/final_psnr.json (same field names).
"$PY" - "$REF_EVAL_OUT" "$OUT/final_psnr.json" "$ITERS" <<'PYCONV'
import json, sys
src, dst, iters = sys.argv[1], sys.argv[2], int(sys.argv[3])
d = json.load(open(src))
json.dump({
    "psnr_srgb": d.get("psnr_mean_per_image"),
    "ssim_srgb": d.get("ssim_mean_per_image"),
    "n_test_cams": d.get("n_test"),
    "final_n_gaussians": d.get("n_gaussians"),
    "iters": d.get("iteration", iters),
    "eval_holdout": True,
    "eval_holdout_stride": 8,
    "source": "SADGS reference (REF_PIN_SH0=1)",
}, open(dst, "w"), indent=2)
print(f"[convert] wrote {dst}")
PYCONV

echo "SADGS reference output: $OUT"
ls -1 "$OUT"
