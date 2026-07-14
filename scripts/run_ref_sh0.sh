#!/usr/bin/env bash
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
export REF_POP_OUT=$REPO/outputs/ref_popstat_sh0.jsonl
export REF_EVAL_OUT=$REPO/outputs/ref_eval_sh0.json

cd "$REPO/SADGS/SADGS"
exec "$HOME/mm/envs/sadgs/bin/python" train.py \
  -s "$REPO/datasets/360_v2/bicycle" -i images_4 --eval \
  --optimizer_type hybrid --split_ratio_threshold 0.4 --sample_bbox_faces \
  --warmup_densification --mult 0.25 \
  --iterations "$ITERS" --position_lr_max_steps 3000 \
  --opacity_reset_interval 1200 --densification_interval 500 \
  --densify_until_iter 1900 --densify_from_iter 100 \
  --save_iterations 3000 --model_path /tmp/ref_bike_sh0
