#!/usr/bin/env bash
# Full SADGS reference pipeline on FLOWERS @ our config (images_4, 3000 iters, SH0),
# mirroring scripts/run_ref_sh0_full.sh (verified on bicycle). --eval uses llffhold=8,
# the SAME every-8th-camera holdout the Slang trainer reports on, so the PSNR is
# directly comparable to outputs/train/mipnerf360_run_1/flowers.
set -euo pipefail

REPO=/mnt/c/Users/sscolari/Desktop/slang_3dgs_trainer
OUT=$REPO/outputs/SADGS_ref_sh0_flowers

export MAMBA_ROOT_PREFIX=$HOME/mm CUDA_HOME=$HOME/mm/envs/sadgs
export PATH=$HOME/mm/envs/sadgs/bin:$PATH
export LD_LIBRARY_PATH=$HOME/mm/envs/sadgs/lib/python3.11/site-packages/torch/lib:$HOME/mm/envs/sadgs/lib:${LD_LIBRARY_PATH:-}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export REF_ST_BATCH=16
export REF_PIN_SH0=1
export REF_POP_OUT=$OUT/ref_popstat.jsonl
export REF_EVAL_OUT=$OUT/ref_eval_inmem.json

PY=$HOME/mm/envs/sadgs/bin/python
mkdir -p "$OUT"
cd "$REPO/SADGS/SADGS"

echo "==================== [1/3] TRAIN (flowers, SH0, 3000 it) ===================="
"$PY" train.py \
  -s "$REPO/datasets/360_extra_scenes/flowers" -i images_4 --eval \
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
cat "$OUT/results.json" 2>/dev/null || echo "(no results.json)"
