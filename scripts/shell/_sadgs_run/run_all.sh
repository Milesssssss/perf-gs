#!/usr/bin/env bash
# SADGS reference (SH0) on ALL Mip-NeRF 360 scenes (7 v2 + flowers/treehill),
# same config/resolution as run_flowers.sh, --eval (llffhold=8) so PSNR is on
# the same held-out test split the Slang trainer reports. Idempotent: skips any
# scene that already has results.json. Output -> outputs/SADGS_ref_sh0_<scene>/.
set -uo pipefail   # NOT -e: one scene failing must not abort the batch

REPO=/mnt/c/Users/sscolari/Desktop/slang_3dgs_trainer
export MAMBA_ROOT_PREFIX=$HOME/mm CUDA_HOME=$HOME/mm/envs/sadgs
export PATH=$HOME/mm/envs/sadgs/bin:$PATH
export LD_LIBRARY_PATH=$HOME/mm/envs/sadgs/lib/python3.11/site-packages/torch/lib:$HOME/mm/envs/sadgs/lib:${LD_LIBRARY_PATH:-}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export REF_ST_BATCH=16
export REF_PIN_SH0=1
PY=$HOME/mm/envs/sadgs/bin/python
cd "$REPO/SADGS/SADGS"

SCENES=(
  "360_v2/bicycle"
  "360_v2/bonsai"
  "360_v2/counter"
  "360_v2/garden"
  "360_v2/kitchen"
  "360_v2/room"
  "360_v2/stump"
  "360_extra_scenes/flowers"
  "360_extra_scenes/treehill"
)

for sp in "${SCENES[@]}"; do
  name=$(basename "$sp")
  OUT=$REPO/outputs/SADGS_ref_sh0_$name
  if [ -f "$OUT/results.json" ]; then
    echo "==== SKIP $name (results.json exists) ===="
    continue
  fi
  echo "==================== [$(date +%H:%M:%S)] TRAIN $name ===================="
  export REF_EVAL_OUT=$OUT/ref_eval_inmem.json
  mkdir -p "$OUT"
  "$PY" train.py -s "$REPO/datasets/$sp" -i images_4 --eval \
    --optimizer_type hybrid --split_ratio_threshold 0.4 --sample_bbox_faces \
    --warmup_densification --mult 0.25 \
    --iterations 3000 --position_lr_max_steps 3000 \
    --opacity_reset_interval 1200 --densification_interval 500 \
    --densify_until_iter 1900 --densify_from_iter 100 \
    --save_iterations 3000 --model_path "$OUT" || { echo "TRAIN FAILED $name"; continue; }
  echo "==================== RENDER $name ===================="
  "$PY" render.py -m "$OUT" --iteration 3000 --mult 0.25 || { echo "RENDER FAILED $name"; continue; }
  echo "==================== METRICS $name ===================="
  "$PY" metrics.py -m "$OUT" || { echo "METRICS FAILED $name"; continue; }
  echo "==== [$(date +%H:%M:%S)] DONE $name ===="
  cat "$OUT/results.json"
done
echo "==================== ALL SCENES COMPLETE ===================="
