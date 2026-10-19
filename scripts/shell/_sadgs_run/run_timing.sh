# Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

set -uo pipefail

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
  OUT=$REPO/outputs/sadgs_timing/$name
  if [ -f "$OUT/timing.json" ]; then
    echo "==== SKIP $name (timing.json exists) ===="
    continue
  fi
  echo "==================== [$(date +%H:%M:%S)] TIMING TRAIN $name ===================="
  mkdir -p "$OUT"
  export REF_TIMING_OUT=$OUT/timing.json
  export REF_EVAL_OUT=$OUT/eval.json
  "$PY" train.py -s "$REPO/datasets/$sp" -i images_4 --eval \
    --optimizer_type hybrid --split_ratio_threshold 0.4 --sample_bbox_faces \
    --warmup_densification --mult 0.25 \
    --iterations 3000 --position_lr_max_steps 3000 \
    --opacity_reset_interval 1200 --densification_interval 500 \
    --densify_until_iter 1900 --densify_from_iter 100 \
    --save_iterations 4000 --model_path "$OUT" || { echo "TIMING FAILED $name"; continue; }
  echo "==== [$(date +%H:%M:%S)] DONE $name ===="
  cat "$OUT/timing.json"
  echo
done
echo "==================== ALL TIMING RUNS COMPLETE ===================="
