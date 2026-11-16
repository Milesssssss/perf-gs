#!/usr/bin/env bash
# Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

# SADGS reference counterparts on THIS machine's GPU, for the README comparison:
# the Mip-NeRF 360 indoor scenes at images_2 (the 3DGS evaluation resolution, as in
# the README) with the Config 3 recipe, at SH0 (active degree pinned) and
# SH3. On a 16 GB card the outdoor scenes do not fit: bicycle fills the GPU at the
# iteration-1000 densification and pages, so their times would be meaningless. The indoor
# scenes at images_2 also reserve ~18 GB, so those runs report quality only.
# Images stay in host memory (--data_device cpu) for the same reason, and the
# allocator avoids expandable segments, which fail under WSL here ("device not ready").
# train (loop timed by timed_train.py) -> render test views -> metrics. Idempotent:
# a scene with results.json is skipped. Output: outputs/sadgs_counterparts/sh<D>/<scene>/.
# Run from Windows: wsl -e bash /mnt/c/.../scripts/shell/_sadgs_run/run_counterparts.sh
set -uo pipefail

REPO=/mnt/c/Users/sscolari/Desktop/slang_3dgs_trainer
export SADGS_ROOT=/mnt/c/Users/sscolari/Desktop/SADGS
export MAMBA_ROOT_PREFIX=$HOME/mm CUDA_HOME=$HOME/mm/envs/sadgs
export PATH=$HOME/mm/envs/sadgs/bin:$PATH
export LD_LIBRARY_PATH=$HOME/mm/envs/sadgs/lib/python3.11/site-packages/torch/lib:$HOME/mm/envs/sadgs/lib:${LD_LIBRARY_PATH:-}
export PYTORCH_CUDA_ALLOC_CONF=garbage_collection_threshold:0.7,max_split_size_mb:256
PY=$HOME/mm/envs/sadgs/bin/python

M360=(--optimizer_type hybrid --split_ratio_threshold 0.4 --sample_bbox_faces
      --warmup_densification --mult 0.25 --iterations 3000 --position_lr_max_steps 3000
      --opacity_reset_interval 1200 --densification_interval 500 --densify_until_iter 1900
      --densify_from_iter 100 --prune_iterations 1900)

run_scene() {  # <sh degree> <dataset-relative scene path>
  local sh=$1 scene_path=$2 name out pin=0
  name=$(basename "$scene_path")
  out=$REPO/outputs/sadgs_counterparts/sh$sh/$name
  if [ -f "$out/results.json" ]; then
    echo "==== SKIP $name sh$sh (results.json exists) ===="
    return
  fi
  [ "$sh" = 0 ] && pin=1
  mkdir -p "$out"
  echo "==================== [$(date +%H:%M:%S)] TRAIN $name sh$sh ===================="
  REF_PIN_SH0=$pin REF_TIMING_OUT=$out/timing.json "$PY" "$REPO/scripts/shell/_sadgs_run/timed_train.py" \
    -s "$REPO/datasets/$scene_path" -i images_2 --eval --data_device cpu "${M360[@]}" \
    --save_iterations 3000 --model_path "$out" > "$out/train.log" 2>&1 \
    || { echo "TRAIN FAILED $name sh$sh"; tail -3 "$out/train.log"; return; }
  (cd "$SADGS_ROOT" && "$PY" render.py -m "$out" --iteration 3000 --skip_train --mult 0.25) \
    > "$out/render.log" 2>&1 || { echo "RENDER FAILED $name sh$sh"; return; }
  (cd "$SADGS_ROOT" && "$PY" metrics.py -m "$out") > "$out/metrics.log" 2>&1 \
    || { echo "METRICS FAILED $name sh$sh"; return; }
  rm -rf "$out/point_cloud"
  echo "==== [$(date +%H:%M:%S)] DONE $name sh$sh ===="
  cat "$out/timing.json" "$out/results.json"
}

for sh in 0 3; do
  for scene in 360_v2/bonsai 360_v2/counter 360_v2/kitchen 360_v2/room; do
    run_scene "$sh" "$scene"
  done
done
echo "==================== ALL COUNTERPARTS COMPLETE ===================="
