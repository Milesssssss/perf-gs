# Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

"""Does the GPU ever wait for the host during training? Timestamps each frozen-LR train_step.

idle ~ 0 means the GPU never starves; idle ~ wall - busy means it waits on the host.
    .venv/Scripts/python.exe scripts/debug/measure_step_overlap.py --scene datasets/360_v2/bicycle --probe-iter 560 --steps 100
"""

from __future__ import annotations

import contextlib
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import slangpy as spy

_SCRIPTS = Path(__file__).resolve().parent.parent
for _sub in ("running", "debug", "plots"):
    _path = str(_SCRIPTS / _sub)
    if _path not in sys.path:
        sys.path.append(_path)

from trainer.backend.compute import gpu
from trainer.config.config import AdamLRs
from trainer.frontend.app import TrainingApplication
from trainer.frontend.cli import build_parser

FROZEN = AdamLRs(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)


def parse_args():
    parser = build_parser()
    parser.add_argument("--probe-iter", type=int, default=560)
    parser.add_argument("--steps", type=int, default=100)
    args = parser.parse_args()
    args.no_viewer = True
    args.preview_every = 0
    args.log_every = 0
    args.eval_every = 0
    args.iters = max(args.iters, args.probe_iter)
    return args


@contextlib.contextmanager
def stamped_train_steps(device, capacity: int):
    """Write a GPU timestamp at the start and end of every train_step batch."""
    pool = device.create_query_pool(spy.QueryType.timestamp, 2 * capacity)
    stamps = {"count": 0}
    new_batch, finish = gpu.GpuRuntime.command_batch, gpu.CommandBatch.finish

    def stamped_batch(runtime, label):
        batch = new_batch(runtime, label)
        if label == "train_step" and stamps["count"] < capacity:
            batch.stamp = 2 * stamps["count"]
            stamps["count"] += 1
            batch.encoder.write_timestamp(pool, batch.stamp)
        return batch

    def stamped_finish(batch):
        if getattr(batch, "stamp", None) is not None:
            batch.encoder.write_timestamp(pool, batch.stamp + 1)
        return finish(batch)

    gpu.GpuRuntime.command_batch, gpu.CommandBatch.finish = stamped_batch, stamped_finish
    try:
        yield lambda: np.asarray(pool.get_timestamp_results(0, 2 * stamps["count"])) * 1e3
    finally:
        gpu.GpuRuntime.command_batch, gpu.CommandBatch.finish = new_batch, finish


def run_steps(app, steps: int) -> tuple[float, float]:
    """Median host record+submit ms and mean pipelined wall ms per frozen step."""
    pipeline, scene = app.pipeline, app.scene_context
    limiter = app.runtime.queue_limiter
    record_ms = []
    pipeline.runtime.wait_for_idle()
    started = time.perf_counter()
    for step in range(steps):
        index = scene.train_indices[step % len(scene.train_indices)]
        pipeline.targets.select(app.target_textures[index])
        limiter.before_submit()
        recorded = time.perf_counter()
        pipeline.train_step(scene.colmapScene.images[index].camera, lrs=FROZEN)
        record_ms.append(1e3 * (time.perf_counter() - recorded))
        limiter.record(pipeline.last_submit_id)
    pipeline.runtime.wait_for_idle()
    return statistics.median(record_ms), 1e3 * (time.perf_counter() - started) / steps


def main() -> int:
    args = parse_args()
    app = TrainingApplication(args)
    app.prepare_training_context()
    app.build_training_loop().advance_to_capture(args.probe_iter)
    count = int(app.pipeline.buffers.primitive.active_count.to_numpy().view(np.uint32)[0])

    run_steps(app, 20)
    with stamped_train_steps(app.pipeline.device, args.steps) as read_stamps:
        host_ms, wall_ms = run_steps(app, args.steps)
        stamps = read_stamps()
    starts, ends = stamps[0::2], stamps[1::2]
    busy, idle = ends - starts, starts[1:] - ends[:-1]
    print(f"[overlap] iter {args.probe_iter}: {count:,} splats | host {host_ms:.2f} "
          f"wall {wall_ms:.2f} busy {busy.mean():.2f} idle {idle.mean():.3f} ms/step "
          f"(idle max {idle.max():.3f})", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
