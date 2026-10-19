# Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

from __future__ import annotations

import hashlib
import json
import platform
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Sequence

import numpy as np
from PIL import Image

from trainer.common.metrics import psnr as psnr_image, ssim as ssim_image

if TYPE_CHECKING:
    import argparse

    from trainer.backend.render.pipeline import TrainingPipeline

SCHEMA_VERSION = 3

# Render-timing sample count for the reported FPS. Enough to average out a
# stray hitch, short enough to stay a rounding error on a 3k-iteration run.
DEFAULT_FPS_FRAMES = 60
FPS_WARMUP_FRAMES = 5



def device_info(pipeline: "TrainingPipeline") -> dict:

    info: dict[str, Any] = {}
    try:
        device = pipeline.device
    except Exception:                                      # noqa: BLE001
        return {"error": "device unavailable"}

    raw = getattr(device, "info", None)
    for field in ("adapter_name", "api_name", "type", "timestamp_frequency",
                  "optix_version"):
        try:
            value = getattr(raw, field, None)
            info[field] = value if isinstance(
                value, (int, float, str, type(None))) else str(value)
        except Exception:                                  # noqa: BLE001
            info[field] = None

    limits = getattr(raw, "limits", None)
    if limits is not None:
        picked = {}
        for field in ("max_compute_dispatch_thread_groups_x",
                      "max_compute_thread_group_size",
                      "max_shader_visible_samplers",
                      "constant_buffer_offset_alignment"):
            try:
                value = getattr(limits, field, None)
                if value is not None:
                    picked[field] = (value if isinstance(value, (int, float, str))
                                     else str(value))
            except Exception:                              # noqa: BLE001
                pass
        if picked:
            info["limits"] = picked

    try:
        info["supported_shader_model"] = str(device.supported_shader_model)
    except Exception:                                      # noqa: BLE001
        pass
    try:
        # Feature list pins what the backend actually offered this run.
        info["features"] = sorted(str(f) for f in device.features)
    except Exception:                                      # noqa: BLE001
        pass
    return info


def _buffer_bytes(obj: Any) -> int:
    """Byte size of an NDBuffer / Buffer, or 0 for anything else.

    Defensive by design: this is accounting, and a slangpy API shift must never
    be able to fail a training run that has already succeeded.
    """
    for path in (("storage", "desc", "size"), ("desc", "size"), ("size",)):
        node = obj
        try:
            for attr in path:
                node = getattr(node, attr)
            if isinstance(node, int) and node > 0:
                return node
        except Exception:                                  # noqa: BLE001
            continue
    return 0


def _texture_bytes(tex: Any) -> int:
    try:
        desc = tex.desc
        width = int(getattr(desc, "width", 0) or 0)
        height = int(getattr(desc, "height", 0) or 0)
        # 4 channels x 4 bytes is the worst case among the formats we create;
        # this is a labelled upper bound, not a claim of exactness.
        return width * height * 16
    except Exception:                                      # noqa: BLE001
        return 0


def gpu_allocation(pipeline: "TrainingPipeline") -> dict:
    """Sum of the trainer's OWN device allocations, by dataclass field.

    This is the reproducible half of "peak GPU memory": it depends only on
    ``--max-splats``, the frame size and the SH degree, so two runs of the same
    config report the same number on any machine. It excludes driver/runtime
    overhead and any allocation the trainer does not own — the suite driver polls
    the device-wide figure separately for that. Reported as both, never merged.
    """
    result: dict[str, Any] = {"buffers_bytes": 0, "textures_bytes": 0,
                              "by_field": {}}
    try:
        buffers = pipeline.buffers
    except Exception:                                      # noqa: BLE001
        return result

    total_buf = 0
    by_field: dict[str, int] = {}
    for name in dir(buffers):
        if name.startswith("_"):
            continue
        try:
            value = getattr(buffers, name)
        except Exception:                                  # noqa: BLE001
            continue
        if callable(value):
            continue
        size = 0
        if isinstance(value, dict):
            size = sum(_buffer_bytes(v) for v in value.values())
        else:
            size = _buffer_bytes(value)
        if size > 0:
            by_field[name] = size
            total_buf += size

    tex_total = 0
    for attr in ("target_texture", "output_texture", "structure_texture"):
        tex = getattr(buffers, attr, None)
        if tex is not None:
            tex_total += _texture_bytes(tex)

    result["buffers_bytes"] = int(total_buf)
    result["textures_bytes"] = int(tex_total)
    result["total_bytes"] = int(total_buf + tex_total)
    result["total_mb"] = round((total_buf + tex_total) / (1024 * 1024), 2)
    result["by_field"] = dict(sorted(by_field.items(), key=lambda kv: -kv[1]))
    result["note"] = (
        "trainer-owned allocations only, walked from GaussianBuffers; excludes "
        "driver/runtime overhead. textures_bytes is an upper bound (16 B/px). "
        "READ THIS AS CAPACITY, NOT USE: the per-splat buffers are sized from "
        "--max-splats up front, so this figure is a constant of the config and "
        "is identical across sampling regimes. Divide by max_splats for "
        "bytes-per-splat, and multiply by the achieved n_gaussians for the "
        "working set actually touched.")
    return result


def stbn_asset_fingerprint(args: "argparse.Namespace") -> dict:
    """Identity of the sampling asset this run actually read.

    The STBN volume is a downloaded, non-vendored, NON-COMMERCIAL asset, so
    "which STBN" is a real experimental variable: the report pins it by content
    hash, not by filename.
    """
    from trainer.backend.render import stbn as stbn_module

    source = str(getattr(args, "stbn_source", ""))
    record: dict[str, Any] = {
        "source": source,
        "pixel_fraction": float(getattr(args, "stbn_pixel_fraction", 1.0)),
        "warmup_iters": int(getattr(args, "stbn_warmup_iters", 0)),
        "independent": bool(getattr(args, "stbn_independent", False)),
        "stbn_seed": int(getattr(args, "stbn_seed", 0)),
        "tile_pixels": stbn_module.pixels_per_tile(
            float(getattr(args, "stbn_pixel_fraction", 1.0))),
        "tile_area": stbn_module.TILE_AREA,
    }
    if source == stbn_module.SOURCE_WHITE:
        record["asset"] = "generated (per-tile permutations from stbn_seed)"
        return record

    override = str(getattr(args, "stbn_file", "") or "")
    path = (Path(override) if override
            else stbn_module.nvidia_volume_path())
    record["asset_path"] = str(path)
    try:
        data = path.read_bytes()
        record["asset_sha256"] = hashlib.sha256(data).hexdigest()
        record["asset_bytes"] = len(data)
        record["asset_name"] = path.name
    except OSError as error:
        record["asset_error"] = f"{type(error).__name__}: {error}"
    return record


def adaptive_sampling_record(pipeline: "TrainingPipeline") -> dict:
    """What the importance-guided sampler actually did, or ``{"enabled": False}``.

    Two halves, and the split is the point. ``config`` is what was ASKED for; the
    rest is what was MEASURED on the device — the realized mean/min/max p, the
    saturated fraction, and the actually-selected pixel count. The budget claim
    ``(1/N) sum p_i = rho`` is therefore verified per run rather than asserted by
    the solver, and ``budget_error_max`` is the worst single iteration.

    Draining the per-iteration stats needs a device readback, so this runs once,
    after the timed loop — never inside it.
    """
    sampler = getattr(pipeline, "adaptive_sampler", None)
    if sampler is None:
        return {"enabled": False}
    try:
        return {"enabled": True, "config": sampler.config_record(),
                "measured": sampler.stats_summary()}
    except Exception as error:                             # noqa: BLE001
        return {"enabled": True, "error": f"{type(error).__name__}: {error}"}


def process_environment() -> dict:
    """Interpreter + numeric-stack versions, for the manifest's reproducibility
    block. Version drift in numpy/scipy can move SSIM in the last digits, so it
    is recorded rather than assumed."""
    versions: dict[str, str] = {}
    for module in ("numpy", "scipy", "slangpy", "PIL"):
        try:
            versions[module] = str(__import__(module).__version__)
        except Exception:                                  # noqa: BLE001
            versions[module] = "unavailable"
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "executable": sys.executable,
        "packages": versions,
    }


# ---------------------------------------------------------------------------
# Held-out per-view metrics + render dump
# ---------------------------------------------------------------------------

def evaluate_test_views(
    pipeline: "TrainingPipeline",
    scene,
    test_indices: Sequence[int],
    targets_srgb: Sequence[np.ndarray],
    *,
    render_dir: Path | None,
    with_lpips: bool = False,
    progress: Callable[[int, int], None] = lambda _d, _t: None,
) -> tuple[list[dict], dict]:
    """Per-view PSNR/SSIM over the held-out split, optionally dumping renders.

    Metrics are computed from the FLOAT render (exactly as the existing final
    metrics pass does) — not from the PNG — so the 8-bit round trip cannot move
    a reported dB. The PNG is written for downstream perceptual metrics and
    qualitative figures, which are 8-bit metrics anyway.
    """
    pipeline.runtime.wait_for_idle()
    if render_dir is not None:
        render_dir.mkdir(parents=True, exist_ok=True)

    lpips_ready = False
    if with_lpips:
        from trainer.common import perceptual
        lpips_ready = perceptual.load()

    per_view: list[dict] = []
    total = max(1, len(test_indices))
    for done, cam_i in enumerate(test_indices, start=1):
        progress(done, total)
        entry = scene.images[cam_i]
        pipeline.render(entry.camera)
        rendered = np.clip(pipeline.read_output_image()[..., :3], 0.0, 1.0)
        target = targets_srgb[cam_i][..., :3]

        record = {
            "cam_idx": int(cam_i),
            "image_name": entry.image_name,
            "psnr_srgb": psnr_image(rendered, target),
            "ssim_srgb": ssim_image(rendered, target),
            "height": int(rendered.shape[0]),
            "width": int(rendered.shape[1]),
        }
        # Computed from the same float render as PSNR/SSIM, before any PNG
        # round trip, so all three metrics describe one identical image.
        record["lpips"] = (perceptual.lpips(rendered, target)
                                if lpips_ready else None)
        if render_dir is not None:
            name = f"cam_{cam_i:04d}.png"
            Image.fromarray((rendered * 255.0 + 0.5).astype(np.uint8),
                            mode="RGB").save(render_dir / name)
            record["render_png"] = name
        per_view.append(record)

    def _mean(key: str) -> float | None:
        values = [r[key] for r in per_view if r.get(key) is not None]
        return float(np.mean(values)) if values else None

    def _std(key: str) -> float | None:
        values = [r[key] for r in per_view if r.get(key) is not None]
        return float(np.std(values)) if values else None

    summary = {
        # Paper metric: the MEAN of per-image scores over the held-out split.
        "psnr_srgb": _mean("psnr_srgb"),
        "ssim_srgb": _mean("ssim_srgb"),
        "lpips": _mean("lpips"),
        # Spread ACROSS VIEWS of one run — not a seed-to-seed uncertainty. The
        # suite computes the across-seed std separately; conflating them would
        # overstate what a single run establishes.
        "psnr_srgb_std_views": _std("psnr_srgb"),
        "ssim_srgb_std_views": _std("ssim_srgb"),
        "lpips_std_views": _std("lpips"),
        "n_test_views": len(per_view),
        "n_lpips_views": sum(1 for r in per_view
                             if r.get("lpips") is not None),
        "renders_dir": (str(render_dir) if render_dir is not None else None),
    }
    if with_lpips:
        from trainer.common import perceptual
        summary["lpips_backend"] = perceptual.availability()
    return per_view, summary


def measure_render_fps(pipeline: "TrainingPipeline", scene,
                       cam_idx: int, frames: int) -> dict:
    """Steady-state render throughput for ONE held-out camera.

    Serialized on purpose: each frame waits for the GPU, so this is per-frame
    latency (1/latency FPS), not pipelined throughput. That is the number a
    "render FPS" column should mean, and it is the one that stays comparable
    between a dense and a sparse run — the sparse mask only touches the backward,
    so any large divergence here is a red flag rather than a result.
    """
    if frames <= 0:
        return {"frames": 0, "fps": None, "ms_per_frame": None}
    camera = scene.images[cam_idx].camera
    for _ in range(FPS_WARMUP_FRAMES):
        pipeline.render(camera)
    pipeline.runtime.wait_for_idle()

    started = time.perf_counter()
    for _ in range(frames):
        pipeline.render(camera)
        pipeline.runtime.wait_for_idle()
    elapsed = time.perf_counter() - started

    per_frame = elapsed / frames
    return {
        "frames": int(frames),
        "cam_idx": int(cam_idx),
        "fps": (1.0 / per_frame) if per_frame > 0 else None,
        "ms_per_frame": per_frame * 1e3,
        "total_s": elapsed,
        "mode": "serialized (wait_for_idle per frame)",
    }


# ---------------------------------------------------------------------------
# Report assembly
# ---------------------------------------------------------------------------

def build_report(
    *,
    pipeline: "TrainingPipeline",
    scene_context,
    args: "argparse.Namespace",
    run_name: str,
    result,
    per_view: list[dict],
    quality: dict,
    render: dict,
    loop_timing: dict,
    process_started_at: float,
) -> dict:
    """Assemble the full record. Pure aggregation — no GPU work."""
    wall = float(getattr(result, "wall_time_seconds", 0.0) or 0.0)
    observation = float(loop_timing.get("observation_s", 0.0) or 0.0)
    completed = int(getattr(result, "iterations", 0) or 0)
    active = max(0.0, wall - observation)

    n_gaussians = None
    try:
        n_gaussians = int(pipeline.buffers.num_gaussians_counter
                          .to_numpy().view(np.uint32)[0])
    except Exception:                                      # noqa: BLE001
        pass

    return {
        "schema": SCHEMA_VERSION,
        "run_name": run_name,
        "scene_path": str(args.scene),
        "iters_requested": int(args.iters),
        "iters_completed": completed,
        "timing": {
            # Wall-clock across the loop, frozen at the last iteration with the
            # GPU drained (identical to test_psnr.json's train_wall_s).
            "train_wall_s": wall,
            # The number sparse-vs-dense speed claims must use.
            "train_active_s": active,
            # In-window observation cost: the part contained in train_wall_s and
            # therefore the only part subtracted from it.
            "observation_s": observation,
            # Every observation, including the post-freeze final sweep. Reported
            # so the two can be reconciled instead of appearing to disagree.
            "observation_total_s": float(
                loop_timing.get("observation_total_s", observation) or 0.0),
            "iters_per_s_active": (completed / active) if active > 0 else None,
            "ms_per_iter_active": (active * 1e3 / completed) if completed else None,
            "process_wall_s": time.time() - process_started_at,
            "note": ("train_active_s = train_wall_s - observation_s; "
                     "observation_s is every mid-loop eval/psnr-curve/log pass. "
                     "Runs with no mid-loop evaluation have observation_s == 0."),
        },
        "quality": {**quality, "per_view": per_view},
        "model": {
            "n_gaussians": n_gaussians,
            "sh_degree": int(args.sh_degree),
            "max_splats": int(args.max_splats),
        },
        "render": render,
        "memory": gpu_allocation(pipeline),
        "device": device_info(pipeline),
        "scene": {
            "width": int(scene_context.width),
            "height": int(scene_context.height),
            "downsample": int(args.downsample),
            "n_cams": int(scene_context.camera_count),
            "n_train": len(scene_context.train_indices),
            "n_test": len(scene_context.test_indices),
            "holdout_stride": int(scene_context.holdout_stride),
            "eval_holdout": bool(args.eval_holdout),
            "split_rule": (
                f"test = cameras where (name-sorted index) % "
                f"{scene_context.holdout_stride} == 0"
                if args.eval_holdout else "no holdout (train == test == all)"),
            "test_indices": [int(i) for i in scene_context.test_indices],
            "n_init_points": int(scene_context.initial_gaussian_count),
            "extent": float(scene_context.extent),
        },
        "stbn": stbn_asset_fingerprint(args),
        "adaptive_stbn": adaptive_sampling_record(pipeline),
        "config": {
            "densify_mode": str(getattr(args, "densify_mode", "")),
            "mip_splatting": bool(args.mip_splatting),
            "seed": int(args.seed),
            "sample_mode": str(args.sample_mode),
            "lambda_ssim": float(args.lambda_ssim),
            "backend": str(args.backend),
            "flat_bucket_backward": bool(args.flat_bucket_backward),
            "backward_pixel_stride": int(args.backward_pixel_stride),
        },
        "argv": list(sys.argv),
        "cli_args": {k: (v if isinstance(v, (int, float, str, bool, type(None)))
                         else str(v))
                     for k, v in vars(args).items()},
        "environment": process_environment(),
    }


def write_run_report(report: dict, run_dir: Path) -> Path:
    path = run_dir / "run_report.json"
    path.write_text(json.dumps(report, indent=2))
    return path
