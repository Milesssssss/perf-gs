# Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

"""Section 18: what the allocation actually looks like, on a real camera.

Two figures per camera, both built from the SHIPPED GPU path (not the host
reference) so what is shown is what trains:

  panel     render, ground truth, the raw screen-gradient magnitude s_i, the
            smoothed importance s~_i, the resulting p_i, the current UNIFORM STBN
            mask, and the adaptive STBN mask — all for the same camera and the
            same slice, so the last two differ only in the threshold.

  visits    the same camera sampled repeatedly, showing accumulated per-pixel
            visit counts for uniform STBN, adaptive STBN and adaptive IID. This is
            the figure that has to carry the central claim visually: high-importance
            regions get MORE samples while STBN keeps spreading them, rather than
            clumping the way independent draws do.

One probability colormap is used everywhere (``sampling.P_COLORMAP``), so a colour
means the same p in every figure of the report.

The state is reached by DENSE supervision and then frozen, exactly as the gradient
study does it, so the picture is a property of the scene rather than of whichever
regime happened to draw it.

RUNS AS ITS OWN PROCESS, and that is not incidental. It builds a full
TrainingApplication, which allocates ~15 GB on a 24 GB card at the default
``--max-splats``. Called in-process from the stage dispatcher, that allocation was
still resident when the next stage spawned a training cell wanting another 15 GB,
and the resulting over-subscription slowed the cell by 10x — a measurement that
looked exactly like "the adaptive method is slow" and was not. The whole suite
already runs GPU work in subprocesses for this reason; this was the one stage that
did not.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from paper_suite import figures as figs

from . import config as cfg
from .sampling import P_COLORMAP, P_VMAX, P_VMIN

# Visits used for the accumulation figure. 16 is a full quarter of the STBN
# volume's 64-slice period — long enough for the temporal spread to be visible,
# short enough that the uniform mask has not yet covered everything (at rho=0.25,
# 16 visits leave ~1% of pixels unvisited under independent sampling).
DEFAULT_VISITS = 16
DEFAULT_STATE = 1500
DEFAULT_CAMERAS = 2


def _linear_to_display(image: np.ndarray) -> np.ndarray:
    """RGB in [0,1] for imshow. The renderer composites in sRGB already (see
    shaders/losses/l1_l2_loss.slang), so this only clips."""
    return np.clip(np.asarray(image, dtype=np.float64)[..., :3], 0.0, 1.0)


def _target_rgb(texture) -> np.ndarray:
    """The ground truth as float RGB in [0,1].

    Target textures are ``rgba8_unorm``, so a readback is uint8 [0,255] while the
    shader sees it normalized — the dtype check mirrors what the test kit does
    rather than assuming one or the other.
    """
    values = np.asarray(texture.to_numpy())
    if values.dtype == np.uint8:
        values = values.astype(np.float64) / 255.0
    return np.asarray(values, dtype=np.float64)


def _percentile_norm(field: np.ndarray, high: float = 99.0) -> np.ndarray:
    """Scale to [0,1] by a high percentile rather than the max.

    s_i is extremely heavy-tailed — a handful of pixels sit orders of magnitude
    above the rest — so a max-normalized image is uniformly black with a few
    bright dots and shows nothing. The percentile is stated in the caption.
    """
    values = np.asarray(field, dtype=np.float64)
    top = float(np.percentile(values, high))
    return np.clip(values / max(top, 1e-12), 0.0, 1.0)


def run(paths: dict, *, python: str, scene: str = "bicycle",
        logs_dir: Path | None = None, force: bool = False,
        timeout_s: float = 3600.0) -> dict:
    """Render the figures in a SUBPROCESS, then load its record.

    See the module docstring for why the isolation is required rather than tidy.
    """
    import json as _json
    import subprocess

    record_path = paths["figures"] / f"visuals_{scene}.json"
    if record_path.is_file() and not force:
        print(f"[visuals] cached {record_path.name}", flush=True)
        try:
            return _json.loads(record_path.read_text())
        except ValueError:
            pass
    if cfg.scene_dir(scene) is None:
        return {"status": "skipped",
                "reason": f"scene {scene!r} not present under datasets/"}

    logs = logs_dir or paths["logs"]
    logs.mkdir(parents=True, exist_ok=True)
    stdout = logs / f"visuals_{scene}.stdout.log"
    stderr = logs / f"visuals_{scene}.stderr.log"
    # Re-enters through run.py rather than executing this file directly: run.py
    # owns the sys.path bootstrap and the package context that the relative
    # imports here need, so there is only one place that knows how to be started.
    argv = [python, str(cfg.REPO / "scripts" / "running" / "adaptive_stbn"
                        / "run.py"),
            "--stages", "visuals", "--visuals-in-process",
            "--visual-scene", scene, "--out", str(paths["root"])]
    try:
        with (stdout.open("w", encoding="utf-8", errors="replace") as out,
              stderr.open("w", encoding="utf-8", errors="replace") as err):
            proc = subprocess.run(argv, cwd=str(cfg.REPO), stdout=out,
                                  stderr=err, timeout=timeout_s)
        code = proc.returncode
    except (subprocess.TimeoutExpired, OSError,
            subprocess.SubprocessError) as error:
        return {"status": "failed", "error": f"{type(error).__name__}: {error}",
                "stderr": str(stderr.relative_to(cfg.REPO))}
    if code != 0 or not record_path.is_file():
        return {"status": "failed", "return_code": code,
                "stderr": str(stderr.relative_to(cfg.REPO))}
    try:
        return _json.loads(record_path.read_text())
    except ValueError as error:
        return {"status": "failed", "error": f"unreadable record: {error}"}


def render(paths: dict, *, scene: str = "bicycle", state: int = DEFAULT_STATE,
        cameras: int = DEFAULT_CAMERAS, visits: int = DEFAULT_VISITS,
        alpha: float = cfg.PRIMARY_ALPHA, sigma: float = cfg.PRIMARY_SIGMA,
        fraction: float = cfg.PRIMARY_FRACTION) -> dict:
    """Render the section-18 figures. Needs a GPU and the scene on disk."""
    directory = cfg.scene_dir(scene)
    if directory is None:
        return {"status": "skipped",
                "reason": f"scene {scene!r} not present under datasets/"}
    plt = figs.pyplot()
    if plt is None:
        return {"status": "skipped", "reason": "matplotlib unavailable"}

    import dataclasses

    from trainer.backend.render import adaptive_stbn as ad
    from trainer.backend.render import stbn as stbn_module
    from trainer.backend.render.adaptive_sampler import AdaptiveSampler
    from trainer.config.config import (DensificationProperties,
                                       IterationContext,
                                       IterationContextualVariables,
                                       StbnVariables)
    from trainer.contracts.codegen import TILE_AREA
    from trainer.frontend.app import TrainingApplication
    from trainer.frontend.cli import build_parser

    argv = [
        "--scene", str(directory),
        "--downsample", str(cfg.scene_downsample(scene)),
        "--seed", str(cfg.PRIMARY_SEED),
        "--iters", str(state),
        "--sh-degree", "3",
        "--densify-mode", "structure_aware",
        "--mip-splatting",
        "--eval", "1", "--eval-holdout-stride", "8", "--no-viewer",
        "--stbn-pixel-fraction", f"{fraction:.4f}",
        "--stbn-select-mode", "global", "--stbn-source", "nvidia",
        "--run-name", f"adaptive_visuals_{scene}",
    ]
    args = build_parser().parse_args(argv)
    for name, value in (("profile_run", False), ("capture_external", False),
                        ("preview_every", 0), ("log_every", 0),
                        ("eval_every", 0), ("eval_iters", None),
                        ("psnr_curve", False), ("render_all_every", 0),
                        ("final_test_psnr", False),
                        ("compute_final_metrics", False), ("run_report", False),
                        ("save_test_renders", False), ("stage_timing_every", 0),
                        ("write_ground_truths", False),
                        ("stbn_adaptive", False)):
        setattr(args, name, value)

    app = TrainingApplication(args)
    app.prepare_training_context()
    loop = app.build_training_loop()
    loop.stbn_density = None                 # advance DENSELY, as the study does
    pipeline = app._pipeline
    scene_context = app._sceneContext
    context = loop.advance_to_capture(state)
    contextual = context.iteration_contextual_variables

    cache = stbn_module.default_cache_dir() / "cache"
    budget = stbn_module.pixels_per_tile(fraction, TILE_AREA)
    enforced = budget / TILE_AREA
    uniform_table = stbn_module.upload_mask(
        pipeline.device,
        stbn_module.StbnMaskSpec(fraction=fraction,
                                 select_mode=stbn_module.SELECT_GLOBAL), cache)
    samplers = {}
    for uniform in (ad.UNIFORM_BLUE_TEMPORAL, ad.UNIFORM_IID):
        source = (stbn_module.SOURCE_WHITE if uniform == ad.UNIFORM_IID
                  else stbn_module.SOURCE_NVIDIA)
        samplers[uniform] = AdaptiveSampler(
            device=pipeline.device, width=scene_context.width,
            height=scene_context.height, kernels=pipeline, cache_dir=cache,
            spec=ad.AdaptiveSpec(fraction=fraction, alpha=alpha, sigma=sigma,
                                 downsample=cfg.PRIMARY_DOWNSAMPLE,
                                 uniform=uniform, source=source))

    def render_state(camera, variables, adaptive_uniform=None):
        pipeline._adaptive = (samplers[adaptive_uniform]
                             if adaptive_uniform is not None else None)
        pipeline._stbn = uniform_table
        step_context = IterationContext(
            step_number=pipeline.iteration + 1, camera_index=0,
            camera_entry=None, learning_rates=None, events=None,
            iteration_contextual_variables=IterationContextualVariables(
                hardness_value=contextual.hardness_value,
                sh_degree=contextual.sh_degree,
                contextual_stbn=variables,
                densification_variables=DensificationProperties(
                    structure_tensor_texture=None, structure_kwargs={})))
        inner = step_context.iteration_contextual_variables
        with pipeline.recording("adaptive_visuals"):
            pipeline._run_forward(camera, inner)
            pipeline._run_loss(inner)
        pipeline.runtime.wait_for_idle()
        return step_context

    train_indices = list(scene_context.train_indices)
    step = max(1, len(train_indices) // max(cameras, 1))
    picked = train_indices[::step][:cameras]

    written: list[Path] = []
    records: list[dict] = []
    for cam_idx in picked:
        pipeline.set_target_texture(app.target_textures[cam_idx])
        camera = scene_context.colmapScene.images[cam_idx].camera

        # 1. Dense pass: the render, and h_i for the whole image.
        render_state(camera, None)
        rendered = np.asarray(pipeline.buffers.output_image.to_numpy(),
                              dtype=np.float64)
        # ``_active_target``, NOT buffers.target_image: set_target_texture swaps
        # the reference to this camera's cached rgba8_unorm texture, so reading the
        # shared scratch one would show whatever was uploaded last.
        target = _target_rgb(pipeline._active_target)
        gradient = np.asarray(pipeline.buffers.dL_dC_texture.to_numpy(),
                              dtype=np.float64)
        s = ad.screen_importance(gradient)
        spec = ad.AdaptiveSpec(fraction=fraction, alpha=alpha, sigma=sigma,
                               downsample=cfg.PRIMARY_DOWNSAMPLE)
        s_tilde = ad.smooth_importance(s, spec)

        # 2. Adaptive pass at one slice: p and the mask, from the GPU.
        slice_index = 7
        render_state(camera, StbnVariables(slice_index=slice_index,
                                          pixels_per_tile=budget,
                                          sample_seed=1),
                     adaptive_uniform=ad.UNIFORM_BLUE_TEMPORAL)
        sampler = samplers[ad.UNIFORM_BLUE_TEMPORAL]
        p = sampler.read_probabilities()
        adaptive_mask = sampler.read_mask()

        # 3. The uniform STBN mask at the SAME slice, so the two masks differ only
        #    in the threshold they compared the identical U field against.
        uniform_mask = ad.wrap_to_screen(
            stbn_module.global_selected_mask(
                dataclasses.replace(stbn_module.StbnMaskSpec(),
                                    fraction=fraction))[slice_index],
            scene_context.height, scene_context.width)

        written += _panel(plt, paths, scene, cam_idx, state,
                          rendered=rendered, target=target, s=s,
                          s_tilde=s_tilde, p=p, uniform_mask=uniform_mask,
                          adaptive_mask=adaptive_mask, alpha=alpha, sigma=sigma,
                          rho=enforced)

        counts = _visit_counts(render_state, camera, samplers, uniform_table,
                               budget, visits, scene_context, ad, stbn_module,
                               dataclasses, fraction, StbnVariables)
        written += _visits_figure(plt, paths, scene, cam_idx, state, p, counts,
                                  visits, enforced)
        records.append({
            "scene": scene, "camera_id": int(cam_idx), "checkpoint": state,
            "p_mean": float(p.mean()), "p_min": float(p.min()),
            "p_max": float(p.max()),
            "saturated_fraction": float((p >= 1.0 - 1e-6).mean()),
            "uniform_selected_fraction": float(uniform_mask.mean()),
            "adaptive_selected_fraction": float(adaptive_mask.mean()),
            "visit_count_correlation_with_p": {
                name: float(np.corrcoef(p.reshape(-1),
                                        count.reshape(-1))[0, 1])
                for name, count in counts.items()},
        })
        print(f"[visuals] {scene} cam {cam_idx}: p in [{p.min():.4f}, "
              f"{p.max():.4f}] mean {p.mean():.6f}; adaptive selected "
              f"{adaptive_mask.mean():.4f} vs uniform {uniform_mask.mean():.4f}",
              flush=True)

    record = {
        "status": "ok", "scene": scene, "checkpoint": state,
        "cameras": [int(c) for c in picked], "visits": visits,
        "alpha": alpha, "sigma": sigma, "rho_enforced": enforced,
        "colormap": P_COLORMAP, "p_color_range": [P_VMIN, P_VMAX],
        "per_camera": records,
        "figures": [str(p.relative_to(paths["root"])) for p in written],
        "note": ("s_i and s~_i panels are normalized by their 99th percentile, "
                 "not their maximum: s is heavy-tailed enough that a "
                 "max-normalized image is black with a few dots."),
    }
    (paths["figures"] / f"visuals_{scene}.json").write_text(
        json.dumps(record, indent=2))
    return record


def _visit_counts(render_state, camera, samplers, uniform_table, budget, visits,
                  scene_context, ad, stbn_module, dataclasses, fraction,
                  StbnVariables) -> dict:
    """Accumulated per-pixel selection counts over ``visits`` repeat visits.

    The uniform arm is taken from the mask table directly rather than by running
    the GPU pass ``visits`` times: it does not depend on the render, so re-running
    the forward for it would cost minutes and change nothing.
    """
    height, width = scene_context.height, scene_context.width
    uniform_volume = stbn_module.global_selected_mask(
        dataclasses.replace(stbn_module.StbnMaskSpec(), fraction=fraction))
    slices = uniform_volume.shape[0]
    counts = {"uniform_stbn": np.zeros((height, width), dtype=np.int32),
              "adaptive_stbn": np.zeros((height, width), dtype=np.int32),
              "adaptive_iid": np.zeros((height, width), dtype=np.int32)}
    for visit in range(visits):
        counts["uniform_stbn"] += ad.wrap_to_screen(
            uniform_volume[visit % slices], height, width).astype(np.int32)
        for name, uniform in (("adaptive_stbn", ad.UNIFORM_BLUE_TEMPORAL),
                              ("adaptive_iid", ad.UNIFORM_IID)):
            render_state(camera,
                         StbnVariables(slice_index=visit,
                                       pixels_per_tile=budget,
                                       sample_seed=visit * 7919 + 13),
                         adaptive_uniform=uniform)
            counts[name] += samplers[uniform].read_mask().astype(np.int32)
    return counts


def _panel(plt, paths, scene, cam_idx, state, *, rendered, target, s, s_tilde, p,
           uniform_mask, adaptive_mask, alpha, sigma, rho) -> list[Path]:
    fig, axes = plt.subplots(1, 7, figsize=(figs.FIG_WIDTH_DOUBLE * 1.6, 2.0))
    panels = [
        ("render", _linear_to_display(rendered), None, {}),
        ("ground truth", _linear_to_display(target), None, {}),
        (r"$s_i=\|h_i\|_2$", _percentile_norm(s), "gray", {}),
        (rf"$\tilde s_i$ ($\sigma={sigma:g}$)", _percentile_norm(s_tilde), "gray",
         {}),
        # Absolute [0,1], not this camera's own range: the point of a shared
        # colormap is that two panels showing different allocations look different.
        (rf"$p_i$ ($\alpha={alpha:g}$)", p, P_COLORMAP,
         {"vmin": P_VMIN, "vmax": P_VMAX}),
        (rf"uniform STBN mask ({uniform_mask.mean():.3f})",
         uniform_mask.astype(float), "gray", {"vmin": 0.0, "vmax": 1.0}),
        (rf"adaptive STBN mask ({adaptive_mask.mean():.3f})",
         adaptive_mask.astype(float), "gray", {"vmin": 0.0, "vmax": 1.0}),
    ]
    image = None
    for ax, (title, data, cmap, kwargs) in zip(axes, panels):
        handle = ax.imshow(data, cmap=cmap, interpolation="nearest", **kwargs)
        if cmap == P_COLORMAP:
            image = handle
        ax.set_title(title, fontsize=7)
        ax.set_xticks([])
        ax.set_yticks([])
    if image is not None:
        fig.colorbar(image, ax=axes[4], shrink=0.8, label=r"$p_i$")
    fig.suptitle(f"{scene} camera {cam_idx} at iteration {state}, "
                 rf"$\rho={rho:g}$", fontsize=8)
    return figs.save(fig, paths["figures"], f"visual_panel_{scene}_cam{cam_idx}")


def _visits_figure(plt, paths, scene, cam_idx, state, p, counts, visits,
                   rho) -> list[Path]:
    """Accumulated visit counts, plus the count-vs-p relation that quantifies it.

    The scatter panel is the one that makes the claim checkable rather than
    suggestive: if the allocation is working, mean visit count must rise linearly
    with p and sit on the line ``visits * p``.
    """
    names = ["uniform_stbn", "adaptive_stbn", "adaptive_iid"]
    top = max(int(counts[n].max()) for n in names)
    fig, axes = plt.subplots(1, 4, figsize=(figs.FIG_WIDTH_DOUBLE, 2.2))
    for ax, name in zip(axes[:3], names):
        handle = ax.imshow(counts[name], cmap="viridis", vmin=0, vmax=top,
                           interpolation="nearest")
        ax.set_title(f"{name.replace('_', ' ')}\n"
                     f"mean {counts[name].mean():.2f}", fontsize=7)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.colorbar(handle, ax=axes[2], shrink=0.8,
                 label=f"visits selected (of {visits})")

    ax = axes[3]
    order = np.argsort(p.reshape(-1), kind="stable")
    sorted_p = p.reshape(-1)[order]
    edges = np.linspace(0, sorted_p.size, 13).astype(np.int64)
    for name in names:
        flat = counts[name].reshape(-1)[order]
        centers, means = [], []
        for low, high in zip(edges[:-1], edges[1:]):
            if high <= low:
                continue
            centers.append(float(sorted_p[low:high].mean()))
            means.append(float(flat[low:high].mean()))
        ax.plot(centers, means, marker="o", markersize=3, linewidth=1.0,
                label=name.replace("_", " "))
    ax.plot([float(sorted_p.min()), float(sorted_p.max())],
            [visits * float(sorted_p.min()), visits * float(sorted_p.max())],
            color="#000000", linewidth=0.8, linestyle=":",
            label=r"$K p_i$ (ideal)")
    ax.axhline(visits * rho, color="#999999", linewidth=0.6)
    ax.set_xlabel(r"$p_i$")
    ax.set_ylabel(f"mean visits (of {visits})")
    ax.legend(fontsize=6)
    fig.suptitle(f"{scene} camera {cam_idx}: {visits} repeat visits at "
                 rf"$\rho={rho:g}$", fontsize=8)
    return figs.save(fig, paths["figures"], f"visual_visits_{scene}_cam{cam_idx}")

