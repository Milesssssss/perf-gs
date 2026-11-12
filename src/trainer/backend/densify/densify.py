# Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

"""Simple/structure-aware densification, Morton sorting, and opacity reset.

These operate ON a :class:`~trainer.pipeline.Pipeline`: they dispatch its
kernels, read/write its buffer bag, and use its ``_scan`` / ``record_forward`` /
encoder infrastructure. Kept as free functions taking the pipeline as the first
argument — mirroring how :mod:`trainer.buffer_io` takes a ``GaussianBuffers`` —
so the Pipeline class stays lean; the Pipeline exposes them as thin delegating
methods (no more ``Pipeline.densify = densify`` monkey-patch).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import slangpy as spy

from trainer.backend.memory.reorder import reorder_soa_planes

if TYPE_CHECKING:
    from trainer.backend.render.pipeline import TrainingPipeline


@dataclass
class DensifyConfig:
    """Shared simple-densify and opacity-reset thresholds."""
    t_opacity: float = 0.005        # prune if sigmoid(opacity) < this. MUST stay BELOW
                                    # t_opacity_reset_value (0.01): reset clamps opacity down
                                    # to the reset value, so a floor ABOVE it prunes every
                                    # splat the instant reset fires (total wipeout). 3DGS uses
                                    # prune 0.005 < reset 0.01 so reset splats survive and only
                                    # genuinely-dead ones decaying below 0.005 get culled.
    t_big_scale: float = 0.1        # prune if max_scale > this (world-space floater removal)
    t_opacity_reset_value: float = 0.01  # reset opacity to this value (CLAMP mode)
    # Opacity-reset mode. False (default): gsplat CLAMP — opacity <- min(opacity,
    # logit(t_opacity_reset_value)). True: SADGS RELATIVE decay — opacity <-
    # opacity * reset_decay (preserves strong/weak ordering; reset_decay is the
    # multiplier, e.g. 0.1). SADGS reset_opacity(opacity_reset_decay=0.1).
    reset_relative: bool = False
    reset_decay: float = 0.1        # RELATIVE-mode multiplier (used iff reset_relative)


@dataclass
class StructureConfig:
    """Structure-aware densification config (SAD-GS, arXiv 2604.28016).

    Matches SADGS train.py gates:
      split_mask = (high_ratio > split_ratio_threshold) & is_grad_high
      prune_mask = (low_ratio  > prune_ratio_threshold) & valid
    where high_ratio/low_ratio are the SCALAR (max-over-axes eta) view
    fractions, and the split factor n_k = ceil(eta_max_k ** split_power)."""
    n_scoring_views: int = 10        # views sampled per OFFLINE scoring step (tests/fallback)
    tau_split: float = 0.8           # split_ratio_threshold: high_ratio must exceed this
    tau_prune: float = 0.8           # prune_ratio_threshold: low_ratio must exceed this
    grad_threshold: float = 1e-5     # is_grad_high: avg 2D-grad magnitude floor (SADGS 1e-5)
    split_power: float = 0.5         # n_k = ceil(eta_max_k ** split_power); 0.5 = sqrt (paper)
    max_split_factor: int = 8        # per-axis cap on n_k (SADGS max_clones_per_axis=8)
    # Child scale along axis k = sigma_k / n_k**ks_scale_power (SADGS
    # scaling / ks**scale_power). 1.0 for Mip-NeRF 360 (the variance-preserving
    # choice paired with the sqrt(12) child spacing in split_anisotropic.slang);
    # 1.2 for Tanks & Temples / Deep Blending (SADGS ks_scale_power=1.2).
    ks_scale_power: float = 1.0
    t_opacity: float = 0.005
    # Post-densify opacity cap (SADGS gaussian_model.py:1046:
    # inverse_sigmoid(min(get_opacity, 0.8)) applied to ALL survivors after the
    # structure prune)
    opacity_cap: float = 0.8
    # Scale threshold separating clone (small) from split (large), = SADGS
    # args.dense * scene_extent. A triggered gaussian with max world-space scale
    # <= this is CLONED (exact duplicate); larger ones are anisotropically split.
    dense_scale: float = 0.0         # dense * extent, in world units; set by the caller
    detailed_diagnostics: bool = False  # full eta readback only when explicitly requested


def densify_simple(pipeline: TrainingPipeline, cfg: DensifyConfig,
                   grad_thresh: float = 0.0002,
                   scale_thresh: float | None = None,
                   prune_big_scale: float | None = None,
                   allow_scale_prune: bool = True,
                   carry_eta: bool = False) -> dict:
    """Compatibility shim; the implementation lives in ``SimpleDensifier``."""
    if pipeline.simple_densifier is None:
        raise RuntimeError("densification is unavailable in viewer-only mode")
    return pipeline.simple_densifier.run(
        cfg, grad_thresh, scale_thresh, prune_big_scale,
        allow_scale_prune, carry_eta)


def reset_opacity(pipeline: TrainingPipeline, cfg: DensifyConfig) -> None:
    """Per-gaussian opacity reset. CLAMP mode (default): opacity <-
    min(opacity, logit(t_opacity_reset_value)). RELATIVE mode (SADGS): opacity
    <- opacity * reset_decay. Forces the optimizer to re-earn opacity on real
    geometry; RELATIVE preserves the strong/weak ranking through the reset."""
    b = pipeline.buffers
    relative = getattr(cfg, "reset_relative", False)
    pipeline.dispatch(pipeline.k_opacity_reset, "opacity_reset",
        thread_count=[pipeline.max_splats, 1, 1],
        gaussians=b.gaussians.storage,
        adam_state=b.adam_state.storage,
        num_gaussians_counter=b.num_gaussians_counter.storage,
        reset_value=(cfg.reset_decay if relative else cfg.t_opacity_reset_value),
        mode=(1 if relative else 0),
    )


def clear_structure_scores(pipeline: TrainingPipeline) -> None:
    """Zero the per-axis eta accumulators. Call at the start of each
    accumulation window (right after a densify reorders gaussian slots), so
    the multiview statistics are gathered over a stable gaussian set."""
    b = pipeline.buffers
    pipeline.dispatch(pipeline.k_clear_structure_scores, "clear_structure_scores",
        thread_count=[pipeline.max_splats, 1, 1],
        **b.eta_kwargs(),
        eta_n_total=b.eta_n_total.storage,
        max_splats=pipeline.max_splats,
    )


def accumulate_structure_scores(pipeline: TrainingPipeline, structure_tensor,
                                pass_idx: int | None = None,
                                opacity_threshold: float = 0.05,
                                per_view_grad_thresh: float = 0.0) -> None:
    """Fold ONE per-gaussian eta sample into the accumulators for the current
    camera (already uploaded to ``pipeline.camera_buffer`` by the forward
    pass). Called once per training iteration from train_step to build the
    N_high/N_low/eta_max multiview statistics online (paper Section 3.6),
    which is what makes the N_high/N_total fraction gate meaningful — a handful
    of snapshot views can't resolve it. Runs inside the active encoder; no sync.

    MUST run AFTER backward (which fills grad_2d_x/y_iter) and BEFORE
    accumulate_iter (which clears them), so the per-view grad gate below sees
    this view's gradient.

    is_active_vis GATES (SADGS): eta is only accumulated for gaussians that
    (a) rasterized this view (tile_count_per_gaussian > 0), (b) have opacity >
    ``opacity_threshold``, and (c) — when ``per_view_grad_thresh`` > 0 — have a
    high per-view screen gradient (SADGS freq_grad_threshold). Without these
    the high fraction saturates (~everywhere >0.8), the running eta_max
    inflates, and densification over-splits / explodes.

    ``per_view_grad_thresh`` is the NDC-convention floor the shader compares the
    per-view grad norm against; the host pre-scales it (see the train loop).
    0 disables gate (c) — used by the offline/test path which has no live grad.

    ``pass_idx`` decorrelates the jitter sample per call; defaults to the
    training iteration so successive samples of the same gaussian differ."""
    b = pipeline.buffers
    if pass_idx is None:
        pass_idx = pipeline.iteration
    pipeline.dispatch(pipeline.k_score_structure_aware, "score_structure_aware",
        thread_count=[pipeline.max_splats, 1, 1],
        gaussians=b.gaussians.storage,
        num_gaussians_counter=b.num_gaussians_counter.storage,
        camera=pipeline.camera_buffer,
        structure_tensor=structure_tensor,
        tile_count_per_gaussian=b.tile_count_per_gaussian.storage,
        grad_2d_x_iter=b.grad_2d_x_iter.storage,
        grad_2d_y_iter=b.grad_2d_y_iter.storage,
        **b.eta_kwargs(),
        eta_n_total=b.eta_n_total.storage,
        opacity_threshold=opacity_threshold,
        per_view_grad_thresh=float(per_view_grad_thresh),
        pass_idx=int(pass_idx) & 0xFFFF,
        max_splats=pipeline.max_splats,
    )


def compute_structure_scores(
    pipeline: TrainingPipeline,
    cameras: list,
    structure_textures: list,
    cfg: StructureConfig,
) -> None:
    """Offline multi-view eta accumulation: clear, then sample each of N views
    once. Used by tests and as a fallback; the production path accumulates
    online per training iteration (see accumulate_structure_scores) so the
    fraction gate has enough samples to converge."""
    clear_structure_scores(pipeline)
    for view_idx, (cam, structure_tex) in enumerate(zip(cameras, structure_textures)):
        # Run the forward pass so tile_count_per_gaussian reflects THIS view's
        # visibility (the eta kernel gates on it). Also uploads the camera.
        pipeline.rasterize(cam)
        accumulate_structure_scores(pipeline, structure_tex, pass_idx=view_idx)

def densify_structure_aware(pipeline: TrainingPipeline, cfg: StructureConfig) -> dict:

    b = pipeline.buffers

    # Convention scaling for the grad gate — identical to densify_simple.
    # grad_mean_2d_mag_accum holds a per-pixel SUM of a no-1/N loss, so the
    # SADGS mean-loss threshold must be scaled up by W*H*channels here.
    grad_thresh_internal = cfg.grad_threshold * (pipeline.width * pipeline.height * 3.0)

    # kwargs for the single-pass structure decide (densify_children_sum is
    # passed separately at the dispatch site).
    decide_inputs = dict(
        gaussians=b.gaussians.storage,
        num_gaussians_counter=b.num_gaussians_counter.storage,
        eta_n_total=b.eta_n_total.storage,
        eta_n_high=b.eta_buffers["n_high"].storage,
        eta_n_low=b.eta_buffers["n_low"].storage,
        eta_max_x=b.eta_buffers["max_x"].storage,
        eta_max_y=b.eta_buffers["max_y"].storage,
        eta_max_z=b.eta_buffers["max_z"].storage,
        grad_mean_2d_mag_accum=b.grad_mean_2d_mag_accum.storage,
        visibility_count_accum=b.visibility_count_accum.storage,
        action=b.action_buffer.storage,
        split_factor_x=b.split_factor_x.storage,
        split_factor_y=b.split_factor_y.storage,
        split_factor_z=b.split_factor_z.storage,
        tau_split=cfg.tau_split,
        tau_prune=cfg.tau_prune,
        grad_threshold_internal=grad_thresh_internal,
        split_power=cfg.split_power,
        max_split_factor=int(cfg.max_split_factor),
        t_opacity=cfg.t_opacity,
        dense_scale=cfg.dense_scale,
        max_splats=pipeline.max_splats,
        # Prune/scan scratch is unused until splitting finishes.
        selected_parents=b.valid_mask_buffer.storage,
        selected_count=b.scan_total.storage,
    )

    with pipeline.recording("densify_structure") as recording, \
            pipeline.marker_group("densify_structure"):
        # P0: zero the requested-tail-growth counter (read back after the tick
        # for the overbudget warning).
        pipeline.dispatch(pipeline.k_clear_uint_buffer, "clear_children_sum",
            thread_count=[1, 1, 1],
            buffer=b.densify_children_sum.storage,
            count=1,
        )
        pipeline.dispatch(pipeline.k_clear_uint_buffer, "clear_selected_count",
            thread_count=[1, 1, 1], buffer=b.scan_total.storage, count=1)
        pipeline.dispatch(pipeline.k_densify_snapshot, "densify_snapshot",
            thread_count=[1, 1, 1], src=b.num_gaussians_counter.storage,
            dst=b.densify_pre_count.storage)
        pipeline.dispatch(pipeline.k_densify_prepare_args, "densify_prepare_args",
            thread_count=[1, 1, 1], count_buf=b.densify_pre_count.storage,
            dispatch_args=b.densify_dispatch_args.storage, max_splats=pipeline.max_splats)

        # P1: single-pass structure decide -> action + split_factor_{x,y,z}
        # (+ atomic children_sum). No capacity demotion: overbudget splits are
        # dropped at the tail by split_anisotropic and surfaced by the host.
        pipeline.dispatch_indirect(pipeline.k_decide_actions_structure, "decide_actions_structure",
            b.densify_dispatch_args,
            densify_children_sum=b.densify_children_sum.storage,
            **decide_inputs,
        )

        # One cooperative workgroup per selected parent.
        pipeline.dispatch(pipeline.k_structure_split_prepare_args, "structure_split_prepare_args",
            thread_count=[1, 1, 1],
            count_buf=b.scan_total.storage,
            dispatch_args=b.densify_dispatch_args.storage,
            max_splats=pipeline.max_splats,
        )

        # P6: split_anisotropic — indirect, self-bounds on pre_count_buf. The SH
        # variant additionally carries features_rest (children inherit) when
        # view-dependent color is enabled.
        split_kwargs = dict(
            gaussians=b.gaussians.storage,
            adam_state=b.adam_state.storage,
            num_gaussians_counter=b.num_gaussians_counter.storage,
            action=b.action_buffer.storage,
            split_factor_x=b.split_factor_x.storage,
            split_factor_y=b.split_factor_y.storage,
            split_factor_z=b.split_factor_z.storage,
            pre_count_buf=b.densify_pre_count.storage,
            selected_parents=b.valid_mask_buffer.storage,
            selected_count=b.scan_total.storage,
            ks_scale_power=float(cfg.ks_scale_power),
            max_splats=pipeline.max_splats,
        )
        if b.sh_enabled:
            split_kwargs.update(
                features_rest=b.features_rest.storage,
                adam_rest=b.adam_rest_state.storage,
                rest_floats=b.sh.rest_floats,
            )
        pipeline.dispatch_indirect(
            pipeline.k_split_anisotropic_sh if b.sh_enabled
            else pipeline.k_split_anisotropic,
            "split_anisotropic", b.densify_dispatch_args, **split_kwargs)

        # P7: snapshot post_count.
        pipeline.dispatch(pipeline.k_densify_snapshot, "densify_snapshot",
            thread_count=[1, 1, 1],
            src=b.num_gaussians_counter.storage,
            dst=b.densify_post_count.storage,
        )

        # P8: retire selected-parent scratch. RHI resource tracking inserts the
        # SRV->UAV dependency before these buffers become the prune mask/count.
        pipeline.dispatch(pipeline.k_build_prune_mask, "build_prune_mask",
            thread_count=[pipeline.max_splats, 1, 1],
            action=b.action_buffer.storage,
            valid_mask=b.valid_mask_buffer.storage,
            pre_count_buf=b.densify_pre_count.storage,
            post_count_buf=b.densify_post_count.storage,
            max_splats=pipeline.max_splats,
        )
        pipeline.scan_service.scan(pipeline.max_splats, b.valid_mask_buffer)
        # Shared-arena compaction: compact each resource into the arena, then
        # copy it back into the live buffer (commits in place; no pointer swap).
        # sgl auto-barriers scatter->copy and copy->next-scatter.
        arena = b.scratch_arena
        pipeline.dispatch(pipeline.k_scatter_compact_gaussian, "scatter_compact_gaussian",
            thread_count=[pipeline.max_splats, 1, 1],
            gaussians_in=b.gaussians.storage,
            valid_mask=b.valid_mask_buffer.storage,
            scan=b.scan_buffer.storage,
            gaussians_out=arena.storage,
            max_splats=pipeline.max_splats,
        )
        pipeline.copy_arena_back(arena, b.gaussians, b.gaussians.storage.size,
                                 count_buf=b.scan_total)
        pipeline.dispatch(pipeline.k_scatter_compact_adam, "scatter_compact_adam",
            thread_count=[pipeline.max_splats, 1, 1],
            adam_state_in=b.adam_state.storage,
            valid_mask=b.valid_mask_buffer.storage,
            scan=b.scan_buffer.storage,
            adam_state_out=arena.storage,
            max_splats=pipeline.max_splats,
        )
        pipeline.copy_arena_back(arena, b.adam_state, b.adam_state.storage.size,
                                 count_buf=b.scan_total)
        # SH: carry features_rest + its Adam moments through the SAME compaction.
        if b.sh_enabled:
            pipeline.dispatch(pipeline.k_scatter_compact_features, "scatter_compact_features",
                thread_count=[pipeline.max_splats, 1, 1],
                features_rest_in=b.features_rest.storage,
                valid_mask=b.valid_mask_buffer.storage,
                scan=b.scan_buffer.storage,
                features_rest_out=arena.storage,
                rest_floats=b.sh.rest_floats,
                max_splats=pipeline.max_splats,
            )
            pipeline.copy_arena_back(arena, b.features_rest, b.features_rest.storage.size,
                                     count_buf=b.scan_total)
            reorder_soa_planes(
                commands=pipeline, arena=arena, buffer=b.adam_rest_state,
                kernel=pipeline.k_scatter_compact_adam_rest,
                marker="scatter_compact_adam_rest",
                plane_count=b.sh.moment_planes, capacity=pipeline.max_splats,
                live_count=b.scan_total,
                valid_mask=b.valid_mask_buffer.storage,
                scan=b.scan_buffer.storage,
            )
        # P9: final count = scan_total (GPU copy — no readback).
        pipeline.dispatch(pipeline.k_densify_snapshot, "densify_snapshot",
            thread_count=[1, 1, 1],
            src=b.scan_total.storage,
            dst=b.num_gaussians_counter.storage,
        )

        # P10: post-densify opacity cap (SADGS gaussian_model.py:1046 applies
        # inverse_sigmoid(min(get_opacity, 0.8)) to ALL survivors after the
        # structure prune). resetOpacity does min(logit, logit(cap)) which is
        # exactly that cap. Now runs on the LIVE buffers (compaction already
        # copied back in place — no swap). Skipped when opacity_cap <= 0.
        if cfg.opacity_cap and cfg.opacity_cap > 0.0:
            pipeline.dispatch(pipeline.k_opacity_reset, "opacity_cap",
                thread_count=[pipeline.max_splats, 1, 1],
                gaussians=b.gaussians.storage,
                adam_state=b.adam_state.storage,
                num_gaussians_counter=b.num_gaussians_counter.storage,
                reset_value=float(cfg.opacity_cap),
                mode=0,   # CLAMP: min(logit, logit(0.8)) == inverse_sigmoid(min(a,0.8))
            )

    # Keep the overflow warning with two scalar, reflection-free readbacks.
    # Wait for this submit, not queue-wide idle (the viewer shares the device).
    # Large eta readbacks and percentile reconstruction are opt-in below.
    pipeline.runtime.wait_for_submit(pipeline.last_submit_id)
    children_sum = int(b.densify_children_sum.storage.to_numpy().view(np.uint32)[0])
    pre_count = int(b.densify_pre_count.storage.to_numpy().view(np.uint32)[0])
    headroom = max(pipeline.max_splats - pre_count, 0)
    overbudget = children_sum > headroom
    split_frac = (children_sum / pre_count) if pre_count > 0 else 0.0
    if overbudget:
        print(f"WARNING: structure densify overbudget at {pre_count} splats — "
              f"requested {children_sum} new children ({split_frac:.0%} of the "
              f"population) but only {headroom} fit (max_splats="
              f"{pipeline.max_splats}); overflow children DROPPED at the tail. "
              f"A single tick asking for this many splits usually means the split "
              f"gate is too loose — tune --sad-tau-split / --sad-max-split-factor "
              f"/ --max-splats.")
    if overbudget and cfg.detailed_diagnostics:
        # Full-capacity eta readbacks can move hundreds of MB over PCIe, then
        # allocate more CPU arrays for percentiles. Never do this implicitly in
        # a production densification tick; --densify-diag opts in explicitly.
        def _u32(buf):
            return np.frombuffer(buf.storage.to_numpy().tobytes(), dtype=np.uint32)[:pre_count]
        def _i64(buf):
            return np.frombuffer(buf.storage.to_numpy().tobytes(), dtype=np.int64)[:pre_count]
        n_total = _u32(b.eta_n_total).astype(np.float64)
        n_high = _u32(b.eta_buffers["n_high"]).astype(np.float64)
        max_x = _u32(b.eta_buffers["max_x"]).astype(np.float64) / 65536.0
        max_y = _u32(b.eta_buffers["max_y"]).astype(np.float64) / 65536.0
        max_z = _u32(b.eta_buffers["max_z"]).astype(np.float64) / 65536.0
        vis = _u32(b.visibility_count_accum).astype(np.float64)
        grad_mag = _i64(b.grad_mean_2d_mag_accum).astype(np.float64) / 65536.0
        seen = n_total > 0
        n_seen = int(seen.sum())
        if n_seen > 0:
            high_ratio = np.where(seen, n_high / np.maximum(n_total, 1.0), 0.0)[seen]
            avg_grad = np.where(vis > 0, grad_mag / np.maximum(vis, 1.0), 0.0)[seen]
            eta_max = np.maximum(max_x, np.maximum(max_y, max_z))[seen]
            is_grad_high = avg_grad >= grad_thresh_internal
            frac_ratio_pass = float(np.mean(high_ratio > cfg.tau_split))
            frac_grad_pass = float(np.mean(is_grad_high))
            print(f"  eta-gate breakdown (over {n_seen} seen splats): "
                  f"n_total p50={np.percentile(n_total[seen], 50):.0f} "
                  f"p90={np.percentile(n_total[seen], 90):.0f}; "
                  f"eta_max p50={np.percentile(eta_max, 50):.2f} "
                  f"p90={np.percentile(eta_max, 90):.2f}; "
                  f"high_ratio>{cfg.tau_split:.2f}: {frac_ratio_pass:.0%}; "
                  f"grad>thresh: {frac_grad_pass:.0%}. "
                  f"(low n_total => multiview gate meaningless; high eta_max => "
                  f"splats too big / structure field too hot.)")

    # Population counts remain GPU-resident; only overflow telemetry was read.
    return {
        "pre_count": None,
        "post_split_count": None,
        "final_count": None,
        "overbudget": overbudget,
    }



def morton_sort(pipeline: TrainingPipeline, world_min: np.ndarray,
                world_max: np.ndarray, carry_eta: bool = False) -> None:
    """Reorder the [0, num_gaussians) prefix of gaussians + adam_state by
    3D Morton (Z-order) code so spatially-adjacent primitives become
    memory-adjacent. Standard cache-locality optimization from real-time
    rendering (Cluster-Cull-Compact, Laine et al. 2011); the sorted
    layout stays useful for many subsequent iters, so this runs on a
    coarse cadence (e.g. every ~1000 iters and after each densify).

    ``world_min`` / ``world_max`` bound the scene AABB used to quantize
    means into a 21-bit-per-axis Morton code. A stale bbox is fine —
    quantization saturates to the box, so mispredicted means still land
    in a valid cell; ordering is a heuristic, not a correctness contract.
    Pass roughly the AABB of the initial point cloud padded by 20%.

    Slot-indexed window accumulators (grad_mean_2d_mag_accum,
    visibility_count_accum) are NOT permuted
    here — the caller MUST clear them via :func:`clear_tracking` after
    this returns, or the next densify reads statistics for the wrong slot.

    ``carry_eta`` (SAD-GS): when True, the 6 eta accumulators ARE permuted
    through the same Morton ``sorted_values`` (via mortonScatterEta), so the
    multiview frequency window survives the re-sort instead of being reset —
    the caller then does NOT clear_structure_scores. When False, eta is left
    as-is (caller clears it), preserving the historical behavior for non-SAD
    and periodic-standalone-sort callers.

    Wait-free: one encoder, one submit, no CPU readback."""
    b = pipeline.buffers
    world_min = np.asarray(world_min, dtype=np.float32).reshape(3)
    world_max = np.asarray(world_max, dtype=np.float32).reshape(3)
    extent = np.maximum(world_max - world_min, 1e-6)
    inv_extent = (1.0 / extent).astype(np.float32)

    with pipeline.recording("morton_sort") as recording, pipeline.marker_group("morton_sort"):
        pipeline.dispatch(pipeline.k_densify_snapshot, "morton_snapshot",
            thread_count=[1, 1, 1],
            src=b.num_gaussians_counter.storage,
            dst=b.scan_total.storage,
        )

        pipeline.dispatch(pipeline.k_morton_encode, "morton_encode",
            thread_count=[pipeline.max_splats, 1, 1],
            gaussians=b.gaussians.storage,
            count_buf=b.scan_total.storage,
            keys_lo=b.sort_keys_lo_a.storage,
            keys_hi=b.sort_keys_hi_a.storage,
            values=b.sort_values_a.storage,
            world_min=spy.math.float3(*world_min.tolist()),
            inv_extent=spy.math.float3(*inv_extent.tolist()),
            max_splats=pipeline.max_splats,
        )

        pipeline.radix_sort_pairs()

        # Shared-arena Morton re-sort: permute each resource into the arena, then
        # copy the arena back into the live buffer, reusing the arena for the
        # next resource. sgl auto-barriers scatter->copy and copy->next-scatter.
        # (Single-output kernels + one arena replace the old two full-size
        # scratch buffers + pointer swap.) Eta keeps its own dedicated scratch.
        arena = b.scratch_arena
        pipeline.dispatch(pipeline.k_morton_scatter_gaussian, "morton_scatter_gaussian",
            thread_count=[pipeline.max_splats, 1, 1],
            gaussians_in=b.gaussians.storage,
            sorted_values=b.sort_values_a.storage,
            count_buf=b.scan_total.storage,
            gaussians_out=arena.storage,
            max_splats=pipeline.max_splats,
        )
        pipeline.copy_arena_back(arena, b.gaussians, b.gaussians.storage.size,
                                 count_buf=b.scan_total)

        pipeline.dispatch(pipeline.k_morton_scatter_adam, "morton_scatter_adam",
            thread_count=[pipeline.max_splats, 1, 1],
            adam_state_in=b.adam_state.storage,
            sorted_values=b.sort_values_a.storage,
            count_buf=b.scan_total.storage,
            adam_state_out=arena.storage,
            max_splats=pipeline.max_splats,
        )
        pipeline.copy_arena_back(arena, b.adam_state, b.adam_state.storage.size,
                                 count_buf=b.scan_total)

        if b.sh_enabled:
            pipeline.dispatch(pipeline.k_morton_scatter_features, "morton_scatter_features",
                thread_count=[pipeline.max_splats, 1, 1],
                features_rest_in=b.features_rest.storage,
                sorted_values=b.sort_values_a.storage,
                count_buf=b.scan_total.storage,
                features_rest_out=arena.storage,
                rest_floats=b.sh.rest_floats,
                max_splats=pipeline.max_splats,
            )
            pipeline.copy_arena_back(arena, b.features_rest, b.features_rest.storage.size,
                                     count_buf=b.scan_total)

            reorder_soa_planes(
                commands=pipeline, arena=arena, buffer=b.adam_rest_state,
                kernel=pipeline.k_morton_scatter_adam_rest,
                marker="morton_scatter_adam_rest",
                plane_count=b.sh.moment_planes, capacity=pipeline.max_splats,
                live_count=b.scan_total,
                sorted_values=b.sort_values_a.storage,
                count_buf=b.scan_total.storage,
            )

        # Eta keeps its dedicated scratch + pointer swap (small, 0.22 GB @10M).
        if carry_eta:
            pipeline.dispatch(pipeline.k_morton_scatter_eta, "morton_scatter_eta",
                thread_count=[pipeline.max_splats, 1, 1],
                sorted_values=b.sort_values_a.storage,
                count_buf=b.scan_total.storage,
                **b.eta_scatter_kwargs(),
                max_splats=pipeline.max_splats,
            )
            recording.after_submit(lambda: b.swap_eta())
