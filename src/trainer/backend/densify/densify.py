"""Simple/structure-aware densification, Morton sorting, and opacity reset.

These operate ON a :class:`~trainer.pipeline.Pipeline`: they dispatch its
kernels, read/write its buffer bag, and use its ``_scan`` / ``_run_forward`` /
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

if TYPE_CHECKING:
    from trainer.backend.render.pipeline import Pipeline


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


def densify_simple(pipeline: Pipeline, cfg: DensifyConfig,
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


def reset_opacity(pipeline: Pipeline, cfg: DensifyConfig) -> None:
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


def clear_structure_scores(pipeline: Pipeline) -> None:
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


def accumulate_structure_scores(pipeline: Pipeline, structure_tensor,
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
    pipeline: Pipeline,
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

def densify_structure_aware(pipeline: Pipeline, cfg: StructureConfig) -> dict:

    b = pipeline.buffers

    # Convention scaling for the grad gate — identical to densify_simple.
    # grad_mean_2d_mag_accum holds a per-pixel SUM of a no-1/N loss, so the
    # SADGS mean-loss threshold must be scaled up by W*H*channels here.
    grad_thresh_internal = cfg.grad_threshold * (pipeline.width * pipeline.height * 3.0)

    # Shared kwargs for the decide passes. Pass 1 needs writes for the
    # capacity-bound scratch; Pass 2 only reads the threshold. Split for
    # clarity but the input list is the same.
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
    )

    with pipeline.recording("densify_structure") as recording, \
            pipeline.marker_group("densify_structure"):
        # P0: zero the capacity-bound scratch (children_sum, fanout_hist,
        # fanout_threshold). Dispatched at [513] so a single workgroup covers
        # every histogram bin.
        pipeline.dispatch(pipeline.k_clear_capacity_scratch, "clear_capacity_scratch",
            thread_count=[513, 1, 1],
            densify_children_sum=b.densify_children_sum.storage,
            densify_fanout_hist=b.densify_fanout_hist.storage,
            densify_fanout_threshold=b.densify_fanout_threshold.storage,
        )

        # P1: tentative decide + fan-out sum + histogram.
        pipeline.dispatch(pipeline.k_decide_actions_structure_pass1, "decide_actions_structure_pass1",
            thread_count=[pipeline.max_splats, 1, 1],
            densify_children_sum=b.densify_children_sum.storage,
            densify_fanout_hist=b.densify_fanout_hist.storage,
            **decide_inputs,
        )

        # P2: snapshot pre_count (needed by the threshold picker and by the
        # split_anisotropic indirect dispatch).
        pipeline.dispatch(pipeline.k_densify_snapshot, "densify_snapshot",
            thread_count=[1, 1, 1],
            src=b.num_gaussians_counter.storage,
            dst=b.densify_pre_count.storage,
        )

        # P3: single-thread threshold picker.
        pipeline.dispatch(pipeline.k_decide_actions_structure_threshold, "decide_actions_structure_threshold",
            thread_count=[1, 1, 1],
            num_gaussians_counter=b.num_gaussians_counter.storage,
            densify_children_sum=b.densify_children_sum.storage,
            densify_fanout_hist=b.densify_fanout_hist.storage,
            densify_fanout_threshold=b.densify_fanout_threshold.storage,
            max_splats=pipeline.max_splats,
        )

        # P4: demote high-fan-out splits to CLONE (no-op if threshold sentinel).
        pipeline.dispatch(pipeline.k_decide_actions_structure_pass2, "decide_actions_structure_pass2",
            thread_count=[pipeline.max_splats, 1, 1],
            densify_fanout_threshold=b.densify_fanout_threshold.storage,
            **decide_inputs,
        )

        # P5: indirect-dispatch args for split_anisotropic (groups = ceil(pre_count/256)).
        pipeline.dispatch(pipeline.k_densify_prepare_args, "densify_prepare_args",
            thread_count=[1, 1, 1],
            count_buf=b.densify_pre_count.storage,
            dispatch_args=b.densify_dispatch_args.storage,
            max_splats=pipeline.max_splats,
        )

        # P6: split_anisotropic — indirect, self-bounds on pre_count_buf.
        pipeline.dispatch_indirect(pipeline.k_split_anisotropic, "split_anisotropic",
            b.densify_dispatch_args,
            gaussians=b.gaussians.storage,
            adam_state=b.adam_state.storage,
            num_gaussians_counter=b.num_gaussians_counter.storage,
            action=b.action_buffer.storage,
            split_factor_x=b.split_factor_x.storage,
            split_factor_y=b.split_factor_y.storage,
            split_factor_z=b.split_factor_z.storage,
            pre_count_buf=b.densify_pre_count.storage,
            ks_scale_power=float(cfg.ks_scale_power),
            max_splats=pipeline.max_splats,
        )

        # P7: snapshot post_count.
        pipeline.dispatch(pipeline.k_densify_snapshot, "densify_snapshot",
            thread_count=[1, 1, 1],
            src=b.num_gaussians_counter.storage,
            dst=b.densify_post_count.storage,
        )

        # P8: build prune mask + scan + scatter-compact.
        pipeline.dispatch(pipeline.k_build_prune_mask, "build_prune_mask",
            thread_count=[pipeline.max_splats, 1, 1],
            action=b.action_buffer.storage,
            valid_mask=b.valid_mask_buffer.storage,
            pre_count_buf=b.densify_pre_count.storage,
            post_count_buf=b.densify_post_count.storage,
            max_splats=pipeline.max_splats,
        )
        pipeline.scan_service.scan(pipeline.max_splats, b.valid_mask_buffer)
        pipeline.dispatch(pipeline.k_scatter_compact, "scatter_compact",
            thread_count=[pipeline.max_splats, 1, 1],
            gaussians_in=b.gaussians.storage,
            adam_state_in=b.adam_state.storage,
            valid_mask=b.valid_mask_buffer.storage,
            scan=b.scan_buffer.storage,
            gaussians_out=b.gaussians_scratch.storage,
            adam_state_out=b.adam_state_scratch.storage,
            max_splats=pipeline.max_splats,
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
        # exactly that cap. Runs on the just-compacted scratch buffer; host
        # ownership changes only after submit succeeds. Skipped when
        # opacity_cap <= 0 (ablation off).
        if cfg.opacity_cap and cfg.opacity_cap > 0.0:
            pipeline.dispatch(pipeline.k_opacity_reset, "opacity_cap",
                thread_count=[pipeline.max_splats, 1, 1],
            gaussians=b.gaussians_scratch.storage,
            adam_state=b.adam_state_scratch.storage,
                num_gaussians_counter=b.num_gaussians_counter.storage,
                reset_value=float(cfg.opacity_cap),
                mode=0,   # CLAMP: min(logit, logit(0.8)) == inverse_sigmoid(min(a,0.8))
            )
            recording.after_submit(b.swap_primitive_storage)

    # Wait-free: return None counts. The caller reads num_gaussians_counter on
    # its own cadence (log-every ticks).
    return {
        "pre_count": None,
        "post_split_count": None,
        "final_count": None,
    }



def morton_sort(pipeline: Pipeline, world_min: np.ndarray,
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
        # 1. Publish n GPU-side: scan_total[0] = num_gaussians_counter[0].
        pipeline.dispatch(pipeline.k_densify_snapshot, "morton_snapshot",
            thread_count=[1, 1, 1],
            src=b.num_gaussians_counter.storage,
            dst=b.scan_total.storage,
        )

        # 2. Encode Morton codes over max_splats. Tail slots get
        # 0xFFFF...FF so the radix parks them at the end.
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

        # 3. 8-pass LSD radix; deterministic return to the A slot.
        pipeline.radix_sort_pairs()

        # 4. Gather-permute Gaussian3D + AdamState via sorted_values.
        pipeline.dispatch(pipeline.k_morton_scatter, "morton_scatter",
            thread_count=[pipeline.max_splats, 1, 1],
            gaussians_in=b.gaussians.storage,
            adam_state_in=b.adam_state.storage,
            sorted_values=b.sort_values_a.storage,
            count_buf=b.scan_total.storage,
            gaussians_out=b.gaussians_scratch.storage,
            adam_state_out=b.adam_state_scratch.storage,
            max_splats=pipeline.max_splats,
        )

        # 4b. (SAD-GS) Permute the eta accumulators by the SAME sorted_values,
        # so the multiview frequency window survives the re-sort. Pure
        # permutation (no create/destroy), reading the identical sorted_values
        # slot as step 4 — the two permutations are consistent by construction.
        if carry_eta:
            pipeline.dispatch(pipeline.k_morton_scatter_eta, "morton_scatter_eta",
                thread_count=[pipeline.max_splats, 1, 1],
                sorted_values=b.sort_values_a.storage,
                count_buf=b.scan_total.storage,
                **b.eta_scatter_kwargs(),
                max_splats=pipeline.max_splats,
            )
        recording.after_submit(
            lambda: b.swap_primitive_storage(carry_eta=carry_eta))
