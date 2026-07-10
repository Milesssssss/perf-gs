"""Densification strategies, importance scoring, and opacity reset.

Two strategies live here:
- :func:`densify` (score-ranked / FastGS): score → CPU rank → select top-B →
  split/clone → compact. Uses CPU readback (breaks async), for the score path.
- :func:`densify_simple` (reference 3DGS): threshold on avg_grad, fully
  wait-free (all GPU, no readback).

Plus :func:`compute_importance_scores` (the scoring pass feeding ``densify``)
and :func:`reset_opacity`.

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

from trainer.buffers import GAUSSIAN_FLOAT_COUNT

if TYPE_CHECKING:
    from trainer.pipeline import Pipeline


@dataclass
class DensifyConfig:
    """Threshold-free score-ranked densification with budget-scheduled growth.

    Growth follows a sqrt schedule from initial count to max_splats over
    the densification window. The budget at each step is deterministic.

    Growth selection is purely rank-based: the top-B gaussians by
    composite score get densified each step. No signal thresholds — the
    schedule determines how many, the ranking determines which ones.

    Score = importance * avg_grad * opacity * surface_area_proxy.
    Split/clone classification: median surface area within the selected set.
    """
    n_scoring_views: int = 10       # cameras sampled for importance scoring
    loss_thresh: float = 0.1        # L1 threshold for binary error mask
    t_opacity: float = 0.005        # prune if sigmoid(opacity) < this. MUST stay BELOW
                                    # t_opacity_reset_value (0.01): reset clamps opacity down
                                    # to the reset value, so a floor ABOVE it prunes every
                                    # splat the instant reset fires (total wipeout). 3DGS uses
                                    # prune 0.005 < reset 0.01 so reset splats survive and only
                                    # genuinely-dead ones decaying below 0.005 get culled.
    t_big_scale: float = 0.1        # prune if max_scale > this (world-space floater removal)
    opacity_cap: float = 0.8        # post-densify opacity cap (softer than reset)
    t_opacity_reset_value: float = 0.01  # reset opacity to this value


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
    t_opacity: float = 0.005         # low-opacity prune floor (always active)
    # Scale threshold separating clone (small) from split (large), = SADGS
    # args.dense * scene_extent. A triggered gaussian with max world-space scale
    # <= this is CLONED (exact duplicate); larger ones are anisotropically split.
    dense_scale: float = 0.0         # dense * extent, in world units; set by the caller


def compute_importance_scores(
    pipeline: Pipeline,
    cameras: list,
    target_textures: list,
    cfg: DensifyConfig,
) -> None:
    """Run N scoring views to accumulate per-gaussian importance scores.

    For each view: render → compute binary error mask → walk tiles and
    count how many high-error pixels each gaussian touches. Results
    accumulate into buffers.importance_score_accum (uint per gaussian).
    """
    b = pipeline.buffers

    pipeline._dispatch(pipeline.k_clear_importance, "clear_importance",
        thread_count=[pipeline.max_splats, 1, 1],
        importance_score_accum=b.importance_score_accum.storage,
        max_splats=pipeline.max_splats,
    )

    for cam, target_tex in zip(cameras, target_textures):
        pipeline._run_forward(cam)

        # Initialize minmax: min=+inf (0x7f7fffff), max=0 (0x00000000).
        b.l1_minmax.copy_from_numpy(np.array([0x7f7fffff, 0x00000000], dtype=np.uint32))

        # Pass 1: compute L1 per pixel, find min/max.
        pipeline._dispatch(pipeline.k_compute_l1_minmax, "compute_l1_minmax",
            thread_count=[pipeline.width, pipeline.height, 1],
            output_image=b.output_image,
            target_image=target_tex,
            l1_map=b.l1_map,
            minmax_buf=b.l1_minmax.storage,
            frame_dim=spy.uint2(pipeline.width, pipeline.height),
        )

        # Pass 2: normalize and threshold.
        pipeline._dispatch(pipeline.k_compute_error_mask, "compute_error_mask",
            thread_count=[pipeline.width, pipeline.height, 1],
            l1_map=b.l1_map,
            minmax_buf=b.l1_minmax.storage,
            error_mask=b.error_mask,
            frame_dim=spy.uint2(pipeline.width, pipeline.height),
            loss_thresh=cfg.loss_thresh,
        )

        sorted_values = pipeline._sorted_values
        pipeline._dispatch(pipeline.k_score_forward, "score_forward",
            thread_count=[b.num_tiles_x * 16, b.num_tiles_y * 16, 1],
            gaussians=b.gaussians.storage,
            projected_2d=b.projected_2d.storage,
            tile_start=b.tile_start.storage,
            tile_end=b.tile_end.storage,
            sorted_gaussian_ids=sorted_values.storage,
            camera=pipeline._camera_buffer,
            error_mask=b.error_mask,
            importance_score_accum=b.importance_score_accum.storage,
        )


def densify(pipeline: Pipeline, cfg: DensifyConfig,
            densify_step: int, n_densify_steps: int,
            init_count: int) -> dict:
    """Run one densification pass: score → rank → select top-B → split/clone.

    Growth follows a sqrt schedule: target(x) = S + (B - S) * sqrt(x/N).
    Front-loads growth (large steps early when sparse, tapering toward
    the target). The per-step budget is max(0, target - current_count),
    catching up if pruning dropped the count below the curve.

    Phases:
      A  decide_actions     - compute scores, apply pruning rules
      B  (CPU) rank+select  - top-B by score, classify split/clone by scale
      C  split_and_clone    - append children, overwrite parent slot for splits
      D  build_prune_mask   - mark valid[i] across post-split range
      E  scan + compact     - prefix-sum + scatter

    Returns a stats dict for logging.
    """
    ACTION_KEEP = 0
    ACTION_SPLIT = 1
    ACTION_CLONE = 2
    ACTION_PRUNE = 3

    b = pipeline.buffers

    pipeline.device.wait_for_idle()
    pre_count = int(b.num_gaussians_counter.to_numpy().view(np.uint32)[0])
    if pre_count == 0:
        return {"pre_count": 0, "post_split_count": 0, "final_count": 0,
                "n_split": 0, "n_clone": 0, "n_prune": 0, "n_keep": 0}

    # Growth schedule: sqrt curve front-loads growth (big steps early when
    # sparse, tapering as we approach target). Catch-up: if pruning dropped
    # count below the curve, we grow more this step.
    B = pipeline.max_splats
    S = init_count
    N = max(n_densify_steps, 1)
    x = min(densify_step, N)
    target_at_x = S + (B - S) * (x / N) ** 0.5
    growth = max(0, int(target_at_x) - pre_count)

    # Prune budget: at most 50% of current count.
    prune = max(1, pre_count // 2)
    b.prune_budget.copy_from_numpy(np.array([prune], dtype=np.int32))

    # Phase A: GPU computes scores and applies pruning.
    pipeline._dispatch(pipeline.k_decide_actions, "decide_actions",
        thread_count=[pre_count, 1, 1],
        gaussians=b.gaussians.storage,
        num_gaussians_counter=b.num_gaussians_counter.storage,
        importance_score_accum=b.importance_score_accum.storage,
        grad_mean_2d_mag_accum=b.grad_mean_2d_mag_accum.storage,
        visibility_count_accum=b.visibility_count_accum.storage,
        action=b.action_buffer.storage,
        densify_score=b.densify_score.storage,
        prune_budget=b.prune_budget.storage,
        t_opacity=cfg.t_opacity,
        t_big_scale=cfg.t_big_scale,
        n_scoring_views=cfg.n_scoring_views,
    )
    pipeline.device.wait_for_idle()

    # Phase B (CPU): read scores + actions, rank, select top-B.
    actions_raw = b.action_buffer.to_numpy()
    actions = np.frombuffer(actions_raw.tobytes(), dtype=np.uint32)[:pre_count].copy()

    scores_raw = b.densify_score.to_numpy()
    scores = np.frombuffer(scores_raw.tobytes(), dtype=np.float32)[:pre_count]

    if growth > 0:
        # Read gaussian scales for split/clone classification.
        gauss_raw = b.gaussians.to_numpy()
        gauss_floats = np.frombuffer(gauss_raw.tobytes(), dtype=np.float32).reshape(-1, GAUSSIAN_FLOAT_COUNT)
        scales = np.exp(gauss_floats[:pre_count, 3:6])
        # Surface area proxy: sum of pairwise axis products (∝ ellipsoid SA).
        sa = (scales[:, 0] * scales[:, 1]
            + scales[:, 0] * scales[:, 2]
            + scales[:, 1] * scales[:, 2])

        # Only consider non-pruned gaussians with positive score.
        candidates = (actions == ACTION_KEEP) & (scores > 0)
        candidate_indices = np.where(candidates)[0]

        if len(candidate_indices) > 0:
            candidate_scores = scores[candidate_indices]
            # Select top-B by score (no threshold).
            n_select = min(growth, len(candidate_indices))
            top_local = np.argpartition(candidate_scores, -n_select)[-n_select:]
            top_indices = candidate_indices[top_local]

            # Classify by surface area: above-median → split (too big,
            # needs subdivision), at/below → clone (right size, region
            # needs more coverage). Relative to the selected set.
            selected_sa = sa[top_indices]
            sa_median = float(np.median(selected_sa))
            for idx in top_indices:
                if sa[idx] > sa_median:
                    actions[idx] = ACTION_SPLIT
                else:
                    actions[idx] = ACTION_CLONE

    # Write the final action buffer back to GPU.
    action_upload = np.zeros(pipeline.max_splats, dtype=np.uint32)
    action_upload[:pre_count] = actions
    b.action_buffer.copy_from_numpy(action_upload)

    # Phase C: split and clone (may increment counter).
    # The shader now reads pre_count from a GPU buffer (unified path with
    # densify_simple's wait-free dispatch); upload our Python-known value.
    b.densify_pre_count.copy_from_numpy(np.array([pre_count], dtype=np.uint32))
    pipeline._dispatch(pipeline.k_split_and_clone, "split_and_clone",
        thread_count=[pre_count, 1, 1],
        gaussians=b.gaussians.storage,
        adam_state=b.adam_state.storage,
        num_gaussians_counter=b.num_gaussians_counter.storage,
        action=b.action_buffer.storage,
        pre_count_buf=b.densify_pre_count.storage,
        max_splats=pipeline.max_splats,
    )
    pipeline.device.wait_for_idle()
    post_split_count = int(b.num_gaussians_counter.to_numpy().view(np.uint32)[0])
    post_split_count = min(post_split_count, pipeline.max_splats)

    # Phases D/E: compact. Same upload-then-pass-buffer pattern.
    b.densify_post_count.copy_from_numpy(np.array([post_split_count], dtype=np.uint32))
    pipeline._dispatch(pipeline.k_build_prune_mask, "build_prune_mask",
        thread_count=[pipeline.max_splats, 1, 1],
        action=b.action_buffer.storage,
        valid_mask=b.valid_mask_buffer.storage,
        pre_count_buf=b.densify_pre_count.storage,
        post_count_buf=b.densify_post_count.storage,
        max_splats=pipeline.max_splats,
    )
    pipeline._scan(pipeline.max_splats)
    pipeline._dispatch(pipeline.k_scatter_compact, "scatter_compact",
        thread_count=[pipeline.max_splats, 1, 1],
        gaussians_in=b.gaussians.storage,
        adam_state_in=b.adam_state.storage,
        valid_mask=b.valid_mask_buffer.storage,
        scan=b.scan_buffer.storage,
        gaussians_out=b.gaussians_scratch.storage,
        adam_state_out=b.adam_state_scratch.storage,
        max_splats=pipeline.max_splats,
    )

    b.gaussians, b.gaussians_scratch = b.gaussians_scratch, b.gaussians
    b.adam_state, b.adam_state_scratch = b.adam_state_scratch, b.adam_state

    pipeline.device.wait_for_idle()
    final_count = int(b.scan_total.to_numpy().view(np.uint32)[0])
    b.num_gaussians_counter.copy_from_numpy(np.array([final_count], dtype=np.uint32))

    n_keep  = int((actions == ACTION_KEEP).sum())
    n_split = int((actions == ACTION_SPLIT).sum())
    n_clone = int((actions == ACTION_CLONE).sum())
    n_prune = int((actions == ACTION_PRUNE).sum())

    return {
        "pre_count": pre_count,
        "post_split_count": post_split_count,
        "final_count": final_count,
        "n_keep": n_keep,
        "n_split": n_split,
        "n_clone": n_clone,
        "n_prune": n_prune,
    }


def densify_simple(pipeline: Pipeline, cfg: DensifyConfig,
                   grad_thresh: float = 0.0002,
                   scale_thresh: float | None = None,
                   prune_big_scale: float | None = None,
                   allow_scale_prune: bool = True) -> dict:
    """Reference 3DGS densification: threshold on avg_grad, split/clone by scale.

    No rank-selection, no importance scoring, no growth budget. Every gaussian
    that exceeds grad_thresh gets densified. Simple and proven.

    ``prune_big_scale`` is the world-space max-scale prune threshold
    (gsplat prune_scale3d * scene_scale). ``allow_scale_prune`` gates it so the
    caller can disable big-scale pruning until after the first opacity reset,
    matching gsplat's ``step > reset_every`` guard. Low-opacity pruning is
    always active.
    """

    b = pipeline.buffers

    if scale_thresh is None:
        scale_thresh = cfg.t_big_scale * 0.1

    # ``grad_thresh`` is expressed in gsplat's convention: the NDC-space
    # densification gradient of the MEAN L1 loss (F.l1_loss averages over
    # H*W*channels). Our accumulator already carries the NDC conversion
    # (accumulate_iter multiplies by frame_dim/2) but NOT the loss-mean
    # factor — the loss shader emits the raw per-pixel gradient sum, and
    # the 1/(H*W*3) factor is too small to survive the 2^16 fixed-point
    # accumulator. So we scale the threshold UP by H*W*channels here to
    # compare against our sum-convention signal. With this, grad_thresh
    # stays resolution- and convention-compatible with gsplat's 0.0002.
    channels = 3.0
    grad_thresh_internal = grad_thresh * (pipeline.width * pipeline.height * channels)
    big_scale = prune_big_scale if prune_big_scale is not None else cfg.t_big_scale

    # Record all densify dispatches into ONE encoder + ONE submit so the
    # whole pass is a single command buffer with no mid-flight submit
    # overhead. Uses the same pattern as train_step.
    enc = pipeline.device.create_command_encoder()
    pipeline._encoder = enc
    with pipeline._marker_group("densify_simple"):
        # ---- Wait-free densify pipeline (all GPU, no CPU readbacks) ----
        #
        # Counts flow like this:
        #   num_gaussians_counter[0] ─(snapshot)─▶ densify_pre_count[0]
        #                                             │
        #     ┌───────────────────────────────────────┘
        #     ▼
        #   splitAndClone atomically bumps num_gaussians_counter[0]
        #     │
        #     └─(snapshot)─▶ densify_post_count[0]
        #                       │
        #     ┌─────────────────┴──────────────────────┐
        #     ▼                                        ▼
        #   buildPruneMask over [max_splats]         scatterCompact over [max_splats]
        #     (zeros valid_mask tail beyond           (skips i where valid_mask=0)
        #      post_count)
        #     │
        #     ▼
        #   _scan over max_splats → scan_total[0] = final_count
        #     │
        #     └─(snapshot)─▶ num_gaussians_counter[0]  (final_count)
        #
        # Everything except splitAndClone runs at [max_splats] and self-bounds
        # off the snapshot buffers, so we only need one indirect-dispatch path.

        # Phase A: decide actions. Dispatches [max_splats]; shader early-outs
        # per-thread on i >= num_gaussians_counter[0] (GPU-resident).
        pipeline._dispatch(pipeline.k_decide_actions_simple, "decide_actions_simple",
            thread_count=[pipeline.max_splats, 1, 1],
            gaussians=b.gaussians.storage,
            num_gaussians_counter=b.num_gaussians_counter.storage,
            grad_mean_2d_mag_accum=b.grad_mean_2d_mag_accum.storage,
            visibility_count_accum=b.visibility_count_accum.storage,
            action=b.action_buffer.storage,
            grad_thresh=grad_thresh_internal,
            scale_thresh=scale_thresh,
            t_opacity=cfg.t_opacity,
            t_big_scale=big_scale,
            allow_scale_prune=1 if allow_scale_prune else 0,
            max_splats=pipeline.max_splats,
        )

        # Snapshot pre_count = num_gaussians_counter[0] (before splitAndClone
        # atomically bumps it). Then compute indirect-dispatch args from that
        # snapshot for splitAndClone.
        pipeline._dispatch(pipeline.k_densify_snapshot, "densify_snapshot",
            thread_count=[1, 1, 1],
            src=b.num_gaussians_counter.storage,
            dst=b.densify_pre_count.storage,
        )
        pipeline._dispatch(pipeline.k_densify_prepare_args, "densify_prepare_args",
            thread_count=[1, 1, 1],
            count_buf=b.densify_pre_count.storage,
            dispatch_args=b.densify_dispatch_args.storage,
            max_splats=pipeline.max_splats,
        )

        # Phase B: split and clone. Indirect so group count scales with
        # pre_count, not max_splats.
        pipeline._dispatch_indirect(pipeline.k_split_and_clone, "split_and_clone",
            b.densify_dispatch_args,
            gaussians=b.gaussians.storage,
            adam_state=b.adam_state.storage,
            num_gaussians_counter=b.num_gaussians_counter.storage,
            action=b.action_buffer.storage,
            pre_count_buf=b.densify_pre_count.storage,
            max_splats=pipeline.max_splats,
        )

        # Snapshot post_count = num_gaussians_counter[0] (after splitAndClone).
        pipeline._dispatch(pipeline.k_densify_snapshot, "densify_snapshot",
            thread_count=[1, 1, 1],
            src=b.num_gaussians_counter.storage,
            dst=b.densify_post_count.storage,
        )

        # Phase C: mark valid. Dispatched over [max_splats]; shader zeros
        # valid_mask for i >= post_count so the subsequent max_splats-wide
        # scan sees a padded 0-tail.
        pipeline._dispatch(pipeline.k_build_prune_mask, "build_prune_mask",
            thread_count=[pipeline.max_splats, 1, 1],
            action=b.action_buffer.storage,
            valid_mask=b.valid_mask_buffer.storage,
            pre_count_buf=b.densify_pre_count.storage,
            post_count_buf=b.densify_post_count.storage,
            max_splats=pipeline.max_splats,
        )

        # Phase D: prefix-scan valid_mask into scan_buffer. Runs at max_splats
        # (the tail 0s contribute nothing to scan_total[0]).
        pipeline._scan(pipeline.max_splats)

        # Phase E: scatter live gaussians into scratch. Dispatched over
        # [max_splats]; skips slots where valid_mask=0 (includes the tail).
        pipeline._dispatch(pipeline.k_scatter_compact, "scatter_compact",
            thread_count=[pipeline.max_splats, 1, 1],
            gaussians_in=b.gaussians.storage,
            adam_state_in=b.adam_state.storage,
            valid_mask=b.valid_mask_buffer.storage,
            scan=b.scan_buffer.storage,
            gaussians_out=b.gaussians_scratch.storage,
            adam_state_out=b.adam_state_scratch.storage,
            max_splats=pipeline.max_splats,
        )

        b.gaussians, b.gaussians_scratch = b.gaussians_scratch, b.gaussians
        b.adam_state, b.adam_state_scratch = b.adam_state_scratch, b.adam_state

        # Final: num_gaussians_counter[0] = scan_total[0]. GPU copy — no
        # readback, no upload.
        pipeline._dispatch(pipeline.k_densify_snapshot, "densify_snapshot",
            thread_count=[1, 1, 1],
            src=b.scan_total.storage,
            dst=b.num_gaussians_counter.storage,
        )

    cmd = enc.finish()
    with pipeline.gpu_lock:
        pipeline.last_submit_id = pipeline.device.submit_command_buffer(cmd)
    pipeline._encoder = None

    # No wait_for_idle. Return an empty stats dict; the caller reads
    # num_gaussians_counter on its own cadence (e.g. log-every).
    return {
        "pre_count": None,
        "post_split_count": None,
        "final_count": None,
    }


def reset_opacity(pipeline: Pipeline, cfg: DensifyConfig) -> None:
    """Per-gaussian opacity <- low value. Forces optimizer to re-earn
    opacity on real geometry."""
    b = pipeline.buffers
    pipeline._dispatch(pipeline.k_opacity_reset, "opacity_reset",
        thread_count=[pipeline.max_splats, 1, 1],
        gaussians=b.gaussians.storage,
        num_gaussians_counter=b.num_gaussians_counter.storage,
        reset_value=cfg.t_opacity_reset_value,
    )


# ----------------------------------------------------------------------
# Structure-aware densification (SAD-GS, arXiv 2604.28016)
# ----------------------------------------------------------------------

def clear_structure_scores(pipeline: Pipeline) -> None:
    """Zero the per-axis eta accumulators. Call at the start of each
    accumulation window (right after a densify reorders gaussian slots), so
    the multiview statistics are gathered over a stable gaussian set."""
    b = pipeline.buffers
    pipeline._dispatch(pipeline.k_clear_structure_scores, "clear_structure_scores",
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
    camera (already uploaded to ``pipeline._camera_buffer`` by the forward
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
    pipeline._dispatch(pipeline.k_score_structure_aware, "score_structure_aware",
        thread_count=[pipeline.max_splats, 1, 1],
        gaussians=b.gaussians.storage,
        num_gaussians_counter=b.num_gaussians_counter.storage,
        camera=pipeline._camera_buffer,
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
        pipeline._run_forward(cam)
        accumulate_structure_scores(pipeline, structure_tex, pass_idx=view_idx)

def densify_structure_aware(pipeline: Pipeline, cfg: StructureConfig) -> dict:
    """One structure-aware densification pass — wait-free, GPU-driven.

    Mirrors ``densify_simple``'s single-encoder discipline: no CPU readbacks,
    no host numpy, no ``wait_for_idle``. Assumes ``compute_structure_scores``
    (offline) or the online per-iter ``accumulate_structure_scores`` has
    populated the eta accumulators for this window. Phases (all in one
    command encoder):

      P0  clear_capacity_scratch  — zero the tiny capacity-bound scratch.
      P1  decide_actions_structure_pass1  — tentative decide + fan-out sum +
          histogram (max_splats threads).
      P2  snapshot pre_count.
      P3  decide_actions_structure_threshold  — single-thread top-down
          histogram scan picks the fan-out demote threshold (UINT_MAX = none).
      P4  decide_actions_structure_pass2  — demote highest-fan-out splits to
          CLONE (max_splats threads; no-op if threshold == UINT_MAX).
      P5  prepare indirect-dispatch args from pre_count.
      P6  split_anisotropic (indirect)  — grid-split SPLITs into n_x*n_y*n_z
          children; append CLONE duplicates; parent slot handling per the
          shader.
      P7  snapshot post_count.
      P8  build_prune_mask, scan, scatter_compact  — same compaction as
          densify_simple; swap in the compacted scratch.
      P9  final counter snapshot (num_gaussians_counter = scan_total).

    Returns ``None`` for the count stats (matches ``densify_simple``); the
    training loop reads the counter on log/preview ticks.
    """
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

    enc = pipeline.device.create_command_encoder()
    pipeline._encoder = enc
    with pipeline._marker_group("densify_structure"):
        # P0: zero the capacity-bound scratch (children_sum, fanout_hist,
        # fanout_threshold). Dispatched at [513] so a single workgroup covers
        # every histogram bin.
        pipeline._dispatch(pipeline.k_clear_capacity_scratch, "clear_capacity_scratch",
            thread_count=[513, 1, 1],
            densify_children_sum=b.densify_children_sum.storage,
            densify_fanout_hist=b.densify_fanout_hist.storage,
            densify_fanout_threshold=b.densify_fanout_threshold.storage,
        )

        # P1: tentative decide + fan-out sum + histogram.
        pipeline._dispatch(pipeline.k_decide_actions_structure_pass1, "decide_actions_structure_pass1",
            thread_count=[pipeline.max_splats, 1, 1],
            densify_children_sum=b.densify_children_sum.storage,
            densify_fanout_hist=b.densify_fanout_hist.storage,
            **decide_inputs,
        )

        # P2: snapshot pre_count (needed by the threshold picker and by the
        # split_anisotropic indirect dispatch).
        pipeline._dispatch(pipeline.k_densify_snapshot, "densify_snapshot",
            thread_count=[1, 1, 1],
            src=b.num_gaussians_counter.storage,
            dst=b.densify_pre_count.storage,
        )

        # P3: single-thread threshold picker.
        pipeline._dispatch(pipeline.k_decide_actions_structure_threshold, "decide_actions_structure_threshold",
            thread_count=[1, 1, 1],
            num_gaussians_counter=b.num_gaussians_counter.storage,
            densify_children_sum=b.densify_children_sum.storage,
            densify_fanout_hist=b.densify_fanout_hist.storage,
            densify_fanout_threshold=b.densify_fanout_threshold.storage,
            max_splats=pipeline.max_splats,
        )

        # P4: demote high-fan-out splits to CLONE (no-op if threshold sentinel).
        pipeline._dispatch(pipeline.k_decide_actions_structure_pass2, "decide_actions_structure_pass2",
            thread_count=[pipeline.max_splats, 1, 1],
            densify_fanout_threshold=b.densify_fanout_threshold.storage,
            **decide_inputs,
        )

        # P5: indirect-dispatch args for split_anisotropic (groups = ceil(pre_count/256)).
        pipeline._dispatch(pipeline.k_densify_prepare_args, "densify_prepare_args",
            thread_count=[1, 1, 1],
            count_buf=b.densify_pre_count.storage,
            dispatch_args=b.densify_dispatch_args.storage,
            max_splats=pipeline.max_splats,
        )

        # P6: split_anisotropic — indirect, self-bounds on pre_count_buf.
        pipeline._dispatch_indirect(pipeline.k_split_anisotropic, "split_anisotropic",
            b.densify_dispatch_args,
            gaussians=b.gaussians.storage,
            adam_state=b.adam_state.storage,
            num_gaussians_counter=b.num_gaussians_counter.storage,
            action=b.action_buffer.storage,
            split_factor_x=b.split_factor_x.storage,
            split_factor_y=b.split_factor_y.storage,
            split_factor_z=b.split_factor_z.storage,
            pre_count_buf=b.densify_pre_count.storage,
            max_splats=pipeline.max_splats,
        )

        # P7: snapshot post_count.
        pipeline._dispatch(pipeline.k_densify_snapshot, "densify_snapshot",
            thread_count=[1, 1, 1],
            src=b.num_gaussians_counter.storage,
            dst=b.densify_post_count.storage,
        )

        # P8: build prune mask + scan + scatter-compact.
        pipeline._dispatch(pipeline.k_build_prune_mask, "build_prune_mask",
            thread_count=[pipeline.max_splats, 1, 1],
            action=b.action_buffer.storage,
            valid_mask=b.valid_mask_buffer.storage,
            pre_count_buf=b.densify_pre_count.storage,
            post_count_buf=b.densify_post_count.storage,
            max_splats=pipeline.max_splats,
        )
        pipeline._scan(pipeline.max_splats)
        pipeline._dispatch(pipeline.k_scatter_compact, "scatter_compact",
            thread_count=[pipeline.max_splats, 1, 1],
            gaussians_in=b.gaussians.storage,
            adam_state_in=b.adam_state.storage,
            valid_mask=b.valid_mask_buffer.storage,
            scan=b.scan_buffer.storage,
            gaussians_out=b.gaussians_scratch.storage,
            adam_state_out=b.adam_state_scratch.storage,
            max_splats=pipeline.max_splats,
        )
        b.gaussians, b.gaussians_scratch = b.gaussians_scratch, b.gaussians
        b.adam_state, b.adam_state_scratch = b.adam_state_scratch, b.adam_state

        # P9: final count = scan_total (GPU copy — no readback).
        pipeline._dispatch(pipeline.k_densify_snapshot, "densify_snapshot",
            thread_count=[1, 1, 1],
            src=b.scan_total.storage,
            dst=b.num_gaussians_counter.storage,
        )

    cmd = enc.finish()
    with pipeline.gpu_lock:
        pipeline.last_submit_id = pipeline.device.submit_command_buffer(cmd)
    pipeline._encoder = None

    # Wait-free: return None counts. The caller reads num_gaussians_counter on
    # its own cadence (log-every ticks).
    return {
        "pre_count": None,
        "post_split_count": None,
        "final_count": None,
    }
