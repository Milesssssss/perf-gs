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
    """One structure-aware densification pass.

    Assumes compute_structure_scores() has populated the eta accumulators for
    this window. Phases:
      A  (CPU) read eta stats → per-axis split factors n_k (multiview-consistent,
         capacity-bounded) + prune mask; upload action + split_factor buffers.
      B  splitAnisotropic — grid-split SPLIT gaussians into n_x*n_y*n_z children.
      C  build_prune_mask + scan + scatter_compact — same compaction as densify().

    Returns a stats dict for logging. Uses CPU readback (like densify()), so it
    is NOT wait-free; it runs on the densify cadence, not every iter.
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

    # --- Phase A (CPU): multiview-consistent decision + per-axis split factor. ---
    def _read_uint(buf):
        return np.frombuffer(buf.to_numpy().tobytes(), dtype=np.uint32)[:pre_count]

    n_total = _read_uint(b.eta_n_total).astype(np.float64)
    seen = n_total > 0
    inv_total = np.where(seen, 1.0 / np.maximum(n_total, 1.0), 0.0)

    # SADGS: n_high/n_low are SCALAR per-gaussian view counts (classified on
    # the max-over-axes eta), so the split/prune GATES use scalar ratios.
    eta = b.eta_buffers
    high_ratio = _read_uint(eta["n_high"]).astype(np.float64) * inv_total
    low_ratio = _read_uint(eta["n_low"]).astype(np.float64) * inv_total
    # Per-axis max eta drives only the split FACTOR.
    eta_max = np.stack([_read_uint(eta["max_x"]),
                        _read_uint(eta["max_y"]),
                        _read_uint(eta["max_z"])], axis=1).astype(np.float64) / 65536.0

    # is_grad_high: per-gaussian avg 2D-gradient magnitude over the window,
    # thresholded — SADGS's `torch.norm(xyz_gradient_accum/denom) >= 1e-5`.
    # Our analog is the same signal densify_simple thresholds: the fixed-point
    # grad_mean_2d_mag_accum divided by the visibility count.
    vis = _read_uint(b.visibility_count_accum).astype(np.float64)
    raw_mag = np.frombuffer(b.grad_mean_2d_mag_accum.to_numpy().tobytes(), dtype=np.int64)[:pre_count]
    avg_grad = np.where(vis > 0, raw_mag.astype(np.float64) / 65536.0 / np.maximum(vis, 1.0), 0.0)
    # CONVENTION SCALING (mirrors densify_simple, see grad_thresh_internal
    # ~line 344): grad_mean_2d_mag_accum holds a per-pixel SUM of a no-1/N loss
    # — accumulate_iter.slang deliberately drops the loss-mean factor
    # 1/(H*W*channels) to avoid underflowing the fixed-point accumulator — so
    # avg_grad is ~H*W*channels LARGER than SADGS's per-view MEAN-loss gradient.
    # SADGS thresholds its mean-loss gradient at 1e-5 (train.py: "[CRITICAL] We
    # MUST use this to prevent 7M points"). Comparing our sum-convention avg_grad
    # to a bare 1e-5 is a unit mismatch that makes is_grad_high ALWAYS TRUE,
    # collapsing the split trigger to (high_ratio>0.8 & seen) and exploding
    # densification. Rescale the threshold into our accumulator's convention.
    grad_thresh_internal = cfg.grad_threshold * (pipeline.width * pipeline.height * 3.0)
    is_grad_high = avg_grad >= grad_thresh_internal

    # Opacity for the prune gate.
    gauss_floats = np.frombuffer(
        b.gaussians.to_numpy().tobytes(), dtype=np.float32
    ).reshape(-1, GAUSSIAN_FLOAT_COUNT)[:pre_count]
    opacity = 1.0 / (1.0 + np.exp(-gauss_floats[:, 10]))  # sigmoid(logit)

    # Densification trigger — matching SADGS train.py:
    #   split_mask = (high_ratio > split_ratio_threshold) & is_grad_high
    # high_ratio (scalar max-over-axes eta fraction) reaches 0.8 for genuine
    # violators; the gradient AND-gate keeps well-placed (low-gradient) large
    # gaussians from densifying. SADGS then routes the SAME triggered set to
    # CLONE (small) vs SPLIT (large) by the dense*extent scale qualifier.
    triggered = (high_ratio > cfg.tau_split) & is_grad_high & seen
    is_prune = (low_ratio > cfg.tau_prune) & seen
    # Always prune very-low-opacity gaussians (matches the other strategies).
    is_prune |= opacity < cfg.t_opacity
    triggered &= ~is_prune

    # Clone vs split by world-space max scale (SADGS clone_qualifiers /
    # split_qualifiers around args.dense*extent): small -> CLONE (exact
    # duplicate), large -> anisotropic SPLIT.
    max_scale = np.exp(gauss_floats[:, 3:6]).max(axis=1)  # world-space per-axis scale
    is_clone = triggered & (max_scale <= cfg.dense_scale)
    is_split = triggered & (max_scale > cfg.dense_scale)

    # Per-axis split factor n_k = ceil(eta_max_k ** split_power), clamped.
    nk = np.ceil(np.power(np.maximum(eta_max, 0.0), cfg.split_power)).astype(np.int64)
    nk = np.clip(nk, 1, cfg.max_split_factor)
    # Non-splitting gaussians keep n_k = 1 on all axes (no children).
    nk = np.where(is_split[:, None], nk, 1)
    prod = nk[:, 0] * nk[:, 1] * nk[:, 2]
    # A split gaussian only splits if its product factor actually exceeds 1;
    # those that would produce a single child fall back to a clone instead
    # (so a triggered gaussian always densifies, matching SADGS).
    degenerate_split = is_split & (prod <= 1)
    is_clone |= degenerate_split
    is_split &= prod > 1

    # --- DIAGNOSTIC (temporary): show exactly where the population comes from. ---
    # densify runs only every ~100 iters, so this print is cheap. Remove once the
    # growth curve is understood.
    if bool(getattr(cfg, "debug_decision", True)):
        import sys as _sys
        seen_n = int(seen.sum())
        hi_n = int(((high_ratio > cfg.tau_split) & seen).sum())
        grad_n = int((is_grad_high & seen).sum())
        vis_seen = avg_grad[seen]
        # Threshold-response: how many seen gaussians survive is_grad_high at a
        # sweep of candidate floors (so we can pick the constant from data).
        # Reported as (grad_threshold_value -> count still passing).
        sweep_vals = [1e-5, 1e-4, 1e-3, 0.0008, 0.002, 0.01, 0.05, 0.2]
        sweep = {
            f"{gt:.0e}x": int((vis_seen >= gt * pipeline.width * pipeline.height * 3.0).sum())
            for gt in sweep_vals
        }
        added_by_split = int((prod[is_split] - 1).sum()) if is_split.any() else 0
        pct = (np.percentile(vis_seen, [50, 90, 99, 100]) if vis_seen.size else [0, 0, 0, 0])
        # eta distribution over seen gaussians: the SCALAR max-over-axes eta
        # (what the gate classifies on) is eta_max.max(axis=1); the per-axis
        # eta_max drives nk. If these are ~>=25-49 the running-max is inflated
        # (accumulation-match target); if high_ratio itself is ~everywhere >0.8
        # the gate is saturated (no longer selective).
        eta_scalar = eta_max.max(axis=1)[seen]
        hr_seen = high_ratio[seen]
        ep = np.percentile(eta_max[seen], [50, 90, 99, 100], axis=0) if seen_n else np.zeros((4, 3))
        esp = np.percentile(eta_scalar, [50, 90, 99, 100]) if eta_scalar.size else [0, 0, 0, 0]
        hrp = np.percentile(hr_seen, [50, 90, 99, 100]) if hr_seen.size else [0, 0, 0, 0]
        # How selective would the high-ratio gate be at stricter eta_high bins?
        # (fraction of seen with scalar eta_max above a sweep of thresholds)
        eta_hi_sweep = {f">{t}": int((eta_scalar > t).sum()) for t in (1.0, 2.0, 4.0, 8.0, 16.0)}
        print(
            f"[SAD-DBG] iter={getattr(pipeline, 'iteration', -1)} pre={pre_count} "
            f"seen={seen_n} high_ratio>{cfg.tau_split}:{hi_n} is_grad_high:{grad_n} "
            f"(floor={grad_thresh_internal:.3g}) triggered={int(triggered.sum())} "
            f"split={int(is_split.sum())} clone={int(is_clone.sum())} prune={int(is_prune.sum())} "
            f"children_from_split={added_by_split}\n"
            f"[SAD-DBG]   avg_grad(seen) p50/p90/p99/max = "
            f"{pct[0]:.3g}/{pct[1]:.3g}/{pct[2]:.3g}/{pct[3]:.3g}\n"
            f"[SAD-DBG]   is_grad_high survivors by grad_threshold: {sweep}\n"
            f"[SAD-DBG]   eta_scalar(seen) p50/p90/p99/max = "
            f"{esp[0]:.3g}/{esp[1]:.3g}/{esp[2]:.3g}/{esp[3]:.3g}  n(eta>)={eta_hi_sweep}\n"
            f"[SAD-DBG]   eta_max per-axis p50 = ({ep[0][0]:.3g},{ep[0][1]:.3g},{ep[0][2]:.3g}) "
            f"p90 = ({ep[1][0]:.3g},{ep[1][1]:.3g},{ep[1][2]:.3g})\n"
            f"[SAD-DBG]   high_ratio(seen) p50/p90/p99/max = "
            f"{hrp[0]:.3g}/{hrp[1]:.3g}/{hrp[2]:.3g}/{hrp[3]:.3g}\n"
            f"[SAD-DBG]   nk per-axis max = {tuple(int(nk[:, a].max()) for a in range(3))} "
            f"prod p50/p90/max (split only) = "
            f"{tuple(np.percentile(prod[is_split], [50, 90, 100]).astype(int)) if is_split.any() else (0,0,0)}",
            flush=True, file=_sys.stderr,
        )

    # Capacity bound: clones add 1 child each, splits add (prod-1). Clones are
    # cheap and are reserved first; if splits then overflow max_splats, drop the
    # highest-fan-out splits until we fit. Keeps the pass within capacity without
    # a GPU-side reservation protocol.
    headroom = pipeline.max_splats - pre_count
    n_clone = int(is_clone.sum())
    if n_clone > headroom:
        # Extremely unlikely, but bound clones too: drop the excess arbitrarily.
        clone_idx = np.where(is_clone)[0][headroom:]
        is_clone[clone_idx] = False
        n_clone = int(is_clone.sum())
    headroom -= n_clone
    split_idx = np.where(is_split)[0]
    if split_idx.size > 0:
        added = prod[split_idx] - 1
        order = np.argsort(-added)          # largest fan-out first
        cum = np.cumsum(added[order])
        keep = cum <= headroom
        dropped = split_idx[order[~keep]]
        is_split[dropped] = False

    actions = np.full(pre_count, ACTION_KEEP, dtype=np.uint32)
    actions[is_split] = ACTION_SPLIT
    actions[is_clone] = ACTION_CLONE
    actions[is_prune] = ACTION_PRUNE

    # Upload action + per-axis split factors (padded to max_splats).
    def _upload(buf, arr_pre):
        full = np.zeros(pipeline.max_splats, dtype=np.uint32)
        full[:pre_count] = arr_pre.astype(np.uint32)
        buf.copy_from_numpy(full)

    _upload(b.action_buffer, actions)
    _upload(b.split_factor_x, np.where(is_split, nk[:, 0], 1))
    _upload(b.split_factor_y, np.where(is_split, nk[:, 1], 1))
    _upload(b.split_factor_z, np.where(is_split, nk[:, 2], 1))

    # --- Phase B: anisotropic grid split. ---
    b.densify_pre_count.copy_from_numpy(np.array([pre_count], dtype=np.uint32))
    pipeline._dispatch(pipeline.k_split_anisotropic, "split_anisotropic",
        thread_count=[pre_count, 1, 1],
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
    pipeline.device.wait_for_idle()
    post_split_count = min(
        int(b.num_gaussians_counter.to_numpy().view(np.uint32)[0]), pipeline.max_splats
    )

    # --- Phase C: build prune mask + scan + scatter-compact (as in densify()). ---
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

    return {
        "pre_count": pre_count,
        "post_split_count": post_split_count,
        "final_count": final_count,
        "n_keep": int((actions == ACTION_KEEP).sum()),
        "n_split": int(is_split.sum()),
        "n_clone": int(is_clone.sum()),
        "n_prune": int(is_prune.sum()),
    }
