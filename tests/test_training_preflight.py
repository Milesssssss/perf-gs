"""Characterization + contract tests for trainer.frontend.training_preflight.

Pins the pure setup-phase derivations extracted from app.main() — the held-out
eval split, the Morton normalization AABB, the densify/structure config objects
(extent scaling), and the SAD-GS-matched loss/LR resolution — against the parsed
CLI defaults so the extraction stays behavior-preserving.
"""

from __future__ import annotations

import numpy as np
from v2_train import build_parser

from trainer.config.config import AdamLRs
from trainer.frontend.training_preflight import (
    build_densify_configs,
    compute_morton_bounds,
    resolve_holdout_split,
    resolve_loss_and_lrs,
)


def _args(**overrides):
    args = build_parser().parse_args([])
    for k, v in overrides.items():
        setattr(args, k, v)
    return args


def test_holdout_split_on() -> None:
    args = _args(eval_holdout=1, eval_holdout_stride=4)
    stride, test_indices, train_indices = resolve_holdout_split(args, 10)
    assert stride == 4
    assert test_indices == [0, 4, 8]
    assert train_indices == [1, 2, 3, 5, 6, 7, 9]
    # Partition covers every camera exactly once.
    assert sorted(test_indices + train_indices) == list(range(10))


def test_holdout_split_off() -> None:
    args = _args(eval_holdout=0, eval_holdout_stride=8)
    stride, test_indices, train_indices = resolve_holdout_split(args, 6)
    assert test_indices == list(range(6))
    assert train_indices == list(range(6))


def test_holdout_stride_floored_to_one() -> None:
    args = _args(eval_holdout=1, eval_holdout_stride=0)
    stride, _, _ = resolve_holdout_split(args, 5)
    assert stride == 1


def test_morton_bounds_pad_20pct() -> None:
    means = np.array([[0.0, 0.0, 0.0], [10.0, 10.0, 10.0]], dtype=np.float32)
    lo, hi = compute_morton_bounds(means)
    # extent = 10 per axis; padded 20% -> [-2, 12].
    assert np.allclose(lo, [-2.0, -2.0, -2.0])
    assert np.allclose(hi, [12.0, 12.0, 12.0])


def test_morton_bounds_degenerate_axis() -> None:
    # All points share a coordinate on one axis -> extent floored to 1e-6.
    means = np.array([[0.0, 1.0, 0.0], [0.0, 3.0, 0.0]], dtype=np.float32)
    lo, hi = compute_morton_bounds(means)
    assert hi[0] - lo[0] > 0.0  # never collapses to a zero-width box
    assert np.isclose(lo[1], 1.0 - 0.2 * 2.0)
    assert np.isclose(hi[1], 3.0 + 0.2 * 2.0)


def test_densify_configs_extent_scaling() -> None:
    args = _args()
    extent = 3.0
    cfg, structure_cfg = build_densify_configs(args, extent)
    assert cfg.t_big_scale == args.t_big_scale * extent
    assert structure_cfg.n_scoring_views == args.n_scoring_views
    assert structure_cfg.dense_scale == args.grow_scale3d * extent
    assert structure_cfg.t_opacity == args.sad_structure_min_opacity


def test_loss_and_lrs_sad_match() -> None:
    args = _args(densify_mode="structure_aware", sad_match_loss_lrs=True)
    l2, ssim, base = resolve_loss_and_lrs(args)
    assert l2 == args.sad_lambda_l2
    assert ssim == args.lambda_ssim
    assert (base.scale, base.rot, base.opacity, base.color) == (0.01, 0.002, 0.05, 0.0025)


def test_loss_and_lrs_defaults_when_not_structure() -> None:
    args = _args(densify_mode="simple")
    l2, ssim, base = resolve_loss_and_lrs(args)
    default = AdamLRs()
    assert l2 == args.lambda_l2
    assert ssim == args.lambda_ssim
    assert (base.scale, base.rot, base.opacity, base.color) == (
        default.scale, default.rot, default.opacity, default.color)
