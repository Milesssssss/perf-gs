# Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

"""How parsed arguments resolve into a run: its config objects and train/test split."""

from __future__ import annotations

from _colmap_testkit import write_colmap_scene
from v2_train import build_parser

from trainer.config.config import AdamLRs
from trainer.config.resolve import RunConfig
from trainer.frontend.scene import holdout_split, load_scene_context


def _args(*flags: str):
    return build_parser().parse_args(list(flags))


def test_holdout_split_on() -> None:
    test_indices, train_indices = holdout_split(10, 4, holdout=True)
    assert test_indices == [0, 4, 8]
    assert train_indices == [1, 2, 3, 5, 6, 7, 9]


def test_holdout_split_off_trains_and_evaluates_everything() -> None:
    assert holdout_split(6, 8, holdout=False) == (list(range(6)), list(range(6)))


def test_scene_floors_the_holdout_stride_to_one(tmp_path) -> None:
    scene_dir = write_colmap_scene(tmp_path / "scene")
    scene = load_scene_context(_args("--scene", str(scene_dir), "--downsample", "1",
                                     "--eval-holdout-stride", "0"))
    assert scene.holdout_stride == 1
    assert sorted(scene.test_indices + scene.train_indices) == list(range(scene.camera_count))


def test_densify_thresholds_scale_with_the_scene_extent() -> None:
    args = _args()
    config = RunConfig.from_args(args, 3.0)
    assert config.densify.t_big_scale == args.t_big_scale * 3.0
    assert config.structure.n_scoring_views == args.n_scoring_views
    assert config.structure.dense_scale == args.grow_scale3d * 3.0
    assert config.structure.t_opacity == args.sad_structure_min_opacity


def test_structure_aware_runs_match_sadgs_loss_and_lrs() -> None:
    args = _args("--densify-mode", "structure_aware", "--sad-match-loss-lrs", "1")
    config = RunConfig.from_args(args, 1.0)
    assert config.loss.l2_weight == args.sad_lambda_l2
    assert config.loss.ssim_weight == args.lambda_ssim
    base = config.base_lrs
    assert (base.scale, base.rot, base.opacity, base.color) == (0.01, 0.002, 0.05, 0.0025)


def test_other_modes_keep_the_default_loss_and_lrs() -> None:
    args = _args("--densify-mode", "simple")
    config = RunConfig.from_args(args, 1.0)
    assert config.loss.l2_weight == args.lambda_l2
    assert config.loss.ssim_weight == args.lambda_ssim
    assert config.base_lrs == AdamLRs()
