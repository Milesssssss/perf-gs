# Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

"""How parsed arguments resolve into a run: its config objects and train/test split."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from _colmap_testkit import write_colmap_scene

from trainer.config.config import AdamLRs
from trainer.config.resolve import RunConfig
from trainer.frontend.app import TrainingLoop
from trainer.frontend.cli import build_parser, parse_args
from trainer.frontend.scene import holdout_split, load_scene_context
from trainer.frontend.schedule import AppearanceSchedule, LearningRateSchedule


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
    assert config.scale_rotation_lr_decay == (1.0, 1.0)


def test_hub_uses_the_chooser_unless_a_scene_is_supplied() -> None:
    assert _args("--hub").scene is None
    assert _args("--hub", "--scene", "my_scene").scene == "my_scene"
    assert _args().scene == "datasets/360_v2/bicycle"


def test_quality_recipe_lowers_the_opacity_lr_and_decays_scale_and_rotation() -> None:
    config = RunConfig.from_args(
        _args("--densify-mode", "structure_aware", "--lr-recipe", "quality"), 1.0)
    assert config.base_lrs.opacity == pytest.approx(0.035)
    assert config.scale_rotation_lr_decay == (3.0, 0.5)


def test_presets_supply_defaults_that_explicit_flags_override() -> None:
    parser = build_parser()
    for name, pixels, sh, splats, steps, fraction in (
        ("ultrafast", 0.25, 2, 1_850_000, 3000, 1.0),
        ("balanced", 1.0, 2, 2_600_000, 3000, 1.0),
        ("quality", 1.0, 3, 4_000_000, 3500, 0.75),
        ("quality+", 1.0, 3, 4_000_000, 0, 0.75),
    ):
        args = parser.parse_args(["--preset", name])
        assert (args.stbn_pixel_fraction, args.sh_degree, args.max_splats) == (pixels, sh, splats)
        assert (args.iters, args.final_splat_fraction) == (steps, fraction)
        assert RunConfig.from_args(args, 1).structure.rank_split_families == (name in ("quality", "quality+"))
        assert args.width is None and args.height is None and args.downsample is None
        assert args.auto_lens and args.stereo_init and args.lr_recipe == "short"
    assert parse_args(["--stbn-pixel-fraction", "1", "--preset", "ultrafast"]).stbn_pixel_fraction == 1
    assert parse_args(["--preset", "quality", "--max-splats", "500000"]).max_splats == 500000
    plus = parse_args(["--preset", "quality+", "--iters", "700"])
    assert plus.iters == 700 and RunConfig.from_args(plus, 1).appearance.band_interval == 583 and not plus.evaluation
    assert vars(parse_args([])) == vars(parser.parse_args(["--preset", "balanced"]))
    with pytest.raises(SystemExit):
        parser.parse_args(["--preset", "perf"])


@pytest.mark.parametrize("cap", [0, 700, 3300])
def test_quality_plus_replays_quality_then_holds_its_last_update(cap):
    scene = SimpleNamespace(train_indices=[1, 2, 3], test_indices=[0], initial_gaussian_count=1,
                            width=16, height=16, colmapScene=SimpleNamespace(images=list(range(4))))

    def loop_for(*flags):
        args = _args("--preview-every", "0", "--log-every", "0", "--seed", "42", *flags)
        cfg = RunConfig.from_args(args, 1.0)
        pipe = SimpleNamespace(stbn_config=cfg.stbn, loss_config=cfg.loss, loss=SimpleNamespace())
        lrs = LearningRateSchedule(cfg.base_lrs, 1.0, cfg.position_lr_max_steps,
                                   cfg.schedule.optimization_steps, cfg.scale_rotation_lr_decay)
        return TrainingLoop(args=args, scene=scene, config=cfg, targets=None, target_textures={},
                            structure_textures=dict.fromkeys(scene.train_indices),
                            runtime=SimpleNamespace(pipeline=pipe, profiler=None), paths=None,
                            learning_rate_scheduler=lrs, appearance_scheduler=AppearanceSchedule(cfg.appearance))

    quality = loop_for("--preset", "quality")
    plus = loop_for("--preset", "quality+", "--iters", str(cap))
    assert plus.image_prune_steps == quality.image_prune_steps == range(1901, 2101)
    assert plus.convergence.first == 3500
    for _ in range(cap or 3500):
        expected = quality._next_iteration()
        actual = plus._next_iteration()
        assert actual == expected
        assert plus.runtime.pipeline.loss.config == quality.runtime.pipeline.loss.config
    if cap == 0:
        for _ in range(100):
            actual = plus._next_iteration()
            assert actual.learning_rates == expected.learning_rates
            assert plus.runtime.pipeline.loss.config == quality.runtime.pipeline.loss.config
        assert actual.events.mip_periodic and actual.step_number == 3600


def test_other_modes_keep_the_default_loss_and_lrs() -> None:
    args = _args("--densify-mode", "simple")
    config = RunConfig.from_args(args, 1.0)
    assert config.loss.l2_weight == args.lambda_l2
    assert config.loss.ssim_weight == args.lambda_ssim
    assert config.base_lrs == AdamLRs()
