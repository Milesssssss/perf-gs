"""Characterization test: freeze the v2 trainer CLI surface + effective defaults.

The refactor brief lists "CLI flag names or acceptance surface" and "effective
defaults" as NON-NEGOTIABLE. This asserts the FULL parsed-default namespace of
``build_parser()``, so any added / removed / renamed flag or changed default
fails loudly and forces a conscious update.

When the parser moves into ``trainer.frontend.cli`` (from the scripts shim),
repoint the import below; ``EXPECTED_DEFAULTS`` stays the contract the extracted
parser must reproduce byte-for-byte.
"""

from __future__ import annotations

from v2_train import build_parser

# dest -> default, exactly as build_parser() defines today. This is the frozen
# CLI contract. The former quirky `store_true, default=True` flags were fixed by
# the CLI/config-correctness refactor: cluster_cull / sad_relative_reset are now
# BooleanOptionalAction (still default True, but --no-... can disable them), and
# the three overlapping densify booleans (simple / importance / structure) were
# replaced by the single --densify-mode choice (default "structure_aware").
EXPECTED_DEFAULTS = {
    "scene": "datasets/360_v2/bicycle",
    "iters": 3000,
    "position_lr_max_steps": 3000,
    "downsample": 4,
    "width": None,
    "height": None,
    "preview_every": 3000,
    "log_every": 0,
    "render_all_every": 0,
    "eval_every": 3000,
    "eval_holdout": 1,
    "eval_holdout_stride": 8,
    "psnr_curve": False,
    "psnr_curve_every": 0,
    "debug_bin_viz": False,
    "densify_every": 100,
    "densify_from": 100,
    "densify_to": 1900,
    "n_scoring_views": 10,
    "t_opacity": 0.005,
    "t_big_scale": 0.1,
    "prune_scale3d": 0.1,
    "reset_every": 1200,
    "opacity_reg": 0.0,
    "scale_reg": 0.0,
    "w_alpha": 0.0,
    "lambda_ssim": 0.2,
    "max_splats": 10_000_000,
    "z_near_cull": 0.01,
    "profile_every": 0,
    "morton_every": 0,
    "cluster_cull": True,
    "densify_mode": "structure_aware",
    "sad_tau_split": 0.8,
    "sad_tau_prune": 0.8,
    "sad_grad_threshold": 1e-5,
    "sad_accum_every": 10,
    "sad_structure_every": 500,
    "sad_freq_grad_threshold": 0.00002,
    "sad_split_power": 0.5,
    "sad_max_split_factor": 8,
    "sad_ks_scale_power": 1.0,
    "sad_structure_min_opacity": 0.1,
    "sad_opacity_cap": 0.8,
    "sad_relative_reset": True,
    "sad_reset_decay": 0.1,
    "densify_diag": False,
    "sad_match_loss_lrs": 1,
    "sad_lambda_l2": 2.0,
    "lambda_l2": 0.0,
    "grad_thresh": 0.0005,
    "no_viewer": False,
    "launch_viewer": False,
    "wait_for_viewer": False,
    "stream_max_splats": 7_000_000,
    "stream_every": 10,
    "backend": "vulkan",
    "seed": 0,
    "mip_splatting": True,
    "mip_kernel_size": 0.1,
    "sh_degree": 0,
    "sh_band_interval": 0,
    "sample_mode": "sadgs",
    "run_name": None,
    "init_opacity": 0.1,
    "grow_scale3d": 0.001,
    "opacity_reset_value": 0.01,
    "profile_run": False,
    "capture_iter": 0,
    "capture_external": False,
    "capture_frames": 240,
    "nsight_debug_names": None,
    "flat_bucket_backward": True,
    "backward_pixel_stride": 1,
    "extinction_hardness": 1.0,
    "extinction_hardness_warmup": 500,
    "final_test_psnr": False,
    "compute_final_metrics": False,
}


def test_cli_defaults_frozen() -> None:
    """Every CLI default matches the frozen contract (values AND flag set)."""
    args = build_parser().parse_args([])
    assert vars(args) == EXPECTED_DEFAULTS


def test_cli_flag_set_frozen() -> None:
    """The exact set of dest names is frozen — catches a flag added or removed
    even independently of its default value."""
    args = build_parser().parse_args([])
    assert set(vars(args)) == set(EXPECTED_DEFAULTS)


def test_cli_backend_choices() -> None:
    """Backend is restricted to the two supported RHI backends."""
    assert build_parser().parse_args(["--backend", "d3d12"]).backend == "d3d12"
    assert build_parser().parse_args(["--backend", "vulkan"]).backend == "vulkan"
