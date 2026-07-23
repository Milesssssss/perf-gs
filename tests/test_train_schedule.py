"""Characterization test: freeze the training-loop event schedule + split.

The v2 loop decides per-iteration events (densify / opacity-reset / structure /
preview / render-all / log / eval / morton / mip-recompute) with inline modulo
logic, plus the train/test holdout split. The refactor extracts these into a
pure ``TrainingSchedule`` + split helper; these golden cases pin the CURRENT
semantics — including the tricky invariants the brief calls out:

  * opacity reset WINS over densify on a colliding iter,
  * structure-densify fires on ABSOLUTE multiples of ``sad_structure_every``
    (only on densify ticks that are not resets),
  * preview is BUFFERED after densify/reset (+50, or +350 on a reset iter).

The ``_reference_events`` / ``_reference_split`` functions below replicate
today's inline logic; the parametrized constants are hand-verified golden
outputs. When the pure schedule module lands, assert IT reproduces these.
"""

from __future__ import annotations

import pytest

# Scheduling knobs at their CLI defaults (see test_cli_defaults.EXPECTED_DEFAULTS).
DEFAULT_CFG = dict(
    iters=3000,
    densify_from=100,
    densify_to=1900,
    densify_every=100,
    reset_every=1200,
    structure_densify=True,
    sad_structure_every=500,
    preview_every=3000,
    log_every=0,
    render_all_every=0,
    eval_every=3000,
    morton_every=0,
    psnr_curve_enabled=False,
    psnr_curve_every=0,
    mip_enabled=True,
)

EVENT_KEYS = (
    "densify", "opacity_reset", "structure", "preview", "render_all",
    "log", "eval", "morton", "mip",
)


def _reference_events(it, *, iters, densify_from, densify_to, densify_every,
                      reset_every, structure_densify, sad_structure_every,
                      preview_every, log_every, render_all_every, eval_every,
                      morton_every, psnr_curve_enabled, psnr_curve_every,
                      mip_enabled):
    """Replica of the current v2_train.py per-iteration event logic."""
    do_densify = (densify_from < it <= densify_to
                  and (it - densify_from) % densify_every == 0)
    do_opacity_reset = (reset_every > 0 and it % reset_every == 0
                        and densify_from <= it <= densify_to)
    # Reset wins over densify on a colliding iter.
    if do_densify and do_opacity_reset:
        do_densify = False
    do_structure = (structure_densify and do_densify
                    and (it % sad_structure_every == 0))

    do_preview = False
    if preview_every > 0:
        buffered = [itr + 350 if itr % reset_every == 0 else itr + 50
                    for itr in range(preview_every, iters, preview_every)]
        do_preview = (it in buffered) or (it == iters)

    if render_all_every > 0:
        do_render_all = (it % render_all_every == 0) or (it == iters)
    else:
        do_render_all = do_preview

    do_log = log_every > 0 and ((it % log_every == 0) or (it == 1) or (it == iters))
    do_eval = eval_every > 0 and (it % eval_every == 0)
    do_psnr = psnr_curve_enabled and ((it + 1) % max(1, psnr_curve_every) == 0)
    do_morton = morton_every > 0 and (it % morton_every == 0) and not do_densify
    do_mip = mip_enabled and (
        do_densify
        or (it > densify_to and it % 100 == 0 and it < iters - 100)
    )
    return {
        "densify": do_densify,
        "opacity_reset": do_opacity_reset,
        "structure": do_structure,
        "preview": do_preview,
        "render_all": do_render_all,
        "log": do_log,
        "eval": do_eval,
        "psnr_curve": do_psnr,
        "morton": do_morton,
        "mip": do_mip,
    }


def _active(it, **cfg):
    return {k for k, v in _reference_events(it, **cfg).items() if v}


# (iteration, set-of-active-events) golden cases for the DEFAULT schedule.
DEFAULT_GOLDEN = [
    (1, set()),
    (100, set()),                                   # densify_from is exclusive
    (200, {"densify", "mip"}),                       # first densify (AbsGS filler)
    (500, {"densify", "structure", "mip"}),          # structure tick (500 | 500)
    (1000, {"densify", "structure", "mip"}),
    (1200, {"opacity_reset"}),                       # reset WINS over densify
    (1500, {"densify", "structure", "mip"}),
    (1900, {"densify", "mip"}),                       # last densify (not a 500 mult)
    (2000, {"mip"}),                                 # post-densify_to periodic mip
    (2900, set()),                                    # mip skips last ~100 iters
    (3000, {"preview", "render_all", "eval"}),       # final iter: preview + eval sweep
]


@pytest.mark.parametrize("it,expected", DEFAULT_GOLDEN)
def test_default_schedule_golden(it, expected) -> None:
    assert _active(it, **DEFAULT_CFG) == expected


def test_reset_wins_over_densify() -> None:
    ev = _reference_events(1200, **DEFAULT_CFG)
    assert ev["opacity_reset"] is True
    assert ev["densify"] is False
    assert ev["structure"] is False


def test_structure_only_on_absolute_multiples() -> None:
    # 500/1000/1500 are structure ticks; 200/300/.../1900 densify but not structure.
    for it in (500, 1000, 1500):
        assert _active(it, **DEFAULT_CFG) == {"densify", "structure", "mip"}
    for it in (200, 300, 700, 1900):
        assert "structure" not in _active(it, **DEFAULT_CFG)


def test_preview_buffering_plus50_and_plus350() -> None:
    # preview_every=600 over reset_every=1200: iters at 600/1800 buffer +50,
    # iters at 1200/2400 (reset multiples) buffer +350.
    cfg = {**DEFAULT_CFG, "preview_every": 600}
    preview_iters = {it for it in range(1, cfg["iters"] + 1)
                     if _reference_events(it, **cfg)["preview"]}
    assert preview_iters == {650, 1550, 1850, 2750, 3000}


def test_psnr_curve_fires_one_iter_before_boundary() -> None:
    cfg = {**DEFAULT_CFG, "psnr_curve_enabled": True, "psnr_curve_every": 100}
    assert _reference_events(99, **cfg)["psnr_curve"] is True     # (99+1) % 100 == 0
    assert _reference_events(199, **cfg)["psnr_curve"] is True
    assert _reference_events(100, **cfg)["psnr_curve"] is False


def test_standalone_morton_skips_densify_iters() -> None:
    cfg = {**DEFAULT_CFG, "morton_every": 100}
    # 200 is a densify tick -> standalone morton suppressed (densify does its own).
    assert _reference_events(200, **cfg)["morton"] is False
    # 250 is not a densify tick and 250 % 100 != 0 -> no morton either.
    assert _reference_events(2500, **cfg)["morton"] is True   # 2500 % 100 == 0, no densify


# --- train/test holdout split -------------------------------------------------

def _reference_split(n_cams, stride, eval_holdout):
    if eval_holdout:
        test = [i for i in range(n_cams) if i % stride == 0]
        train = [i for i in range(n_cams) if i % stride != 0]
    else:
        test = list(range(n_cams))
        train = list(range(n_cams))
    return train, test


def test_split_holdout_every_stride() -> None:
    train, test = _reference_split(16, 8, eval_holdout=1)
    assert test == [0, 8]
    assert train == [1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15]
    assert set(train).isdisjoint(test)
    assert sorted(train + test) == list(range(16))


def test_split_disabled_trains_and_evals_all() -> None:
    train, test = _reference_split(10, 8, eval_holdout=0)
    assert train == list(range(10))
    assert test == list(range(10))
