"""Contract test: TrainingSchedule reproduces the frozen inline loop logic.

Slice 3 extracts the v2 loop's inline per-iteration timing into
``trainer.training.schedule.TrainingSchedule``. This proves the extracted module
matches the ORIGINAL inline logic EXACTLY — both at the hand-verified golden
iterations and EXHAUSTIVELY across the full run for several configs that
exercise preview buffering, the standalone morton / periodic-mip sites, the
psnr-curve, render-all, and structure/mip toggles.
"""

from __future__ import annotations

from dataclasses import asdict

import pytest

from trainer.config_types import DensificationMode, ScheduleConfig
from trainer.training.schedule import TrainingSchedule

# Schedule knobs at their CLI defaults.
DEFAULT_KNOBS = dict(
    iterations=3000, densify_from=100, densify_to=1900, densify_every=100,
    reset_every=1200, densify_mode=DensificationMode.STRUCTURE_AWARE,
    sad_structure_every=500,
    sad_accum_every=10, preview_every=3000, log_every=0, render_all_every=0,
    eval_every=3000, psnr_curve_enabled=False, psnr_curve_every=0,
    morton_every=0, stream_every=10, mip_enabled=True,
)

EVENT_FIELDS = ("densify", "opacity_reset", "structure", "preview",
                "render_all", "log", "eval", "psnr_curve", "morton",
                "mip_after_densify", "mip_periodic")


def _schedule(**overrides):
    return TrainingSchedule(ScheduleConfig(**{**DEFAULT_KNOBS, **overrides}))


def _active(ev):
    return {f for f in EVENT_FIELDS if getattr(ev, f)}


DEFAULT_GOLDEN = [
    (1, set()),
    (100, set()),
    (200, {"densify", "mip_after_densify"}),
    (500, {"densify", "structure", "mip_after_densify"}),
    (1000, {"densify", "structure", "mip_after_densify"}),
    (1200, {"opacity_reset"}),
    (1500, {"densify", "structure", "mip_after_densify"}),
    (1900, {"densify", "mip_after_densify"}),
    (2000, {"mip_periodic"}),
    (2900, set()),
    (3000, {"preview", "render_all", "eval"}),
]


@pytest.mark.parametrize("it,expected", DEFAULT_GOLDEN)
def test_module_golden(it, expected) -> None:
    assert _active(_schedule().events_at(it)) == expected


def _inline(it, k):
    """Authoritative replica of the ORIGINAL inline v2_train schedule logic."""
    do_densify = (k["densify_mode"] != DensificationMode.NONE
                  and k["densify_from"] < it <= k["densify_to"]
                  and (it - k["densify_from"]) % k["densify_every"] == 0)
    do_reset = (k["reset_every"] > 0 and it % k["reset_every"] == 0
                and k["densify_from"] <= it <= k["densify_to"])
    if do_densify and do_reset:
        do_densify = False
    do_structure = (k["densify_mode"] == DensificationMode.STRUCTURE_AWARE
                    and do_densify and it % k["sad_structure_every"] == 0)
    do_preview = False
    if k["preview_every"] > 0:
        buffered = [itr + 350 if (k["reset_every"] > 0 and itr % k["reset_every"] == 0)
                    else itr + 50
                    for itr in range(k["preview_every"], k["iterations"],
                                     k["preview_every"])]
        do_preview = (it in buffered) or (it == k["iterations"])
    if k["render_all_every"] > 0:
        do_render_all = (it % k["render_all_every"] == 0) or (it == k["iterations"])
    else:
        do_render_all = do_preview
    do_log = (k["log_every"] > 0
              and ((it % k["log_every"] == 0) or it == 1 or it == k["iterations"]))
    do_eval = k["eval_every"] > 0 and it % k["eval_every"] == 0
    do_psnr = k["psnr_curve_enabled"] and (it + 1) % max(1, k["psnr_curve_every"]) == 0
    do_morton = k["morton_every"] > 0 and it % k["morton_every"] == 0 and not do_densify
    mip_after = k["mip_enabled"] and do_densify
    mip_periodic = (k["mip_enabled"] and it > k["densify_to"]
                    and it % 100 == 0 and it < k["iterations"] - 100)
    return {
        "densify": do_densify, "opacity_reset": do_reset, "structure": do_structure,
        "preview": do_preview, "render_all": do_render_all, "log": do_log,
        "eval": do_eval, "psnr_curve": do_psnr, "morton": do_morton,
        "mip_after_densify": mip_after, "mip_periodic": mip_periodic,
    }


EQUIV_CONFIGS = [
    {},
    {"preview_every": 600},                        # exercises +50 / +350 buffering
    {"morton_every": 100},                         # standalone morton site
    {"psnr_curve_enabled": True, "psnr_curve_every": 100},
    {"render_all_every": 50},
    {"log_every": 25},
    {"mip_enabled": False},
    {"densify_mode": DensificationMode.SIMPLE},
    {"densify_mode": DensificationMode.NONE},
]


@pytest.mark.parametrize("overrides", EQUIV_CONFIGS)
def test_module_matches_inline_exhaustively(overrides) -> None:
    sched = _schedule(**overrides)
    knobs = {**DEFAULT_KNOBS, **overrides}
    for it in range(1, knobs["iterations"] + 1):
        got = {f: getattr(sched.events_at(it), f) for f in EVENT_FIELDS}
        expected = _inline(it, knobs)
        assert got == expected, f"it={it} cfg={overrides}: {got} != {expected}"


def test_events_carry_iteration() -> None:
    ev = _schedule().events_at(1234)
    assert ev.iteration == 1234
    # asdict round-trips cleanly (used by callers that log events).
    assert asdict(ev)["iteration"] == 1234
