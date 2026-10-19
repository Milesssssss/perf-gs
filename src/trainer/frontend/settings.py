# Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

"""The training knobs the CLI and the hub panel BOTH expose - declared once.

A :class:`Setting` is one knob, stated in full exactly once: how argparse should
accept it (flag, type, choices, default, help) and how the panel should render it
(caption, group, drag range, format). ``cli.py`` mounts these rows on its parser
(:meth:`Setting.add_to`) and the viewer's TRAINING SETUP section renders the same
rows, so the two cannot disagree about a type, a default or a set of choices -
there is no second copy to update.

    SETTINGS        - the knobs (a subset of the CLI; the rest of cli.py is
                      launch-only and has no panel widget)
    GROUPS          - the panel's grouping/order, so related knobs sit together
    TrainingSetup   - the lock-protected value store the panel edits

In hub mode the viewer comes up BEFORE a dataset (and therefore before the run) is
chosen, so these stay open. The orchestrator seeds the store from the parsed args
(:meth:`TrainingSetup.from_args`) and calls :meth:`TrainingSetup.apply_to` once,
immediately before the run is built. Nothing reads the values continuously - a run
resolves its schedules and buffer sizes at construction time, which is why the
section stops being editable while a run is being built or trained (see
``SessionSnapshot.can_configure``). It opens back up when the run finishes: the
store is what the NEXT run is built from, so editing it and pressing Train again is
how a run is re-launched with new values.

Adding a knob is ONE row plus one ``add_to`` line in ``cli.py``: it then exists on
the command line AND in the panel, and :func:`validate` (walked by the tests) fails
if the two ever drift apart.

Two settings cannot be honoured as freely as the rest, and the table says so
instead of quietly ignoring them:

  * ``backward_pixel_stride`` is a compile-time shader define, injected into
    ``compiler_options`` when the DEVICE is created - which in hub mode happens
    before the panel even exists. It is therefore ``launch_only``: shown with its
    launch value, not editable.
  * ``stream_max_splats`` sizes the viewer's own gaussian buffers. The snapshot is
    a raw GPU->GPU copy of the trainer's buffers INCLUDING the splat counter, so a
    viewer smaller than the trainer would point its rasterizer past its own
    storage; :func:`viewer_stream_capacity` therefore only ever raises it.
"""
from __future__ import annotations

import argparse
import threading
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from trainer.config.config_types import DensificationMode

# Widget kinds, derived from a Setting's own declaration (see :attr:`Setting.kind`).
INT = "int"
FLOAT = "float"
BOOL = "bool"
CHOICE = "choice"

# Panel groups, in display order. Keeping this list ordered (rather than sorting
# SETTINGS by group) is what guarantees the UI order is a deliberate choice.
GROUPS: tuple[tuple[str, str], ...] = (
    ("schedule", "SCHEDULE"),
    ("densify", "DENSIFICATION"),
    ("loss", "LOSS"),
    ("appearance", "APPEARANCE"),
    ("perf", "PERFORMANCE"),
    ("eval", "EVALUATION"),
    ("stream", "VIEWER STREAM"),
)


@dataclass(frozen=True)
class Setting:
    setting_name: str
    caption: str
    group: str
    type: type = int
    default: Any = 0
    choices: tuple[Any, ...] = ()
    help: str = ""
    # Panel-only: argparse validates no ranges, so the drag domain lives here.
    minimum: float = 0.0
    maximum: float = 0.0
    speed: float = 1.0
    fmt: str = ""
    note: str = ""
    launch_only: bool = False

    @property
    def flag(self) -> str:
        """``densify_from`` -> ``--densify-from``: the dest IS the flag."""
        return "--" + self.setting_name.replace("_", "-")

    @property
    def label(self) -> str:
        return self.caption

    @property
    def kind(self) -> str:
        if self.choices:
            return CHOICE
        if self.type is bool:
            return BOOL
        return INT if self.type is int else FLOAT

    def add_to(self, parser: argparse.ArgumentParser) -> argparse.Action:
        if self.kind == BOOL:
            return parser.add_argument(self.flag, default=self.default,
                                       action=argparse.BooleanOptionalAction,
                                       help=self.help or None)
        return parser.add_argument(self.flag, type=self.type, default=self.default,
                                   choices=list(self.choices) or None,
                                   help=self.help or None)

    def coerce(self, value: Any) -> Any:
        """Clamp/convert an incoming value into this setting's domain."""
        if self.kind == BOOL:
            return bool(value)
        if self.kind == CHOICE:
            return value if value in self.choices else self.choices[0]
        if self.kind == INT:
            return int(min(self.maximum, max(self.minimum, int(value))))
        return float(min(self.maximum, max(self.minimum, float(value))))

    def choice_index(self, value: Any) -> int:
        return self.choices.index(value) if value in self.choices else 0

    def choice_labels(self) -> list[str]:
        return [str(choice) for choice in self.choices]

    def display(self, value: Any) -> str:
        if self.kind == BOOL:
            return "on" if value else "off"
        if self.kind == FLOAT:
            return (self.fmt or "%.3f") % float(value)
        return f"{int(value):,}" if self.kind == INT else str(value)


SETTINGS: tuple[Setting, ...] = (
    # ---- SCHEDULE: how long the run is. The mean-LR decay horizon deliberately
    #      is NOT here: it follows --iters (cli.py --position-lr-max-steps 0).
    Setting("iters", "iterations", "schedule", int, default=3000,
            minimum=100, maximum=200_000, speed=25,
            note="the LR decay horizon follows this"),
    # ---- DENSIFICATION: the growth schedule and its ceiling.
    Setting("densify_mode", "strategy", "densify", str,
            default=DensificationMode.STRUCTURE_AWARE.value,
            choices=tuple(m.value for m in DensificationMode),
            help="densification strategy the loop runs on a densify tick "
                 "(default: structure_aware). Modes: 'none' = no densification; "
                 "'simple' = reference 3DGS threshold-based clone/split; "
                 "'structure_aware' = SAD-GS anisotropic grid-split "
                 "(arXiv 2604.28016) with an interleaved AbsGS gradient filler on "
                 "the non-structure densify ticks."),
    Setting("densify_from", "first densify iter", "densify", int, default=0,
            minimum=0, maximum=200_000, speed=25),
    Setting("densify_to", "last densify iter", "densify", int, default=1900,
            minimum=0, maximum=200_000, speed=25),
    Setting("densify_every", "densify interval", "densify", int, default=100,
            minimum=1, maximum=10_000, speed=5),
    Setting("reset_every", "opacity reset interval", "densify", int, default=1200,
            help="opacity-reset interval. Big-scale prune is disabled until after "
                 "the first reset.",
            minimum=0, maximum=200_000, speed=25, note="0 = never reset"),
    Setting("max_splats", "splat capacity", "densify", int, default=10_000_000,
            minimum=100_000, maximum=100_000_000, speed=50_000),
    # ---- LOSS.
    Setting("lambda_ssim", "SSIM weight", "loss", float, default=0.2,
            help="SSIM loss weight (3DGS default 0.2); 0 = pure L2",
            minimum=0.0, maximum=1.0, speed=0.005, fmt="%.3f",
            note="0 = pure L1/L2"),
    Setting("log_every", "log every N iters", "loss", int, default=0,
            help="sample the loss (and the single-camera PSNR) every N iters. Each "
                 "sample re-renders the sampled camera and reads the frame back to "
                 "the host, so this is the cadence of a GPU sync, not a free "
                 "counter: it feeds the viewer's LOSS graph and the log lines. "
                 "0 = off (no samples, empty graph).",
            minimum=0, maximum=10_000, speed=10, note="0 = off"),

    # ---- APPEARANCE: view-dependent color + the Mip-Splatting filter.
    Setting("sh_degree", "SH degree", "appearance", int, default=0, choices=(0, 1, 2, 3),
            help="Spherical-harmonics degree for view-dependent color (classic "
                 "3DGS). 0 (default) = view-independent SH band-0 (DC) only, the "
                 "original bit-identical path. 1-3 enables higher bands, stored in "
                 "a separate features_rest buffer allocated only when >0, activated "
                 "one band every --sh-band-interval iters (oneupSHdegree)."),
    Setting("sh_band_interval", "SH band interval", "appearance", int, default=0,
            help="Iterations between activating successive SH bands (classic 3DGS "
                 "oneupSHdegree). Only used with --sh-degree > 0. "
                 "0 (default) = AUTO: scale to --iters so the last band turns on "
                 "by iters//2 (interval = iters // (2*sh_degree)); the classic "
                 "constant 1000 assumes a 30k-iter run and starves the top bands "
                 "on short runs. Pass a positive value to override.",
            minimum=0, maximum=20_000, speed=10, note="0 = auto from iterations"),
    Setting("mip_splatting", "Mip-Splatting", "appearance", bool, default=True,
            help="Enable Mip-Splatting (Yu et al., CVPR 2024): a per-gaussian 3D "
                 "smoothing filter (from the TRAIN cameras) folded into the "
                 "projected covariance, plus the 2D Mip antialiasing filter with "
                 "differentiable opacity compensation. Targets the high-frequency "
                 "detail gap vs SADGS (which uses the same 3D filter). Default on; "
                 "pass --no-mip-splatting for the bit-identical vanilla renderer."),
    # ---- PERFORMANCE: speed-for-quality trades.
    Setting("extinction_hardness", "extinction hardness k", "perf", float, default=1.0,
            help="Compositor extinction hardness k (a shader UNIFORM, so "
                 "no recompile). 1.0 (default) = standard alpha-over "
                 "(bit-identical). k>1 hardens the per-splat transmittance "
                 "to (1-alpha)^k so T decays faster => stronger occlusion "
                 "culling (earlier early-out in forward AND backward) at a "
                 "quality tradeoff. Beer-Lambert-style; requires the "
                 "per-splat backward (the default).",
            minimum=1.0, maximum=16.0, speed=0.05, fmt="%.2f",
            note="1 = standard alpha-over"),
    Setting("extinction_hardness_warmup", "hardness warmup steps", "perf", int, default=500,
            help="Steps to ramp the extinction hardness from 1.0 up to "
                 "--extinction-hardness AFTER the first densification. "
                 "k is held at 1.0 (standard alpha-over) until the first "
                 "densify fires, then log-linearly ramps 1 -> target over "
                 "this many steps (like the mean-LR schedule), so "
                 "extinction only sharpens once geometry starts forming. "
                 "0 = jump straight to the target after the first densify; "
                 "<0 = disable the schedule (constant target from step 0, "
                 "the pre-schedule behavior). No-op when "
                 "--extinction-hardness <= 1.",
            minimum=-1, maximum=20_000, speed=10, note="-1 = no ramp"),
    Setting("backward_pixel_stride", "backward pixel stride", "perf", int, default=1,
            choices=(1, 2, 4, 8, 16),
            help="Stochastic pixel-subsampling stride for the backward "
                 "rasterizer (BACKWARD_PIXEL_STRIDE shader define). "
                 "1 = exact full-pixel backward; N>1 visits 1/N of tile "
                 "pixels per iter (unbiased SGD estimator, rotating "
                 "phase) for a faster backward at some gradient-noise "
                 "cost. Injected as a compile-time define.",
            launch_only=True,
            note="compile-time: set it with --backward-pixel-stride"),
    Setting("stbn_pixel_fraction", "STBN pixel fraction", "perf", float, default=.1,
            help="Fraction rho of pixels each backward supervises, chosen by a "
                 "spatiotemporal blue-noise mask (NVIDIA STBN, re-ranked per "
                 "tile so EXACTLY rho*64 of every 8x8 tile is sampled). 1.0 "
                 "(default) = every pixel, feature off and bit-identical. Below "
                 "1.0 the surviving per-pixel terms are scaled by 1/rho, so the "
                 "gradient stays an unbiased estimate of the full-image one and "
                 "the LR needs no retuning. The mask advances per CAMERA VISIT, "
                 "so repeat visits of one view walk the mask's temporal axis "
                 "instead of resampling the same pixels. Requires the asset: "
                 "python scripts/data/download_stbn.py (non-commercial license). "
                 "Mutually exclusive with --backward-pixel-stride > 1.",
            minimum=0.02, maximum=1.0, speed=0.01, fmt="%.3f",
            note="1 = every pixel (off)"),
    Setting("stbn_warmup_iters", "STBN warmup steps", "perf", int, default=0,
            help="Steps spent ramping the STBN pixel fraction from 1.0 down to "
                 "--stbn-pixel-fraction (log-linear, so it passes through the "
                 "intermediate densities). Sparse supervision is hardest on the "
                 "thin structures early densification is trying to create, so "
                 "starting dense is the cheap hedge. 0 (default) = the target "
                 "fraction from step 1. No-op at --stbn-pixel-fraction 1.0.",
            minimum=0, maximum=20_000, speed=10, note="0 = no ramp"),
    # ---- EVALUATION: the passes that render the WHOLE camera set.
    Setting("evaluation", "eval + final metrics", "eval", bool, default=True,
            help="Run the passes that render every camera: the periodic eval "
                 "sweep (--eval-every, which lands on the LAST iteration whenever "
                 "it divides --iters and is most of the wait after training "
                 "'ends'), the post-training metrics pass and .ply export "
                 "(--compute-final-metrics), and the held-out test PSNR "
                 "(--final-test-psnr). On by default. --no-evaluation is a VETO "
                 "over all three, so a run that only needs the trained model "
                 "stops paying for them; it does not turn anything on that its "
                 "own flag left off, and it does not touch the explicit "
                 "--render-all-every / --psnr-curve opt-ins.",
            note="off = no full-camera-set renders"),
    # ---- VIEWER STREAM: what the live viewer costs.
    Setting("stream_every", "snapshot every N iters", "stream", int, default=10,
            help="push a gaussian snapshot to the in-process viewer every "
                 "N iters. Now a zero-copy GPU->GPU blit, so a fine cadence "
                 "is cheap (was coarse when it required a GPU->CPU readback).",
            minimum=1, maximum=1_000, speed=1),
    Setting("stream_max_splats", "viewer capacity", "stream", int, default=7_000_000,
            help="capacity (splats) of the in-process viewer's own gaussian "
                 "buffers. The snapshot is a raw GPU->GPU copy of the "
                 "trainer's buffers INCLUDING the splat counter, so the "
                 "viewer can never be smaller than --max-splats: this only "
                 "ever RAISES its capacity (effective value = max of the "
                 "two). Truncation for viewing is no longer possible (it "
                 "belonged to the old shared-memory stream, which needed a "
                 "host-side readback of the count).",
            minimum=100_000, maximum=100_000_000, speed=50_000,
            note="raised to the splat capacity if lower"),
)

SETTINGS_MAP: dict[str, Setting] = {s.setting_name: s for s in SETTINGS}


def group_settings(group: str) -> list[Setting]:
    """The settings of one group, in table order."""
    return [s for s in SETTINGS if s.group == group]


def captions(settings: Sequence[Setting]) -> list[str]:
    """Small helper for tests/diagnostics: the captions of a group, in order."""
    return [s.caption for s in settings]


def validate() -> list[str]:
    """Every way a row can be wrong or unmounted, as a list of complaints.

    Sharing one object makes type/choices/default drift impossible, so what is
    left to check is internal: that the panel's drag domain CONTAINS the default
    (a range excluding it would silently clamp the documented default the moment
    the panel touched it), and that ``cli.py`` actually mounted the row - a knob
    in the table but not on the parser would give a widget wired to nothing.
    """
    from trainer.frontend.cli import build_parser  # local: cli.py imports us

    mounted = set(vars(build_parser().parse_args([])))
    problems: list[str] = []
    for setting in SETTINGS:
        if setting.setting_name not in mounted:
            problems.append(f"{setting.setting_name}: in SETTINGS but never added to the parser - add `BY_DEST[{setting.setting_name!r}].add_to(parser)`")
        if setting.kind in (INT, FLOAT):
            if not setting.minimum <= setting.default <= setting.maximum:
                problems.append(f"{setting.setting_name}: default {setting.default} is "
                                f"outside the panel range "
                                f"[{setting.minimum}, {setting.maximum}]")
            if setting.speed <= 0:
                problems.append(f"{setting.setting_name}: drag speed must be positive")
        elif setting.kind == CHOICE and setting.default not in setting.choices:
            problems.append(f"{setting.setting_name}: default {setting.default!r} is not "
                            f"one of {setting.choices}")
        if setting.group not in dict(GROUPS):
            problems.append(f"{setting.setting_name}: unknown group {setting.group!r}")
    return problems


def viewer_stream_capacity(max_splats: int, stream_max_splats: int) -> int:
    return max(int(max_splats), int(stream_max_splats))


class TrainingSetup:
    def __init__(self, values: dict[str, Any] | None = None) -> None:
        self._lock = threading.Lock()
        self._values: dict[str, Any] = {s.setting_name: s.coerce(s.default) for s in SETTINGS}
        if values:
            for dest, value in values.items():
                self.set(dest, value)

    @classmethod
    def from_args(cls, args: Any) -> TrainingSetup:
        setup = cls()
        for setting in SETTINGS:
            setup.set(setting.setting_name, getattr(args, setting.setting_name))
        return setup

    def get(self, dest: str) -> Any:
        with self._lock:
            return self._values[dest]

    def set(self, dest: str, value: Any) -> Any:
        """Store a coerced value (out-of-range input is clamped, never rejected)."""
        setting = SETTINGS_MAP[dest]
        coerced = setting.coerce(value)
        with self._lock:
            self._values[dest] = coerced
        return coerced

    def index_of(self, dest: str) -> int:
        """Combo index of a CHOICE setting's current value."""
        return SETTINGS_MAP[dest].choice_index(self.get(dest))

    def set_index(self, dest: str, index: int) -> Any:
        """Store a CHOICE setting from a combo index."""
        choices = SETTINGS_MAP[dest].choices
        return self.set(dest, choices[int(index) % len(choices)])

    def values(self) -> dict[str, Any]:
        with self._lock:
            return dict(self._values)

    # ---- hand over to the run ----

    def apply_to(self, args: Any) -> list[str]:
        changes: list[str] = []
        for setting in SETTINGS:
            if setting.launch_only:
                continue
            new = self.get(setting.setting_name)
            old = getattr(args, setting.setting_name, None)
            if old != new:
                changes.append(f"{setting.setting_name}: "
                               f"{setting.display(old) if old is not None else '-'}"
                               f" -> {setting.display(new)}")
            setattr(args, setting.setting_name, new)
        return changes
