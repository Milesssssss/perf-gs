# Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

"""The experiment plan: what cells exist, and exactly how each is invoked.

Cells are keyed by configuration, not experiment; an experiment is a query over one deduplicated pool.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, replace
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]

# Bump when an invocation change invalidates cached cells; it is part of every cell's hash.
SUITE_SCHEMA = "paper_stbn/v2"

# 3DGS eval resolutions; --downsample divides the COLMAP camera size, and T&T's JPEGs are half of it.
DOWNSAMPLE_BY_FAMILY: dict[str, int] = {
    "mipnerf360_outdoor": 4,
    "mipnerf360_indoor": 2,
    "tanks_and_temples": 2,
    "deep_blending": 1,
}
CAMERA_WIDTH_CAP = 1600

OUTPUT_ROOT = REPO / "outputs" / "paper_stbn"
TRAIN_ROOT = REPO / "outputs" / "train"

# Families the 3DGS literature reports on; scene names are unique across families.
DATASET_FAMILIES: dict[str, tuple[str, ...]] = {
    "mipnerf360_outdoor": ("bicycle", "flowers", "garden", "stump", "treehill"),
    "mipnerf360_indoor": ("bonsai", "counter", "kitchen", "room"),
    "tanks_and_temples": ("train", "truck"),
    "deep_blending": ("drjohnson", "playroom"),
}
SCENES: tuple[str, ...] = tuple(
    s for scenes in DATASET_FAMILIES.values() for s in scenes)
DATASET_ROOTS = ("360_v2", "360_extra_scenes", "tandt", "db")

# Mip-NeRF 360 is normally reported as one dataset, with indoor/outdoor as a sub-analysis.
FAMILY_GROUPS: dict[str, tuple[str, ...]] = {
    "mipnerf360": ("mipnerf360_outdoor", "mipnerf360_indoor"),
    "tanks_and_temples": ("tanks_and_temples",),
    "deep_blending": ("deep_blending",),
}

REGIMES: tuple[str, ...] = ("full", "iid_white", "indep_blue", "stbn")
SPARSE_REGIMES: tuple[str, ...] = ("iid_white", "indep_blue", "stbn")
REGIME_LABELS = {
    "full": "Dense (100%)",
    "iid_white": "IID white",
    "indep_blue": "Indep.\\ blue noise",
    "stbn": "STBN",
}

SEEDS: tuple[int, ...] = (42, 1337, 2026)
PRIMARY_SEED = 42

# "global" thresholds the whole slice, as the NVIDIA asset was optimized; "tile_rank" is Experiment S's ablation.
SELECT_GLOBAL = "global"
SELECT_TILE_RANK = "tile_rank"
PRIMARY_SELECT_MODE = SELECT_GLOBAL
SELECT_MODE_LABELS = {
    SELECT_GLOBAL: "Global slice threshold",
    SELECT_TILE_RANK: r"Per-tile rank (exactly $k$/tile)",
}
PRIMARY_FRACTION = 0.25
SWEEP_FRACTIONS: tuple[float, ...] = (0.10, 0.25, 0.50)
ITERS = 3000

# Experiment C's checkpoint ladder.
CONVERGENCE_ITERS: tuple[int, ...] = (250, 500, 750, 1000, 1500, 2000, 2500, 3000)
# Experiment C's time-to-quality targets, in dB below the dense final PSNR.
QUALITY_TARGETS_DB: tuple[float, ...] = (0.5, 0.2, 0.1)

# Experiment D's detailed-timing scenes: one per dataset family.
TIMING_SCENES: tuple[str, ...] = ("bicycle", "kitchen", "train", "drjohnson")
STAGE_TIMING_EVERY = 10
# Experiment D's reporting windows over the 3k run.
TIMING_WINDOWS: tuple[tuple[int, int], ...] = ((1, 1000), (1001, 2000), (2001, 3000))

# Experiment F's representative scenes.
GRADIENT_SCENES: tuple[str, ...] = ("bicycle", "treehill", "kitchen", "train",
                                    "drjohnson")
GRADIENT_K_VALUES: tuple[int, ...] = (1, 2, 4, 8, 16)
# Early (densification active), mid and late (refinement) training states.
GRADIENT_STATES: tuple[int, ...] = (500, 1500, 3000)

# Experiment E.
SAMPLING_VISITS = 64
SAMPLING_COVERAGE_POINTS: tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64)

RENDER_FPS_FRAMES = 60
BOOTSTRAP_RESAMPLES = 10000
BOOTSTRAP_SEED = 20260818

# Identical for every cell (default zero STBN warmup); --evaluation is required because it vetoes --final-test-psnr.
BASE_ARGS: tuple[str, ...] = (
    "--sh-degree", "3",
    "--densify-mode", "structure_aware",
    "--mip-splatting",
    "--eval", "1",
    "--eval-holdout-stride", "8",
    "--no-viewer",
    "--log-every", "0",
    "--preview-every", "0",
    "--no-write-ground-truths",
    "--evaluation",
    "--final-test-psnr",
    "--run-report",
    "--eval-lpips",
)


def fraction_tag(fraction: float) -> str:
    """Zero-padded permille (``0.25 -> f0250``): an injective, density-sorted cache-key component."""
    return f"f{int(round(float(fraction) * 1000)):04d}"


def scene_downsample(scene: str) -> int:
    return DOWNSAMPLE_BY_FAMILY.get(scene_family(scene), 4)


def scene_family(scene: str) -> str:
    for family, scenes in DATASET_FAMILIES.items():
        if scene in scenes:
            return family
    return "unknown"


def family_group(scene: str) -> str:
    family = scene_family(scene)
    for group, families in FAMILY_GROUPS.items():
        if family in families:
            return group
    return "unknown"


def scene_dir(scene: str) -> Path | None:
    for root in DATASET_ROOTS:
        candidate = REPO / "datasets" / root / scene
        if candidate.is_dir():
            return candidate
    return None


def regime_flags(regime: str, fraction: float,
                 select_mode: str = PRIMARY_SELECT_MODE) -> tuple[str, ...]:
    """Trainer flags defining a regime; dense omits the select mode so one cell serves both stratification arms."""
    if regime == "full":
        return ("--stbn-pixel-fraction", "1.0")
    value = f"{float(fraction):.4f}"
    mode = ("--stbn-select-mode", select_mode)
    if regime == "iid_white":
        return ("--stbn-pixel-fraction", value,
                "--stbn-source", "white", "--stbn-independent", *mode)
    if regime == "indep_blue":
        return ("--stbn-pixel-fraction", value,
                "--stbn-source", "nvidia", "--stbn-independent", *mode)
    if regime == "stbn":
        return ("--stbn-pixel-fraction", value, "--stbn-source", "nvidia", *mode)
    raise ValueError(f"unknown regime {regime!r}")


@dataclass(frozen=True)
class Cell:
    """One training run, fully determined by these fields (``fraction`` is 1.0 for ``full``)."""

    scene: str
    regime: str
    seed: int
    fraction: float
    iters: int = ITERS
    # Part of the identity: the two modes are different sampling processes and must never be pooled.
    select_mode: str = PRIMARY_SELECT_MODE
    # Instrumentation is part of the identity: convergence sweeps change a cell's wall-clock.
    convergence: bool = False
    stage_timing: bool = False
    save_renders: bool = True
    write_ground_truths: bool = False
    render_fps: bool = True
    # Bookkeeping only, outside the identity and hash, so adding a consumer cannot invalidate a cached run.
    consumers: tuple[str, ...] = field(default=(), compare=False)

    @property
    def key(self) -> str:
        """Stable cell identity and run directory name; sparse keys spell out the select mode for both modes."""
        parts = [self.scene, self.regime, fraction_tag(self.fraction),
                 f"s{self.seed}", f"i{self.iters}", f"d{self.downsample}"]
        # Dense ignores the mask, so tagging it would split one shared baseline into two identical runs.
        if self.regime != "full":
            parts.append(self.select_mode)
        if self.convergence:
            parts.append("conv")
        if self.stage_timing:
            parts.append("timing")
        return "paper_" + "_".join(parts)

    @property
    def family(self) -> str:
        return scene_family(self.scene)

    @property
    def downsample(self) -> int:
        return scene_downsample(self.scene)

    @property
    def family_group(self) -> str:
        return family_group(self.scene)

    @property
    def run_dir(self) -> Path:
        return TRAIN_ROOT / self.key

    def instrumentation_args(self) -> tuple[str, ...]:
        args: list[str] = []
        if self.convergence:
            # Explicit ladder: the points are irregular, and an interval would add costly unrequested sweeps.
            args += ["--eval-every", "0", "--eval-iters",
                     *[str(i) for i in CONVERGENCE_ITERS if i <= self.iters]]
        else:
            # No mid-loop sweeps, so train_wall_s equals train_active_s and needs no correction.
            args += ["--eval-every", "0"]
        if self.stage_timing:
            args += ["--stage-timing-every", str(STAGE_TIMING_EVERY)]
        if self.save_renders:
            args += ["--save-test-renders"]
        if self.write_ground_truths:
            # Overrides BASE_ARGS; one cell per scene dumps the exact ground truth for the qualitative study.
            args += ["--write-ground-truths"]
        if self.render_fps:
            args += ["--render-fps-frames", str(RENDER_FPS_FRAMES)]
        return tuple(args)

    def command(self, python: str) -> list[str]:
        """The exact argv for this cell. Recorded verbatim in the manifest."""
        directory = scene_dir(self.scene)
        if directory is None:
            raise FileNotFoundError(
                f"scene {self.scene!r} not found under "
                + ", ".join(f"datasets/{r}" for r in DATASET_ROOTS))
        return [
            python, str(REPO / "scripts" / "running" / "v2_train.py"),
            "--scene", str(directory),
            "--iters", str(self.iters),
            "--seed", str(self.seed),
            "--downsample", str(self.downsample),
            *BASE_ARGS,
            *regime_flags(self.regime, self.fraction, self.select_mode),
            *self.instrumentation_args(),
            "--run-name", self.key,
        ]

    def config_hash(self, python: str = "python") -> str:
        """Hash of everything that determines this cell's result; machine-specific paths are excluded."""
        argv = self.command(python)
        trimmed = argv[2:]
        if "--scene" in trimmed:
            index = trimmed.index("--scene")
            trimmed = trimmed[:index + 1] + [self.scene] + trimmed[index + 2:]
        payload = json.dumps({"schema": SUITE_SCHEMA, "argv": trimmed},
                             sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()[:16]

    def with_consumer(self, name: str) -> "Cell":
        if name in self.consumers:
            return self
        return replace(self, consumers=tuple(sorted({*self.consumers, name})))


def build_plan(scenes: tuple[str, ...] = SCENES,
               seeds: tuple[int, ...] = SEEDS,
               fractions: tuple[float, ...] = SWEEP_FRACTIONS,
               iters: int = ITERS,
               *,
               with_timing: bool = True,
               timing_scenes: tuple[str, ...] = TIMING_SCENES,
               with_stratification: bool = True) -> list[Cell]:
    """Every unique cell the suite needs; overlapping experiment requirements resolve to one cell with several consumers."""
    cells: dict[str, Cell] = {}

    def add(cell: Cell, consumer: str) -> None:
        existing = cells.get(cell.key)
        cells[cell.key] = (existing or cell).with_consumer(consumer)

    # The one cell per scene that also dumps ground truth, for Experiment G.
    gt_owner = {scene: ("full", PRIMARY_SEED) for scene in scenes}

    for scene in scenes:
        # Experiment A: 4 regimes x every seed at the primary fraction.
        for seed in seeds:
            primary = (seed == PRIMARY_SEED)
            for regime in REGIMES:
                is_gt_owner = (primary
                               and gt_owner.get(scene) == (regime, seed))
                cell = Cell(
                    scene=scene, regime=regime, seed=seed,
                    # Pinned to 1.0 for dense so one dense cell is shared by every Experiment B fraction.
                    fraction=(1.0 if regime == "full" else PRIMARY_FRACTION),
                    iters=iters,
                    # Only the primary seed is observed; the others' wall-clock is a clean speed reference.
                    convergence=primary,
                    save_renders=True,
                    write_ground_truths=is_gt_owner,
                )
                add(cell, "A")
                if primary:
                    add(cell, "C")
                    if regime != "full":
                        # Also the rho=0.25 column of the budget sweep.
                        add(cell, "B")
                    else:
                        add(cell, "B")   # shared dense reference
                    add(cell, "G")

        # Experiment B: the other fractions, primary seed only.
        for fraction in fractions:
            if abs(fraction - PRIMARY_FRACTION) < 1e-9:
                continue          # already covered above
            for regime in SPARSE_REGIMES:
                add(Cell(scene=scene, regime=regime, seed=PRIMARY_SEED,
                         fraction=fraction, iters=iters), "B")

    # Experiment S: per-tile-rank selection, paired with the global arm on the same (scene, seed, rho).
    if with_stratification:
        for scene in scenes:
            for regime in SPARSE_REGIMES:
                add(Cell(scene=scene, regime=regime, seed=PRIMARY_SEED,
                         fraction=PRIMARY_FRACTION, iters=iters,
                         select_mode=SELECT_TILE_RANK), "S")

    # Experiment D: separate timing cells, since the timestamp drain must not land on runs with reported timings.
    if with_timing:
        for scene in timing_scenes:
            if scene not in scenes:
                continue
            for regime in REGIMES:
                add(Cell(scene=scene, regime=regime, seed=PRIMARY_SEED,
                         fraction=(1.0 if regime == "full" else PRIMARY_FRACTION),
                         iters=iters, stage_timing=True, render_fps=False), "D")

    return [cells[key] for key in sorted(cells)]


def plan_summary(cells: list[Cell]) -> dict:
    """Counts by consumer / regime / fraction, for the log and the manifest."""
    def tally(attr) -> dict:
        out: dict[str, int] = {}
        for cell in cells:
            out[str(attr(cell))] = out.get(str(attr(cell)), 0) + 1
        return dict(sorted(out.items()))

    consumers: dict[str, int] = {}
    for cell in cells:
        for consumer in cell.consumers:
            consumers[consumer] = consumers.get(consumer, 0) + 1
    return {
        "total_cells": len(cells),
        "by_experiment": dict(sorted(consumers.items())),
        "by_regime": tally(lambda c: c.regime),
        "by_seed": tally(lambda c: c.seed),
        "by_fraction": tally(lambda c: c.fraction),
        "by_select_mode": tally(lambda c: c.select_mode),
        "by_downsample": tally(lambda c: c.downsample),
        "with_convergence": sum(1 for c in cells if c.convergence),
        "with_stage_timing": sum(1 for c in cells if c.stage_timing),
        "with_saved_renders": sum(1 for c in cells if c.save_renders),
    }
