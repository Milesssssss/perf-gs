"""The experiment plan: what cells exist, and exactly how each is invoked.

This module is the single source of truth for the suite. Everything downstream —
the manifest, the resume logic, every table and figure — is derived from the cell
list built here, so a question like "was STBN at rho=0.1 on treehill trained with
Mip-Splatting on?" has one answer, in one place, recorded per cell.

Two decisions shape the whole design.

**Cells are keyed by CONFIGURATION, not by experiment.** Experiment A wants
13x4x3 cells at rho=0.25; Experiment B wants rho in {0.10, 0.25, 0.50} at seed 42
and explicitly reuses A's rho=0.25 runs and A's dense baseline (density does not
apply to dense supervision). Modelling cells per experiment would train the
overlap twice and — worse — invite the two copies to disagree. So there is ONE
pool of uniquely-identified cells, and an experiment is a *query* over it.

**Instrumentation is a per-cell profile, and it is never free.** Reading GPU
timestamps needs a host drain; an evaluation sweep renders and scores the entire
held-out set. Both are wall-clock the timing columns must not silently absorb. So
the seed-42 cells carry the heavy observation (they feed the convergence and
qualitative studies, and their eval cost is measured and subtracted), while seeds
1337/2026 run unobserved and are the clean reference for speed. Detailed
per-stage timing is restricted to dedicated cells on four representative scenes,
because its measured overhead is a few percent of active training time — not
negligible enough to put on a run whose timing is a reported result.

The controlled contrasts the whole paper rests on, and what makes them valid:

    iid_white  -> indep_blue   what the blue-noise ARRANGEMENT buys
    indep_blue -> stbn         what the TEMPORAL ordering buys

Both sparse regimes draw from the same 64-slice pool with the same per-tile
count, the same 1/rho reweighting, and the same compacted backward. iid_white and
indep_blue differ only in whether the slices were ranked from white noise or from
the NVIDIA STBN volume; indep_blue and stbn differ only in whether the slice
index is drawn independently per step or walks each camera's own visit counter.
Nothing else moves.

One honesty note that belongs in the paper: ``iid_white`` is per-tile STRATIFIED
white noise (exactly k of every 8x8 tile), not unstratified Bernoulli sampling.
That is deliberate — it holds the pixel count, the reweighting and every
downstream code path fixed — but it means the IID control already has some
spatial regularity, so the arrangement contrast is a LOWER bound on what blue
noise buys over truly independent sampling.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, replace
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]

# Bump when a change to the invocation would invalidate cached cells. The hash
# includes this, so a bump forces every cell to be recognized as stale rather
# than silently mixing old and new training semantics in one table.
SUITE_SCHEMA = "paper_stbn/v1"

OUTPUT_ROOT = REPO / "outputs" / "paper_stbn"
TRAIN_ROOT = REPO / "outputs" / "train"

# The three dataset families the 3DGS literature reports on, in canonical order.
# Scene names are unique across families, so a scene implies its root.
DATASET_FAMILIES: dict[str, tuple[str, ...]] = {
    "mipnerf360_outdoor": ("bicycle", "flowers", "garden", "stump", "treehill"),
    "mipnerf360_indoor": ("bonsai", "counter", "kitchen", "room"),
    "tanks_and_temples": ("train", "truck"),
    "deep_blending": ("drjohnson", "playroom"),
}
SCENES: tuple[str, ...] = tuple(
    s for scenes in DATASET_FAMILIES.values() for s in scenes)
DATASET_ROOTS = ("360_v2", "360_extra_scenes", "tandt", "db")

# Coarser grouping for the family-aggregate table: Mip-NeRF 360 is normally
# reported as one dataset, with the indoor/outdoor split as a sub-analysis.
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

# How a density picks pixels. "global" thresholds the whole 128x128 slice — the
# point set NVIDIA's spatial energy function was optimized for, and the faithful
# reading of the asset. "tile_rank" keeps exactly k per 8x8 execution tile, which
# is a different sampling process: measured at rho=0.25 it costs ~30% of the blue
# mask's low-frequency suppression AND makes the white control ~24% bluer than
# true white, compressing the blue-vs-white contrast from both ends.
#
# The whole grid runs at "global"; "tile_rank" gets a focused arm (Experiment S)
# so the stratification choice is reported as a measured ablation rather than
# assumed. The mode is part of Cell.key, so the two arms can never share a run
# directory.
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
# Experiment C's time-to-quality targets, in dB BELOW the dense final PSNR.
QUALITY_TARGETS_DB: tuple[float, ...] = (0.5, 0.2, 0.1)

# Experiment D's detailed-timing scenes: one per dataset family, spanning a large
# unbounded outdoor scene, a dense indoor one, and both non-360 datasets.
TIMING_SCENES: tuple[str, ...] = ("bicycle", "kitchen", "train", "drjohnson")
STAGE_TIMING_EVERY = 10
# Experiment D's reporting windows over the 3k run.
TIMING_WINDOWS: tuple[tuple[int, int], ...] = ((1, 1000), (1001, 2000), (2001, 3000))

# Experiment F's representative scenes.
GRADIENT_SCENES: tuple[str, ...] = ("bicycle", "treehill", "kitchen", "train",
                                    "drjohnson")
GRADIENT_K_VALUES: tuple[int, ...] = (1, 2, 4, 8, 16)
# Training states at which the gradient diagnostic is evaluated: early (geometry
# still forming, densification active), mid, and late (refinement).
GRADIENT_STATES: tuple[int, ...] = (500, 1500, 3000)

# Experiment E.
SAMPLING_VISITS = 64
SAMPLING_COVERAGE_POINTS: tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64)

RENDER_FPS_FRAMES = 60
BOOTSTRAP_RESAMPLES = 10000
BOOTSTRAP_SEED = 20260818

# The published eval configuration, held IDENTICAL across every cell. --evaluation
# is required because it vetoes --final-test-psnr. Density is flat (no warmup, the
# --stbn-warmup-iters default) so the arrangement/temporal contrast is not blurred
# by a dense opening. --preview-every 0 and --no-write-ground-truths remove
# observation-only work (a 194-camera PNG dump and a per-run copy of the
# ground truth) that costs minutes and hundreds of MB per cell without affecting
# a single training step.
BASE_ARGS: tuple[str, ...] = (
    "--sh-degree", "3",
    "--densify-mode", "structure_aware",
    "--mip-splatting",
    "--downsample", "4",
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
    """``0.25 -> f0250``, ``0.10 -> f0100``, ``1.0 -> f1000``: the fraction in
    zero-padded permille.

    Permille rather than a stripped decimal because the tag is a directory name
    and therefore a cache key: it has to be injective (a scheme where 0.10 and
    1.00 both render as "100" would silently alias a sparse cell onto the dense
    one) and stable under float formatting. Zero padding keeps the keys sorting
    in density order.
    """
    return f"f{int(round(float(fraction) * 1000)):04d}"


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
    """The trainer flags that DEFINE a regime.

    ``full`` pins the fraction to 1.0, which turns the mask off entirely (see
    ``stbn.pixels_per_tile``); the other three share every flag but the noise
    source and whether the temporal walk is used.

    The selection mode is passed for the sparse regimes only. Dense supervises
    every pixel, so the mode is meaningless there — and omitting it is what lets
    ONE dense cell serve as the baseline for both arms of the stratification
    ablation instead of being trained twice for no difference.
    """
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
    """One training run, fully determined by these fields.

    ``fraction`` is carried even for ``full`` (where it is 1.0 by definition) so
    that every record has the field populated and no table has to special-case a
    missing value.
    """

    scene: str
    regime: str
    seed: int
    fraction: float
    iters: int = ITERS
    # Part of the identity: the two modes are different sampling processes and
    # their runs must never share a directory or be pooled into one mean.
    select_mode: str = PRIMARY_SELECT_MODE
    # Instrumentation profile, part of the identity: a cell with convergence
    # checkpoints is NOT interchangeable with one without them, because its
    # wall-clock contains sweeps the other's does not.
    convergence: bool = False
    stage_timing: bool = False
    save_renders: bool = False
    write_ground_truths: bool = False
    render_fps: bool = True
    # Which experiments consume this cell. Bookkeeping only — never part of the
    # identity or the hash, so adding a consumer cannot invalidate a cached run.
    consumers: tuple[str, ...] = field(default=(), compare=False)

    @property
    def key(self) -> str:
        """Stable cell identity, also the run directory name.

        The selection mode is spelled out for BOTH modes rather than only for the
        non-default one. Suffixing just the ablation would make an older
        per-tile-rank run directory look like a global run to the resume logic,
        which is the one confusion this key exists to prevent.
        """
        parts = [self.scene, self.regime, fraction_tag(self.fraction),
                 f"s{self.seed}", f"i{self.iters}"]
        # Dense ignores the mask entirely, so tagging it would split one shared
        # baseline into two identical runs.
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
    def family_group(self) -> str:
        return family_group(self.scene)

    @property
    def run_dir(self) -> Path:
        return TRAIN_ROOT / self.key

    def instrumentation_args(self) -> tuple[str, ...]:
        args: list[str] = []
        if self.convergence:
            # Explicit ladder rather than --eval-every: the required points are
            # irregular, and any single interval that hit all of them would also
            # fire on points nobody asked for, each costing a full held-out sweep.
            args += ["--eval-every", "0", "--eval-iters",
                     *[str(i) for i in CONVERGENCE_ITERS if i <= self.iters]]
        else:
            # No mid-loop sweeps at all: observation_s == 0, so train_wall_s and
            # train_active_s coincide and the timing needs no correction.
            args += ["--eval-every", "0"]
        if self.stage_timing:
            args += ["--stage-timing-every", str(STAGE_TIMING_EVERY)]
        if self.save_renders:
            args += ["--save-test-renders"]
        if self.write_ground_truths:
            # Overrides BASE_ARGS' --no-write-ground-truths. One cell per scene
            # does this so the qualitative study has the exact ground truth the
            # trainer scored against, without every cell duplicating it.
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
            *BASE_ARGS,
            *regime_flags(self.regime, self.fraction, self.select_mode),
            *self.instrumentation_args(),
            "--run-name", self.key,
        ]

    def config_hash(self, python: str = "python") -> str:
        """Hash of everything that determines this cell's RESULT.

        The interpreter path and the absolute scene path are excluded — they are
        machine-specific and would make a cached grid unusable after a checkout
        moves. The suite schema is included, so a deliberate semantics change
        invalidates every cell at once.
        """
        argv = self.command(python)
        # Drop the interpreter and the script path (argv[0:2]) and the absolute
        # --scene value, keeping the scene NAME instead.
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
    """Every unique cell the suite needs, deduplicated by :attr:`Cell.key`.

    Built by walking the experiments and unioning their requirements, so overlap
    (A's rho=0.25 seed-42 runs are also B's rho=0.25 column; the dense baseline
    is shared by both, since a pixel fraction does not apply to dense
    supervision) resolves to ONE cell with several recorded consumers.
    """
    cells: dict[str, Cell] = {}

    def add(cell: Cell, consumer: str) -> None:
        existing = cells.get(cell.key)
        cells[cell.key] = (existing or cell).with_consumer(consumer)

    # The one cell per scene that also dumps ground truth, for Experiment G.
    gt_owner = {scene: ("full", PRIMARY_SEED) for scene in scenes}

    for scene in scenes:
        # --- Experiment A: 4 regimes x every seed at the primary fraction ---
        for seed in seeds:
            primary = (seed == PRIMARY_SEED)
            for regime in REGIMES:
                is_gt_owner = (primary
                               and gt_owner.get(scene) == (regime, seed))
                cell = Cell(
                    scene=scene, regime=regime, seed=seed,
                    # Dense supervision has no pixel fraction; pin it to 1.0 so
                    # the dense cell is shared by every fraction in Experiment B
                    # instead of being retrained per fraction.
                    fraction=(1.0 if regime == "full" else PRIMARY_FRACTION),
                    iters=iters,
                    # Only the primary seed carries the convergence ladder and
                    # the render dump; the other seeds stay unobserved so their
                    # wall-clock is a clean speed reference.
                    convergence=primary,
                    save_renders=primary,
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

        # --- Experiment B: the other fractions, primary seed only ---
        for fraction in fractions:
            if abs(fraction - PRIMARY_FRACTION) < 1e-9:
                continue          # already covered above
            for regime in SPARSE_REGIMES:
                add(Cell(scene=scene, regime=regime, seed=PRIMARY_SEED,
                         fraction=fraction, iters=iters), "B")

    # --- Experiment S: the stratification ablation --------------------------
    # The same 13 scenes and three sparse regimes at the primary seed and
    # density, but selecting pixels by per-tile rank instead of a global slice
    # threshold. This is the arm that answers "does forcing exactly k per 8x8
    # execution tile matter?" with a measurement instead of an assumption.
    #
    # Scoped to one seed and one density on purpose: it is a question about the
    # SELECTION, and pairing it against the global arm on the same (scene, seed,
    # rho) is what makes it answerable. The dense baseline is shared — a pixel
    # fraction, and therefore a selection mode, does not apply to dense
    # supervision.
    if with_stratification:
        for scene in scenes:
            for regime in SPARSE_REGIMES:
                add(Cell(scene=scene, regime=regime, seed=PRIMARY_SEED,
                         fraction=PRIMARY_FRACTION, iters=iters,
                         select_mode=SELECT_TILE_RANK), "S")

    # --- Experiment D: dedicated per-stage timing cells ---------------------
    # Separate cells rather than instrumenting the main grid: the timestamp
    # drain costs a few percent of ACTIVE training time, which is small but not
    # nothing, and it must not land on runs whose timing is a reported number.
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
        "with_convergence": sum(1 for c in cells if c.convergence),
        "with_stage_timing": sum(1 for c in cells if c.stage_timing),
        "with_saved_renders": sum(1 for c in cells if c.save_renders),
    }
