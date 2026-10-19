# Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

"""The plan: which cells exist, and exactly how each is invoked.

Single source of truth for the suite. The manifest, the resume logic and every
table are derived from the cell list built here, so "what alpha did the bonsai
adaptive-STBN cell train with?" has one answer, in one place, recorded per cell.

Three decisions shape it.

**Cells are keyed by configuration, not by experiment.** The pilot, the
hyperparameter diagnostic and the fraction sweep overlap heavily — all three want
adaptive STBN on bicycle at alpha=0.75, sigma=8, rho=0.25 — so there is ONE pool
of uniquely-identified cells and an experiment is a query over it. The adaptive
parameters are part of the key AND of the hash, which is what makes "do not
silently reuse a run whose adaptive settings differ" structural rather than a
convention someone has to remember.

**Instrumentation is uniform across every cell, and that includes the
baselines.** The primary performance metric is the sparse rasterizer backward
time, which needs per-stage GPU timestamps; those cost a few percent of active
training time. Rather than put them on the adaptive arm only — which would hand
the baselines a free speed advantage — every cell in the suite carries the
identical profile. The consequence is stated rather than hidden: no cell of the
existing ``paper_stbn`` grid has a matching configuration (its seed-42 rho=0.25
cells all carry either a convergence ladder or a different timing profile), so
none is reused, and all timing numbers here include the same instrumentation
overhead.

**Density is matched by construction.** Every sparse arm passes the same
``--stbn-pixel-fraction``, and the adaptive path enforces the budget against the
SAME quantized rho the uniform table bakes in (``pixels_per_tile(rho)/64``), so
two arms at "rho = 0.10" really do spend the same expected pixels — the uniform
path's actual density there is 6/64 = 0.09375, not 0.10.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, replace
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]

# Bump when a change to the invocation would invalidate cached cells. Part of
# every hash, so a bump forces the whole suite to be recognized as stale rather
# than silently mixing two training semantics in one table.
SUITE_SCHEMA = "adaptive_stbn/v1"

OUTPUT_ROOT = REPO / "outputs" / "adaptive_stbn"
TRAIN_ROOT = REPO / "outputs" / "train"

DATASET_ROOTS = ("360_v2", "360_extra_scenes", "tandt", "db")

# Reuses the paper suite's family map so downsample and family grouping cannot
# drift between the two studies (the resolutions ARE the published protocol).
from paper_suite import config as paper_cfg                        # noqa: E402

DATASET_FAMILIES = paper_cfg.DATASET_FAMILIES
ALL_SCENES = paper_cfg.SCENES

# --- regimes ---------------------------------------------------------------
#
# The two ladders the whole study rests on, at one matched budget:
#
#   iid_white          -> indep_blue          -> stbn            UNIFORM density
#   adaptive_iid       -> adaptive_indep_blue -> adaptive_stbn    ADAPTIVE density
#
# Reading down a column asks what importance allocation buys; reading along a row
# asks what the blue-noise arrangement buys. The second row exists to answer the
# question the first cannot: do STBN's correlations still help AFTER the sampling
# density has been optimized, or was blue noise only ever compensating for a
# uniform marginal?
#
# adaptive_stbn_biased is a labelled DIAGNOSTIC control, never part of a ladder.
REGIME_FULL = "full"
UNIFORM_REGIMES = ("iid_white", "indep_blue", "stbn")
ADAPTIVE_REGIMES = ("adaptive_iid", "adaptive_indep_blue", "adaptive_stbn")
BIASED_REGIME = "adaptive_stbn_biased"
PILOT_REGIMES = (REGIME_FULL,) + UNIFORM_REGIMES + ADAPTIVE_REGIMES
ALL_REGIMES = PILOT_REGIMES + (BIASED_REGIME,)

REGIME_LABELS = {
    "full": "Dense (100%)",
    "iid_white": "IID white",
    "indep_blue": "Indep. blue noise",
    "stbn": "STBN",
    "adaptive_iid": "Adaptive IID",
    "adaptive_indep_blue": "Adaptive indep. blue",
    "adaptive_stbn": "Adaptive STBN",
    "adaptive_stbn_biased": "Adaptive STBN (biased ctrl)",
}
# The controlled pairs the report reads off, as (treatment, control, axis).
CONTRASTS = (
    ("indep_blue", "iid_white", "spatial (uniform density)"),
    ("stbn", "indep_blue", "temporal (uniform density)"),
    ("adaptive_indep_blue", "adaptive_iid", "spatial (adaptive density)"),
    ("adaptive_stbn", "adaptive_indep_blue", "temporal (adaptive density)"),
    ("adaptive_iid", "iid_white", "importance allocation, IID arrangement"),
    ("adaptive_stbn", "stbn", "importance allocation, STBN arrangement"),
    ("adaptive_stbn_biased", "adaptive_stbn", "dropping the 1/p weight"),
)

# --- the locked configuration ---------------------------------------------
#
# ONE global setting for every scene and every seed. Chosen from the two-scene
# hyperparameter diagnostic and then locked; per-scene tuning would make every
# cross-scene mean a report on the tuning rather than on the method.
PRIMARY_ALPHA = 0.75
PRIMARY_SIGMA = 8.0
PRIMARY_DOWNSAMPLE = 4
PRIMARY_FRACTION = 0.25
PRIMARY_SEED = 42
ITERS = 3000
STAGE_TIMING_EVERY = 10
RENDER_FPS_FRAMES = 60

# --- experiment scopes ----------------------------------------------------
PILOT_SCENES = ("bicycle", "garden", "bonsai", "kitchen", "treehill")
HYPER_SCENES = ("bicycle", "bonsai")
HYPER_ALPHAS = (0.0, 0.5, 0.75, 0.9)
HYPER_SIGMAS = (0.0, 4.0, 8.0, 16.0)
FRACTION_SWEEP = (0.10, 0.15, 0.25)
FRACTION_REGIMES = ("stbn", "adaptive_stbn")
BIASED_SCENES = ("bicycle", "bonsai")

# Frozen-state gradient validation (section 11 of the design).
GRADIENT_SCENES = ("bicycle", "bonsai", "counter")
GRADIENT_STATES = (500, 1500, 3000)
GRADIENT_K_VALUES = (1, 2, 4, 8, 16, 32, 64)
GRADIENT_CAMERAS = 4
# Independent K-sequences per (camera, method). One, not two: the study is
# READBACK-bound, not compute-bound — every K point costs a full staged read of
# the 14 int64 gradient buffers — and the K-curve's variance is already estimated
# across 4 cameras x 3 checkpoints x 3 scenes = 36 sequences per method. A second
# sequence per camera would roughly double the wall clock to halve an error bar
# that is already small relative to the effect being measured.
GRADIENT_SEQUENCES = 1
GRADIENT_SINGLE_MASKS = 4
# Whether to measure the SH bands ABOVE the DC term. Off: they are not one of the
# reported groups (section 11 asks for position, rotation, opacity and SH0, and
# SH0 lives in the scalar buffers), they cannot ride the K-accumulation path at
# all because their buffer is overwritten per backward, and reading them costs
# ~130 MB per measurement at a late checkpoint — which dominated the study's wall
# clock for a number that would then be excluded from every headline anyway.
GRADIENT_MEASURE_SH_REST = False
# Parameter groups reported. ``scale`` is measured and written to the raw CSV but
# kept out of the headline: the existing gradient study found the int64
# fixed-point accumulator saturating on a subset of views, and the design says not
# to build conclusions on a group with known pathological diagnostics. The
# overflow FLAG is carried per row either way, so the exclusion is visible.
GRADIENT_GROUPS = ("position", "rotation", "opacity", "color_sh0")

# Sampling / calibration diagnostics (sections 9 and 10). No GPU.
CALIBRATION_VISITS = 512
CALIBRATION_SIZE = 128
RESIDUAL_VISITS = 64

# The eval configuration, held IDENTICAL across every cell of the suite.
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
    # No mid-loop eval sweeps: observation_s == 0, so train_active_s and
    # train_wall_s coincide and the timing needs no correction.
    "--eval-every", "0",
)


def scene_dir(scene: str) -> Path | None:
    for root in DATASET_ROOTS:
        candidate = REPO / "datasets" / root / scene
        if candidate.is_dir():
            return candidate
    return None


def scene_family(scene: str) -> str:
    return paper_cfg.scene_family(scene)


def scene_downsample(scene: str) -> int:
    return paper_cfg.scene_downsample(scene)


def family_group(scene: str) -> str:
    return paper_cfg.family_group(scene)


def fraction_tag(fraction: float) -> str:
    """``0.25 -> f0250``: the fraction in zero-padded permille.

    Injective and stable under float formatting, because it is part of a
    directory name and therefore a cache key.
    """
    return f"f{int(round(float(fraction) * 1000)):04d}"


def _param_tag(value: float) -> str:
    """``0.75 -> 075``, ``16.0 -> 1600``: two decimals, no separator."""
    return f"{int(round(float(value) * 100)):03d}"


def regime_flags(regime: str, fraction: float, *, alpha: float, sigma: float,
                 downsample: int) -> tuple[str, ...]:
    """The trainer flags that DEFINE a regime.

    ``full`` pins the fraction to 1.0, which turns the mask off entirely. The
    three uniform regimes are exactly the pre-existing ones — same flags, same
    values — so this suite's baselines are the published baselines and not a
    re-parameterization of them. The adaptive regimes add ``--stbn-adaptive`` plus
    the importance parameters and select their uniform sequence explicitly.
    """
    if regime == REGIME_FULL:
        return ("--stbn-pixel-fraction", "1.0")
    value = f"{float(fraction):.4f}"
    common = ("--stbn-pixel-fraction", value,
              "--stbn-select-mode", "global")
    if regime == "iid_white":
        return (*common, "--stbn-source", "white", "--stbn-independent")
    if regime == "indep_blue":
        return (*common, "--stbn-source", "nvidia", "--stbn-independent")
    if regime == "stbn":
        return (*common, "--stbn-source", "nvidia")
    if not regime.startswith("adaptive"):
        raise ValueError(f"unknown regime {regime!r}")

    adaptive = ("--stbn-adaptive",
                "--stbn-importance-alpha", f"{float(alpha):.4f}",
                "--stbn-importance-sigma", f"{float(sigma):.4f}",
                "--stbn-importance-downsample", str(int(downsample)))
    if regime == "adaptive_iid":
        # The uniform is a fresh hashed field per visit, so the volume source is
        # irrelevant; 'white' is passed so no run silently depends on the
        # NON-COMMERCIAL NVIDIA asset being present when it does not need it.
        return (*common, "--stbn-source", "white", "--stbn-independent",
                *adaptive, "--stbn-adaptive-uniform", "iid")
    if regime == "adaptive_indep_blue":
        return (*common, "--stbn-source", "nvidia", "--stbn-independent",
                *adaptive, "--stbn-adaptive-uniform", "blue_independent")
    if regime == "adaptive_stbn":
        return (*common, "--stbn-source", "nvidia",
                *adaptive, "--stbn-adaptive-uniform", "blue_temporal")
    if regime == BIASED_REGIME:
        return (*common, "--stbn-source", "nvidia",
                *adaptive, "--stbn-adaptive-uniform", "blue_temporal",
                "--stbn-adaptive-biased")
    raise ValueError(f"unknown regime {regime!r}")


@dataclass(frozen=True)
class Cell:
    """One training run, fully determined by these fields."""

    scene: str
    regime: str
    seed: int = PRIMARY_SEED
    fraction: float = PRIMARY_FRACTION
    alpha: float = PRIMARY_ALPHA
    sigma: float = PRIMARY_SIGMA
    # Divisor for the IMPORTANCE FIELD's resolution, not for the images. The
    # COLMAP resolution divisor is scene-derived; see :attr:`scene_downsample`.
    importance_downsample: int = PRIMARY_DOWNSAMPLE
    iters: int = ITERS
    # Bookkeeping only — never part of the identity or the hash, so adding a
    # consumer cannot invalidate a cached run.
    consumers: tuple[str, ...] = field(default=(), compare=False)

    @property
    def is_adaptive(self) -> bool:
        return self.regime.startswith("adaptive")

    @property
    def key(self) -> str:
        """Stable cell identity, also the run directory name.

        The importance parameters appear ONLY for adaptive cells: tagging a dense
        or uniform cell with an alpha it ignores would split one shared baseline
        into several identical runs. For adaptive cells they always appear, even
        at the locked defaults, so a directory can never look like it belongs to
        different settings than it does.
        """
        parts = ["adaptive", self.scene, self.regime, fraction_tag(self.fraction),
                 f"s{self.seed}", f"i{self.iters}", f"d{self.scene_downsample}"]
        if self.is_adaptive:
            parts += [f"a{_param_tag(self.alpha)}",
                      f"g{_param_tag(self.sigma)}",
                      f"k{int(self.importance_downsample)}"]
        return "_".join(parts)

    @property
    def scene_downsample(self) -> int:
        """The COLMAP resolution divisor (not the importance-field divisor)."""
        return scene_downsample(self.scene)

    @property
    def family(self) -> str:
        return scene_family(self.scene)

    @property
    def family_group(self) -> str:
        return family_group(self.scene)

    @property
    def run_dir(self) -> Path:
        return TRAIN_ROOT / self.key

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
            "--downsample", str(self.scene_downsample),
            *BASE_ARGS,
            *regime_flags(self.regime, self.fraction, alpha=self.alpha,
                          sigma=self.sigma,
                          downsample=self.importance_downsample),
            # Uniform on EVERY cell — see the module docstring.
            "--stage-timing-every", str(STAGE_TIMING_EVERY),
            "--save-test-renders",
            "--render-fps-frames", str(RENDER_FPS_FRAMES),
            "--run-name", self.key,
        ]

    def config_hash(self, python: str = "python") -> str:
        """Hash of everything that determines this cell's RESULT.

        The interpreter path and the absolute scene path are excluded — they are
        machine-specific and would make a cached grid unusable after a checkout
        moves — but the scene NAME is kept. The suite schema is included, so a
        deliberate semantics change invalidates every cell at once.
        """
        argv = self.command(python)
        trimmed = argv[2:]
        if "--scene" in trimmed:
            index = trimmed.index("--scene")
            trimmed = trimmed[:index + 1] + [self.scene] + trimmed[index + 2:]
        payload = json.dumps({"schema": SUITE_SCHEMA, "argv": trimmed},
                            sort_keys=True)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


# ---------------------------------------------------------------------------
# The cell pool
# ---------------------------------------------------------------------------

def build_cells(*, scenes: tuple[str, ...] | None = None,
                stages: tuple[str, ...] = ("training", "hyper", "fractions",
                                           "biased")) -> list[Cell]:
    """Every training cell the requested stages need, deduplicated.

    Deduplication is by ``Cell`` equality (which excludes ``consumers``), so the
    pilot's ``adaptive_stbn`` cell at the locked parameters is ALSO the
    hyperparameter diagnostic's alpha=0.75 point and the fraction sweep's
    rho=0.25 point — trained once, and therefore guaranteed to be the same number
    in all three tables.
    """
    pool: dict[Cell, list[str]] = {}

    def add(cell: Cell, consumer: str) -> None:
        pool.setdefault(cell, []).append(consumer)

    pilot = tuple(scenes) if scenes else PILOT_SCENES

    if "training" in stages:
        for scene in pilot:
            for regime in PILOT_REGIMES:
                add(Cell(scene=scene, regime=regime,
                         fraction=(1.0 if regime == REGIME_FULL
                                   else PRIMARY_FRACTION)), "training")

    if "hyper" in stages:
        for scene in HYPER_SCENES:
            for alpha in HYPER_ALPHAS:
                add(Cell(scene=scene, regime="adaptive_stbn", alpha=alpha),
                    "hyper_alpha")
            for sigma in HYPER_SIGMAS:
                add(Cell(scene=scene, regime="adaptive_stbn", sigma=sigma),
                    "hyper_sigma")

    if "fractions" in stages:
        for scene in pilot:
            for regime in FRACTION_REGIMES:
                for fraction in FRACTION_SWEEP:
                    add(Cell(scene=scene, regime=regime, fraction=fraction),
                        "fractions")

    if "biased" in stages:
        for scene in BIASED_SCENES:
            add(Cell(scene=scene, regime=BIASED_REGIME), "biased")

    return [replace(cell, consumers=tuple(sorted(set(consumers))))
            for cell, consumers in
            sorted(pool.items(), key=lambda kv: kv[0].key)]


def plan_summary(cells: list[Cell]) -> dict:
    """Counts by consumer / regime / fraction, for the log and the manifest."""
    def tally(fn) -> dict:
        out: dict[str, int] = {}
        for cell in cells:
            out[str(fn(cell))] = out.get(str(fn(cell)), 0) + 1
        return dict(sorted(out.items()))

    consumers: dict[str, int] = {}
    for cell in cells:
        for consumer in cell.consumers:
            consumers[consumer] = consumers.get(consumer, 0) + 1
    return {
        "n_cells": len(cells),
        "by_consumer": dict(sorted(consumers.items())),
        "by_regime": tally(lambda c: c.regime),
        "by_scene": tally(lambda c: c.scene),
        "by_fraction": tally(lambda c: fraction_tag(c.fraction)),
        "locked": {"alpha": PRIMARY_ALPHA, "sigma": PRIMARY_SIGMA,
                   "downsample": PRIMARY_DOWNSAMPLE, "seed": PRIMARY_SEED,
                   "iters": ITERS},
    }


# ---------------------------------------------------------------------------
# Output layout (design section 19)
# ---------------------------------------------------------------------------

def output_paths(root: Path | None = None) -> dict:
    root = Path(root) if root is not None else OUTPUT_ROOT
    paths = {
        "root": root,
        "manifest": root / "manifest.json",
        "raw": root / "raw",
        "gradient_validation": root / "gradient_validation",
        "sampling_validation": root / "sampling_validation",
        "training": root / "training",
        "figures": root / "figures",
        "tables": root / "tables",
        "logs": root / "logs",
        "report": root / "report.md",
    }
    for key, path in paths.items():
        if key not in ("root", "manifest", "report"):
            path.mkdir(parents=True, exist_ok=True)
    root.mkdir(parents=True, exist_ok=True)
    return paths
