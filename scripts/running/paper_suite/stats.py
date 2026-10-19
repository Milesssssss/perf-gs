# Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

"""Paired statistics for the ablation, deliberately conservative.

Three choices here decide whether the paper's claims are defensible.

**The pairing unit.** Sampling regimes are compared ON THE SAME SCENE AT THE SAME
SEED. Scene-to-scene variation in PSNR is several dB — an order of magnitude
larger than the effects being measured — so an unpaired comparison of two means
would drown the signal in between-scene variance. Every difference here is
within-(scene, seed).

**The inference unit.** For confidence intervals the independent unit is the
SCENE, not the (scene, seed) cell: three seeds of one scene are repeated measures
of the same thing, and treating them as 39 independent observations would shrink
the interval by ~sqrt(3) for free. So seeds are averaged within a scene first,
and the bootstrap resamples the resulting per-scene differences. Win counts, by
contrast, are reported per (scene, seed), because there the question is "how often
did it help" rather than "how precisely do we know the mean".

**Direction.** PSNR and SSIM are higher-better; LPIPS is lower-better. Every
function takes the direction explicitly rather than assuming, because a sign flip
here would invert a conclusion silently.

Nothing in this module hides a negative result: :func:`paired_summary` reports the
loss count and the worst case alongside the mean, and the interval is reported
whether or not it excludes zero. A difference is called nothing at all — the
report states the mean, the spread, the interval and the win count, and lets those
speak.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

# Metrics and their orientation. Used everywhere a "did it improve?" question is
# asked, so the answer cannot depend on which function is doing the asking.
HIGHER_IS_BETTER = {
    "psnr_srgb": True,
    "ssim_srgb": True,
    "lpips": False,
    "train_active_s": False,
    "train_wall_s": False,
    "n_gaussians": None,        # neither: descriptive only
    "render_fps": True,
    "process_wall_s": False,
}

METRIC_LABELS = {
    "psnr_srgb": "PSNR (dB)",
    "ssim_srgb": "SSIM",
    "lpips": "LPIPS",
    "train_active_s": "active train (s)",
    "train_wall_s": "wall train (s)",
    "n_gaussians": "Gaussians",
    "render_fps": "render FPS",
}


def mean_std(values) -> dict:
    """Mean, SAMPLE standard deviation (ddof=1) and n.

    ddof=1 because these are samples from a larger population of seeds/scenes,
    not the population itself; with n=3 seeds the difference from ddof=0 is 22%,
    which is not a rounding detail in a table that reports +/- values.
    """
    clean = [float(v) for v in values if v is not None and np.isfinite(v)]
    if not clean:
        return {"mean": None, "std": None, "n": 0, "min": None, "max": None}
    array = np.asarray(clean, dtype=float)
    return {
        "mean": float(array.mean()),
        "std": (float(array.std(ddof=1)) if array.size > 1 else 0.0),
        "n": int(array.size),
        "min": float(array.min()),
        "max": float(array.max()),
        "sem": (float(array.std(ddof=1) / math.sqrt(array.size))
                if array.size > 1 else 0.0),
    }


def bootstrap_ci(values, *, alpha: float = 0.05, resamples: int = 10000,
                 seed: int = 20260818) -> dict:
    """Percentile bootstrap CI for the MEAN of ``values``.

    Percentile (not BCa): with ~13 scenes the bias correction is estimated from
    so few points that it adds noise rather than accuracy, and the percentile
    interval is the conventional, easily-described choice. The resample count and
    seed are recorded so the interval is reproducible to the digit.

    n < 3 returns nulls rather than a nonsense interval — an interval from two
    points would look like evidence and be none.
    """
    clean = np.asarray([float(v) for v in values
                        if v is not None and np.isfinite(v)], dtype=float)
    result = {
        "n": int(clean.size),
        "mean": (float(clean.mean()) if clean.size else None),
        "ci_low": None, "ci_high": None,
        "level": 1.0 - alpha, "resamples": int(resamples), "seed": int(seed),
        "method": "percentile bootstrap of the mean",
        "excludes_zero": None,
    }
    if clean.size < 3:
        result["note"] = f"n={clean.size} is too few for a bootstrap interval"
        return result

    rng = np.random.default_rng(seed)
    draws = rng.integers(0, clean.size, size=(resamples, clean.size))
    means = clean[draws].mean(axis=1)
    low = float(np.percentile(means, 100.0 * alpha / 2.0))
    high = float(np.percentile(means, 100.0 * (1.0 - alpha / 2.0)))
    result["ci_low"] = low
    result["ci_high"] = high
    result["excludes_zero"] = bool(low > 0.0 or high < 0.0)
    return result


def sign_test(values) -> dict:
    """Exact two-sided sign test that the median difference is zero.

    Distribution-free and appropriate for 13 paired scene differences, where a
    normality assumption is untestable. Ties are dropped (the standard
    treatment), which is reported because dropping them changes n.
    """
    clean = [float(v) for v in values if v is not None and np.isfinite(v)]
    positive = sum(1 for v in clean if v > 0)
    negative = sum(1 for v in clean if v < 0)
    ties = len(clean) - positive - negative
    n = positive + negative
    if n == 0:
        return {"n": 0, "positive": positive, "negative": negative,
                "ties": ties, "p_two_sided": None}

    # P(X >= k) under Binomial(n, 0.5), doubled and clipped at 1.
    k = max(positive, negative)
    tail = sum(math.comb(n, i) for i in range(k, n + 1)) / (2.0 ** n)
    return {
        "n": n,
        "positive": positive,
        "negative": negative,
        "ties": ties,
        "p_two_sided": float(min(1.0, 2.0 * tail)),
        "note": "exact sign test on paired scene differences; ties dropped",
    }


@dataclass
class PairedResult:
    """One regime-vs-regime comparison on one metric."""

    metric: str
    treatment: str
    baseline: str
    higher_is_better: bool | None
    # Per (scene, seed) differences: the win-count unit.
    cell_deltas: dict[tuple[str, int], float] = field(default_factory=dict)
    # Per scene, seeds averaged: the inference unit.
    scene_deltas: dict[str, float] = field(default_factory=dict)
    cell_summary: dict = field(default_factory=dict)
    scene_summary: dict = field(default_factory=dict)
    ci: dict = field(default_factory=dict)
    signtest: dict = field(default_factory=dict)
    wins: int = 0
    losses: int = 0
    ties: int = 0
    n_scenes: int = 0
    n_cells: int = 0
    unpaired_dropped: list[str] = field(default_factory=list)

    def as_row(self) -> dict:
        """Flat dict for CSV/LaTeX emission."""
        return {
            "metric": self.metric,
            "treatment": self.treatment,
            "baseline": self.baseline,
            "n_scenes": self.n_scenes,
            "n_cells": self.n_cells,
            "mean_delta": self.scene_summary.get("mean"),
            "std_delta_scenes": self.scene_summary.get("std"),
            "min_delta_scene": self.scene_summary.get("min"),
            "max_delta_scene": self.scene_summary.get("max"),
            "ci_low": self.ci.get("ci_low"),
            "ci_high": self.ci.get("ci_high"),
            "ci_excludes_zero": self.ci.get("excludes_zero"),
            "sign_test_p": self.signtest.get("p_two_sided"),
            "wins": self.wins,
            "losses": self.losses,
            "ties": self.ties,
            "win_rate": (self.wins / self.n_cells) if self.n_cells else None,
            "higher_is_better": self.higher_is_better,
            "unpaired_dropped": ";".join(sorted(self.unpaired_dropped)),
        }


def paired_summary(treatment: str, baseline: str, metric: str,
                   *, treatment_values: dict[tuple[str, int], float],
                   baseline_values: dict[tuple[str, int], float],
                   alpha: float = 0.05, resamples: int = 10000,
                   seed: int = 20260818) -> PairedResult:
    """Build a :class:`PairedResult` from two ``{(scene, seed): value}`` maps.

    Only cells present in BOTH maps contribute; the rest are recorded in
    ``unpaired_dropped`` rather than dropped silently, because a missing cell on
    one side is exactly how an incomplete grid turns into a biased comparison.
    """
    direction = HIGHER_IS_BETTER.get(metric)
    result = PairedResult(metric=metric, treatment=treatment, baseline=baseline,
                          higher_is_better=direction)

    keys = set(treatment_values) | set(baseline_values)
    paired = sorted(set(treatment_values) & set(baseline_values))
    result.unpaired_dropped = [f"{s}@s{d}" for s, d in sorted(keys - set(paired))]

    for key in paired:
        result.cell_deltas[key] = (float(treatment_values[key])
                                   - float(baseline_values[key]))

    # Seeds averaged within a scene -> one independent observation per scene.
    per_scene: dict[str, list[float]] = {}
    for (scene, _seed), delta in result.cell_deltas.items():
        per_scene.setdefault(scene, []).append(delta)
    result.scene_deltas = {scene: float(np.mean(v))
                           for scene, v in sorted(per_scene.items())}

    result.n_cells = len(result.cell_deltas)
    result.n_scenes = len(result.scene_deltas)
    result.cell_summary = mean_std(result.cell_deltas.values())
    result.scene_summary = mean_std(result.scene_deltas.values())
    result.ci = bootstrap_ci(list(result.scene_deltas.values()), alpha=alpha,
                             resamples=resamples, seed=seed)
    result.signtest = sign_test(list(result.scene_deltas.values()))

    # Wins per (scene, seed). "Better" follows the metric's direction; a delta of
    # exactly zero is a tie, which for float metrics essentially never happens
    # and is reported rather than assigned to either side.
    if direction is None:
        result.wins = result.losses = result.ties = 0
    else:
        for delta in result.cell_deltas.values():
            improved = (delta > 0) if direction else (delta < 0)
            if delta == 0:
                result.ties += 1
            elif improved:
                result.wins += 1
            else:
                result.losses += 1
    return result


def time_to_target(iters, times, values, target: float, *,
                   higher_is_better: bool = True) -> dict:
    """Interpolated time at which a trajectory first reaches ``target``.

    Linear interpolation BETWEEN the bracketing checkpoints, and never beyond the
    measured range: a trajectory that has not reached the target by its last
    checkpoint returns ``not_reached``. Extrapolating here would invent the single
    most quotable number in the convergence study.

    Returns the FIRST crossing, not the last: "time to reach quality X" means the
    first time it got there, even if it later dipped below again (which it can,
    around a densification or an opacity reset).
    """
    points = [(float(i), float(t), float(v))
              for i, t, v in zip(iters, times, values)
              if v is not None and t is not None and np.isfinite(v)
              and np.isfinite(t)]
    points.sort(key=lambda p: p[0])
    if not points:
        return {"status": "no_data", "target": target}

    def reached(value: float) -> bool:
        return value >= target if higher_is_better else value <= target

    for index, (iteration, seconds, value) in enumerate(points):
        if not reached(value):
            continue
        if index == 0:
            # Already at target by the first checkpoint: report it, and say so,
            # rather than interpolating back towards zero from nothing.
            return {"status": "reached_at_first_checkpoint",
                    "target": target, "seconds": seconds,
                    "iteration": iteration, "interpolated": False}
        prev_iter, prev_seconds, prev_value = points[index - 1]
        span = value - prev_value
        # Guard a flat segment: with no change across the bracket there is no
        # crossing point to solve for, so take the checkpoint that reached it.
        if abs(span) < 1e-12:
            fraction = 1.0
        else:
            fraction = (target - prev_value) / span
        fraction = min(1.0, max(0.0, fraction))
        return {
            "status": "reached",
            "target": target,
            "seconds": prev_seconds + fraction * (seconds - prev_seconds),
            "iteration": prev_iter + fraction * (iteration - prev_iter),
            "interpolated": True,
            "bracket": [prev_iter, iteration],
        }

    best = max(v for _, _, v in points) if higher_is_better else min(
        v for _, _, v in points)
    return {
        "status": "not_reached",
        "target": target,
        "seconds": None,
        "iteration": None,
        "best_value": best,
        "last_iteration": points[-1][0],
        "shortfall": (target - best) if higher_is_better else (best - target),
        "note": "target not reached within the measured range; not extrapolated",
    }


def group_aggregate(rows: list[dict], metric: str, *,
                    expected_n: int | None = None) -> dict:
    """Aggregate one metric over a group of cells, refusing silent gaps.

    ``expected_n`` is the number of cells the group SHOULD contain. When the
    actual count differs the result is still returned — hiding it would be worse —
    but ``complete`` is False and ``missing`` says how many are absent, so tables
    can mark the cell rather than print a mean that looks like the others.
    """
    values = [r.get(metric) for r in rows]
    summary = mean_std(values)
    summary["expected_n"] = expected_n
    summary["complete"] = (expected_n is None or summary["n"] == expected_n)
    summary["missing"] = (None if expected_n is None
                          else max(0, expected_n - summary["n"]))
    summary["metric"] = metric
    return summary
