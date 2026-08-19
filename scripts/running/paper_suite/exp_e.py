"""Experiment E — what the three masks actually are, measured without training.

This characterizes the sampling patterns themselves at rho = 0.25, so the training
results have something mechanistic to point at. No GPU, no training: it builds the
same rank volumes the trainer uploads, walks them the same way the trainer walks
them, and measures the resulting 64-visit sequences.

Reproducing the trainer's sequence exactly is the whole point, so each regime is
generated from the same source as the run:

    iid_white   white rank volume (64 slices of per-tile permutations), slice
                index drawn independently per step from ``--stbn-seed``'s RNG
    indep_blue  the NVIDIA STBN rank volume, SAME independent per-step draw
    stbn        the same NVIDIA volume, slice index = (visit + phase) mod 64 —
                one camera's own visit counter, which is what the trainer does

That last one has a consequence worth stating plainly: over 64 consecutive visits
the STBN walk is a cyclic permutation and touches all 64 slices exactly once,
while 64 independent draws from 64 slices touch only ~40 distinct ones and repeat
the rest. Part of any STBN advantage is therefore simple coverage bookkeeping
rather than spectral magic, and separating those is what the visit-count and
coverage statistics below are for.

Two comparisons keep the numbers interpretable. Analytic references for a truly
independent mask of density rho are computed alongside every temporal statistic
(successive overlap rho^2, coverage 1-(1-rho)^n, mean revisit gap 1/rho), so a
measured value can be read as "better than independent" rather than just as a
number. And every regime is measured with identical code paths, so a difference
cannot come from the measurement.

The analysis is done on the 128x128 mask volume itself, which is what tiles the
screen: the trainer indexes it modulo its dimensions, so screen-space statistics
are this volume's statistics repeated. Stated rather than silently assumed.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from . import config as cfg
from . import figures as figs
from . import tables

# Which regimes this experiment characterizes, and how to build each sequence.
SAMPLING_REGIMES: tuple[str, ...] = ("iid_white", "indep_blue", "stbn")

MODE_COLUMNS = (
    "select_mode", "regime", "pixel_fraction", "measured_density",
    "tile_count_mean", "tile_count_std", "tile_count_min", "tile_count_max",
    "low_freq_power_mean", "successive_overlap_mean", "visit_count_cov",
    "revisit_gap_max",
)

SUMMARY_COLUMNS = (
    "regime", "select_mode", "pixel_fraction", "pixels_per_tile", "tile_area",
    "visits", "tile_count_mean", "tile_count_std", "tile_count_min",
    "tile_count_max",
    "measured_density", "successive_overlap_mean", "successive_overlap_expected_iid",
    "overlap_ratio_vs_iid", "temporal_adjacency_mean",
    "temporal_adjacency_expected_iid", "distinct_slices_used", "slices_available",
    "visit_count_mean", "visit_count_std", "visit_count_cov",
    "visit_count_min", "visit_count_max", "never_visited_fraction",
    "revisit_gap_mean", "revisit_gap_max", "revisit_gap_expected_iid",
    "low_freq_power_mean", "radial_lowband_power_mean",
)

COVERAGE_COLUMNS = (
    "regime", "visits", "unique_coverage_measured", "unique_coverage_expected_iid",
    "coverage_ratio_vs_iid", "coverage_deficit_vs_perfect",
)

RADIAL_COLUMNS = ("regime", "radial_frequency_normalized", "power_mean")
TEMPORAL_COLUMNS = ("regime", "lag", "autocorrelation")
TEMPORAL_SPECTRUM_COLUMNS = ("regime", "temporal_frequency_normalized", "power")
VISITS_COLUMNS = ("regime", "visit_count", "pixel_count", "pixel_fraction")


# ---------------------------------------------------------------------------
# Sequence construction — mirrors the trainer
# ---------------------------------------------------------------------------

def _rank_volume(source: str, seed: int):
    from trainer.backend.render import stbn as stbn_module

    spec = stbn_module.StbnMaskSpec(source=source, seed=seed)
    return stbn_module.build_rank_volume(spec), spec


def build_sequence(regime: str, *, fraction: float, visits: int,
                   stbn_seed: int = 0, camera_index: int = 0,
                   select_mode: str | None = None):
    """``(visits, H, W)`` boolean masks for one camera's successive visits.

    ``camera_index`` only sets the STBN phase. The volume is toroidal in t, so a
    different phase is a cyclic shift of the same sequence and every statistic
    below is shift-invariant — the choice does not affect any reported number,
    and using camera 0 keeps it reproducible.

    ``select_mode`` chooses between the global slice threshold and the per-tile
    ranking. Built through the SAME ``selected_volume`` the GPU table is packed
    from, so these diagnostics describe the mask training actually used rather
    than a re-implementation that could drift from it.
    """
    import dataclasses

    from trainer.backend.render import stbn as stbn_module

    source = (stbn_module.SOURCE_WHITE if regime == "iid_white"
              else stbn_module.SOURCE_NVIDIA)
    spec = dataclasses.replace(
        stbn_module.StbnMaskSpec(source=source, seed=stbn_seed),
        fraction=float(fraction),
        select_mode=(select_mode or cfg.PRIMARY_SELECT_MODE))
    budget = stbn_module.pixels_per_tile(fraction, spec.tile_area)
    selected = stbn_module.selected_volume(spec, budget)     # (slices, H, W)
    slices = selected.shape[0]

    if regime == "stbn":
        # The trainer's per-camera walk: phase from an integer mix of the camera
        # index, then +1 per visit (app.TrainingLoop._stbn_variables).
        phase = (int(camera_index) * 2654435761
                 + int(stbn_seed) * 40503) % slices
        indices = [(phase + visit) % slices for visit in range(visits)]
    else:
        # --stbn-independent: a fresh random slice each step, from an RNG seeded
        # by --stbn-seed and kept separate from the camera draw.
        rng = np.random.default_rng(int(stbn_seed))
        indices = [int(rng.integers(slices)) for _ in range(visits)]

    sequence = np.stack([selected[i] for i in indices], axis=0)
    return sequence, {"slice_indices": indices, "slices_available": slices,
                      "pixels_per_tile": budget, "tile_area": spec.tile_area,
                      "height": spec.height, "width": spec.width,
                      "select_mode": spec.select_mode}


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def spatial_power(sequence: np.ndarray) -> np.ndarray:
    """Mean 2D power spectrum over slices, DC removed, fftshifted.

    Each slice is centered by its OWN mean before the transform so the DC term
    (which is just the density, identical across regimes by construction) cannot
    dominate the picture and hide the arrangement differences that matter.
    """
    total = None
    for frame in sequence:
        centered = frame.astype(np.float64) - frame.mean()
        power = np.abs(np.fft.fft2(centered)) ** 2
        total = power if total is None else total + power
    mean = total / sequence.shape[0]
    return np.fft.fftshift(mean)


def radial_profile(power_shifted: np.ndarray, bins: int = 64):
    """Radially averaged spectrum against frequency normalized to Nyquist."""
    height, width = power_shifted.shape
    cy, cx = height // 2, width // 2
    yy, xx = np.mgrid[0:height, 0:width]
    # Normalize each axis by its own Nyquist so a non-square volume still maps
    # onto [0, ~1.41] consistently.
    radius = np.sqrt(((yy - cy) / (height / 2.0)) ** 2
                     + ((xx - cx) / (width / 2.0)) ** 2)
    edges = np.linspace(0.0, radius.max(), bins + 1)
    centers, values = [], []
    flat_r, flat_p = radius.ravel(), power_shifted.ravel()
    for lo, hi in zip(edges[:-1], edges[1:]):
        picked = (flat_r >= lo) & (flat_r < hi)
        if picked.any():
            centers.append(0.5 * (lo + hi))
            values.append(float(flat_p[picked].mean()))
    return np.asarray(centers), np.asarray(values)


def temporal_autocorrelation(sequence: np.ndarray, max_lag: int | None = None):
    """Per-pixel autocorrelation of the 0/1 selection signal, averaged.

    Normalized by the lag-0 variance, so lag 0 is exactly 1 and a value near 0 at
    lag L means "selection at distance L is uninformative" — the independent
    baseline. Negative values mean anti-correlation: the mask actively avoids
    re-picking a pixel at that spacing, which is the property a spatiotemporal
    mask is built to have.
    """
    visits = sequence.shape[0]
    max_lag = max_lag or (visits // 2)
    signal = sequence.astype(np.float64)
    signal -= signal.mean(axis=0, keepdims=True)
    variance = (signal ** 2).mean(axis=0)
    valid = variance > 1e-12
    out = []
    for lag in range(max_lag + 1):
        if lag == 0:
            out.append(1.0)
            continue
        product = (signal[:visits - lag] * signal[lag:]).mean(axis=0)
        ratio = np.zeros_like(product)
        ratio[valid] = product[valid] / variance[valid]
        out.append(float(ratio[valid].mean()) if valid.any() else 0.0)
    return np.arange(max_lag + 1), np.asarray(out)


def temporal_spectrum(sequence: np.ndarray):
    """Mean per-pixel temporal power spectrum (DC removed), one-sided."""
    signal = sequence.astype(np.float64)
    signal -= signal.mean(axis=0, keepdims=True)
    spectrum = np.abs(np.fft.rfft(signal, axis=0)) ** 2
    power = spectrum.mean(axis=(1, 2))
    freqs = np.fft.rfftfreq(sequence.shape[0])
    return freqs / 0.5, power          # normalized to Nyquist


def revisit_gaps(sequence: np.ndarray) -> dict:
    """Mean and maximum spacing between consecutive selections of a pixel.

    Computed CYCLICALLY (the volume wraps in t, and the trainer cycles it), and
    only over pixels selected at least twice — a gap is undefined for a pixel seen
    once or never, and treating those as gap 0 or gap 64 would both be wrong. How
    many pixels were excluded is reported.
    """
    visits, height, width = sequence.shape
    flat = sequence.reshape(visits, -1)
    gaps_mean, gaps_max = [], []
    excluded = 0
    for pixel in range(flat.shape[1]):
        times = np.flatnonzero(flat[:, pixel])
        if times.size < 2:
            excluded += 1
            continue
        # Wrap the last selection round to the first.
        deltas = np.diff(times)
        wrap = (times[0] + visits) - times[-1]
        allgaps = np.concatenate([deltas, [wrap]])
        gaps_mean.append(float(allgaps.mean()))
        gaps_max.append(float(allgaps.max()))
    return {
        "revisit_gap_mean": (float(np.mean(gaps_mean)) if gaps_mean else None),
        "revisit_gap_max": (float(np.max(gaps_max)) if gaps_max else None),
        "pixels_excluded_fewer_than_two_visits": excluded,
        "pixels_total": int(flat.shape[1]),
    }


def analyze(sequence: np.ndarray, meta: dict, regime: str,
            fraction: float) -> dict:
    """Every scalar statistic for one regime's sequence."""
    visits = sequence.shape[0]
    density = float(sequence.mean())

    successive = []
    for index in range(visits - 1):
        first, second = sequence[index], sequence[index + 1]
        count = int(first.sum())
        if count:
            successive.append(float((first & second).sum()) / count)
    overlap_mean = float(np.mean(successive)) if successive else None

    counts = sequence.sum(axis=0).astype(np.float64)
    mean_count, std_count = float(counts.mean()), float(counts.std())

    # Analytic independent-sampling references at the same density.
    expected_overlap = density
    expected_adjacency = (visits - 1) * density * density
    adjacency = float((sequence[:-1] & sequence[1:]).sum(axis=0).mean())

    power = spatial_power(sequence)
    radii, radial = radial_profile(power)
    lowband = radii <= 0.25
    total_power = float(radial.sum()) if radial.size else 0.0

    from trainer.backend.render import stbn as stbn_module
    # Fraction of spectral power below a quarter of Nyquist: blue noise pushes
    # energy AWAY from low frequencies, so lower is bluer.
    low_freq = float(np.mean([stbn_module.low_frequency_power(frame)
                             for frame in sequence]))

    # Per-8x8-tile selected counts: THE statistic that separates the two
    # selection modes. tile_rank pins every tile to exactly k; a global threshold
    # constrains only the slice and lets local counts breathe, which is what keeps
    # the point set the one the asset was optimized for.
    tiles = sequence.reshape(visits, meta["height"] // 8, 8,
                             meta["width"] // 8, 8)
    tile_counts_flat = tiles.transpose(0, 1, 3, 2, 4).reshape(
        visits, -1, 64).sum(axis=-1).astype(np.float64)

    gaps = revisit_gaps(sequence)
    return {
        "regime": regime,
        "select_mode": meta.get("select_mode"),
        "pixel_fraction": fraction,
        "pixels_per_tile": meta["pixels_per_tile"],
        "tile_count_mean": float(tile_counts_flat.mean()),
        "tile_count_std": float(tile_counts_flat.std()),
        "tile_count_min": float(tile_counts_flat.min()),
        "tile_count_max": float(tile_counts_flat.max()),
        "tile_area": meta["tile_area"],
        "visits": visits,
        "measured_density": density,
        "successive_overlap_mean": overlap_mean,
        # For independent masks, P(selected at t+1 | selected at t) = rho.
        "successive_overlap_expected_iid": expected_overlap,
        "overlap_ratio_vs_iid": ((overlap_mean / expected_overlap)
                                 if overlap_mean is not None and expected_overlap
                                 else None),
        "temporal_adjacency_mean": adjacency,
        "temporal_adjacency_expected_iid": expected_adjacency,
        "distinct_slices_used": len(set(meta["slice_indices"])),
        "slices_available": meta["slices_available"],
        "visit_count_mean": mean_count,
        "visit_count_std": std_count,
        # The headline uniformity number: 0 would be a perfectly even schedule.
        "visit_count_cov": (std_count / mean_count) if mean_count else None,
        "visit_count_min": float(counts.min()),
        "visit_count_max": float(counts.max()),
        "never_visited_fraction": float((counts == 0).mean()),
        "revisit_gap_mean": gaps["revisit_gap_mean"],
        "revisit_gap_max": gaps["revisit_gap_max"],
        # Mean gap for independent sampling is 1/rho.
        "revisit_gap_expected_iid": (1.0 / density) if density else None,
        "low_freq_power_mean": low_freq,
        "radial_lowband_power_mean": (float(radial[lowband].sum() / total_power)
                                      if total_power > 0 else None),
        "_power": power,
        "_radii": radii,
        "_radial": radial,
        "_counts": counts,
    }


def coverage_curve(sequence: np.ndarray, points) -> list[dict]:
    """Fraction of pixels seen at least once after n visits."""
    visits = sequence.shape[0]
    density = float(sequence.mean())
    rows: list[dict] = []
    for n in points:
        if n > visits:
            continue
        seen = sequence[:n].any(axis=0)
        measured = float(seen.mean())
        expected = 1.0 - (1.0 - density) ** n
        rows.append({
            "visits": int(n),
            "unique_coverage_measured": measured,
            "unique_coverage_expected_iid": expected,
            "coverage_ratio_vs_iid": (measured / expected) if expected else None,
            # How far short of the best possible (min(1, n*rho)) it falls.
            "coverage_deficit_vs_perfect": min(1.0, n * density) - measured,
        })
    return rows


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def _save_mask_images(analyses: dict, sequences: dict, out_dir: Path) -> None:
    plt = figs.pyplot()
    if plt is None:
        return
    n_show = 4
    fig, axes = plt.subplots(len(SAMPLING_REGIMES), n_show + 1,
                            figsize=(figs.FIG_WIDTH_DOUBLE,
                                     1.5 * len(SAMPLING_REGIMES)),
                            squeeze=False)
    for row, regime in enumerate(SAMPLING_REGIMES):
        sequence = sequences[regime]
        for column in range(n_show):
            ax = axes[row][column]
            # A 48x48 crop: at 128x128 the individual pixels of a 25% mask are
            # invisible in a figure this size, and the arrangement is the point.
            ax.imshow(sequence[column][:48, :48], cmap="gray_r",
                      interpolation="nearest")
            ax.set_xticks([]); ax.set_yticks([])
            if row == 0:
                ax.set_title(f"visit {column + 1}", fontsize=7)
            if column == 0:
                ax.set_ylabel(figs.style_for(regime)["label"], fontsize=7)
        ax = axes[row][n_show]
        counts = analyses[regime]["_counts"]
        image = ax.imshow(counts[:48, :48], cmap="viridis",
                          interpolation="nearest")
        ax.set_xticks([]); ax.set_yticks([])
        if row == 0:
            ax.set_title(f"visits after {cfg.SAMPLING_VISITS}", fontsize=7)
        plt.colorbar(image, ax=ax, fraction=0.046)
    fig.suptitle("Experiment E: mask realizations (48$\\times$48 crop) and "
                 "per-pixel visit counts", fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    figs.save(fig, out_dir, "expE_masks_and_visits")


def _save_spectra(analyses: dict, out_dir: Path) -> None:
    plt = figs.pyplot()
    if plt is None:
        return
    fig, axes = plt.subplots(1, len(SAMPLING_REGIMES) + 1,
                            figsize=(figs.FIG_WIDTH_DOUBLE, 2.1),
                            squeeze=False)
    # Shared log scale so the three 2D spectra are directly comparable.
    stack = [np.log10(analyses[r]["_power"] + 1e-12) for r in SAMPLING_REGIMES]
    vmin = min(float(s.min()) for s in stack)
    vmax = max(float(s.max()) for s in stack)
    for column, regime in enumerate(SAMPLING_REGIMES):
        ax = axes[0][column]
        ax.imshow(stack[column], cmap="magma", vmin=vmin, vmax=vmax)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(figs.style_for(regime)["label"], fontsize=7.5)
    ax = axes[0][len(SAMPLING_REGIMES)]
    for regime in SAMPLING_REGIMES:
        style = figs.style_for(regime)
        radii = analyses[regime]["_radii"]
        radial = analyses[regime]["_radial"]
        ax.plot(radii, radial / max(radial.max(), 1e-12), color=style["color"],
                linestyle=style["linestyle"], linewidth=1.1,
                label=style["label"])
    ax.set_xlabel("radial frequency (1 = Nyquist)")
    ax.set_ylabel("normalized power")
    ax.set_yscale("log")
    ax.legend(fontsize=6)
    fig.suptitle("Mean 2D spatial power spectrum (log, DC centred and removed) "
                 "and its radial average", fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    figs.save(fig, out_dir, "expE_spatial_spectra")


def _save_temporal(temporal: dict, spectra: dict, coverage: dict,
                   analyses: dict, out_dir: Path) -> None:
    plt = figs.pyplot()
    if plt is None:
        return
    fig, axes = plt.subplots(1, 4, figsize=(figs.FIG_WIDTH_DOUBLE, 2.1))

    ax = axes[0]
    for regime in SAMPLING_REGIMES:
        style = figs.style_for(regime)
        lags, values = temporal[regime]
        ax.plot(lags, values, color=style["color"], linestyle=style["linestyle"],
                linewidth=1.1, label=style["label"])
    ax.axhline(0.0, color="#000000", linewidth=0.6, linestyle=":")
    ax.set_xlabel("temporal lag (visits)")
    ax.set_ylabel("autocorrelation")
    ax.legend(fontsize=6)

    ax = axes[1]
    for regime in SAMPLING_REGIMES:
        style = figs.style_for(regime)
        freqs, power = spectra[regime]
        ax.plot(freqs, power / max(power.max(), 1e-12), color=style["color"],
                linestyle=style["linestyle"], linewidth=1.1)
    ax.set_xlabel("temporal frequency (1 = Nyquist)")
    ax.set_ylabel("normalized power")

    ax = axes[2]
    for regime in SAMPLING_REGIMES:
        style = figs.style_for(regime)
        rows = coverage[regime]
        ax.plot([r["visits"] for r in rows],
                [r["unique_coverage_measured"] for r in rows],
                color=style["color"], marker=style["marker"], markersize=3,
                linestyle=style["linestyle"], linewidth=1.1)
    reference = coverage[SAMPLING_REGIMES[0]]
    ax.plot([r["visits"] for r in reference],
            [r["unique_coverage_expected_iid"] for r in reference],
            color="#999999", linestyle=":", linewidth=1.0,
            label="independent (analytic)")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("visits")
    ax.set_ylabel("unique pixels covered")
    ax.legend(fontsize=6)

    ax = axes[3]
    for regime in SAMPLING_REGIMES:
        style = figs.style_for(regime)
        counts = analyses[regime]["_counts"].ravel()
        bins = np.arange(counts.min() - 0.5, counts.max() + 1.5, 1.0)
        ax.hist(counts, bins=bins, histtype="step", density=True,
                color=style["color"], linewidth=1.1)
    ax.set_xlabel(f"visits per pixel after {cfg.SAMPLING_VISITS}")
    ax.set_ylabel("density")

    fig.suptitle("Temporal structure, cumulative coverage, and visit-count "
                 "uniformity", fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    figs.save(fig, out_dir, "expE_temporal_and_coverage")


def run(out: dict, *, fraction: float = cfg.PRIMARY_FRACTION,
        visits: int = cfg.SAMPLING_VISITS, stbn_seed: int = 0) -> dict:
    analysis_dir: Path = out["sampling_analysis"]
    analysis_dir.mkdir(parents=True, exist_ok=True)

    sequences: dict[str, np.ndarray] = {}
    analyses: dict[str, dict] = {}
    temporal: dict[str, tuple] = {}
    spectra: dict[str, tuple] = {}
    coverage: dict[str, list[dict]] = {}
    metas: dict[str, dict] = {}

    mode_rows: list[dict] = []
    try:
        for regime in SAMPLING_REGIMES:
            sequence, meta = build_sequence(regime, fraction=fraction,
                                            visits=visits, stbn_seed=stbn_seed)
            sequences[regime] = sequence
            metas[regime] = meta
            analyses[regime] = analyze(sequence, meta, regime, fraction)
            temporal[regime] = temporal_autocorrelation(sequence)
            spectra[regime] = temporal_spectrum(sequence)
            coverage[regime] = coverage_curve(sequence,
                                              cfg.SAMPLING_COVERAGE_POINTS)

        # Both selection modes, for the stratification ablation. Cheap (no
        # training), and it is the measurement that says whether forcing exactly
        # k per execution tile damages the mask — the claim Experiment S then
        # tests against trained quality.
        for mode in (cfg.SELECT_GLOBAL, cfg.SELECT_TILE_RANK):
            for regime in SAMPLING_REGIMES:
                seq, mode_meta = build_sequence(
                    regime, fraction=fraction, visits=visits,
                    stbn_seed=stbn_seed, select_mode=mode)
                record = analyze(seq, mode_meta, regime, fraction)
                mode_rows.append({
                    k: record.get(k) for k in MODE_COLUMNS})
    except FileNotFoundError as error:
        # The NVIDIA volume is a non-vendored, opt-in download.
        return {"status": "skipped", "reason": str(error)}

    summary_rows = [{k: v for k, v in analyses[r].items()
                     if not k.startswith("_")} for r in SAMPLING_REGIMES]
    tables.write_csv(analysis_dir / "sampling_summary.csv", summary_rows,
                     SUMMARY_COLUMNS)
    tables.write_csv(analysis_dir / "sampling_select_mode_comparison.csv",
                     mode_rows, MODE_COLUMNS)

    coverage_rows = [{"regime": r, **row}
                     for r in SAMPLING_REGIMES for row in coverage[r]]
    tables.write_csv(analysis_dir / "sampling_coverage.csv", coverage_rows,
                     COVERAGE_COLUMNS)

    radial_rows = [{"regime": r,
                    "radial_frequency_normalized": float(f),
                    "power_mean": float(p)}
                   for r in SAMPLING_REGIMES
                   for f, p in zip(analyses[r]["_radii"], analyses[r]["_radial"])]
    tables.write_csv(analysis_dir / "sampling_radial_spectrum.csv", radial_rows,
                     RADIAL_COLUMNS)

    temporal_rows = [{"regime": r, "lag": int(l), "autocorrelation": float(v)}
                     for r in SAMPLING_REGIMES
                     for l, v in zip(*temporal[r])]
    tables.write_csv(analysis_dir / "sampling_temporal_autocorrelation.csv",
                     temporal_rows, TEMPORAL_COLUMNS)

    spectrum_rows = [{"regime": r,
                      "temporal_frequency_normalized": float(f),
                      "power": float(p)}
                     for r in SAMPLING_REGIMES
                     for f, p in zip(*spectra[r])]
    tables.write_csv(analysis_dir / "sampling_temporal_spectrum.csv",
                     spectrum_rows, TEMPORAL_SPECTRUM_COLUMNS)

    visit_rows: list[dict] = []
    for regime in SAMPLING_REGIMES:
        counts = analyses[regime]["_counts"].ravel()
        values, occurrences = np.unique(counts.astype(int), return_counts=True)
        for value, occurrence in zip(values, occurrences):
            visit_rows.append({
                "regime": regime, "visit_count": int(value),
                "pixel_count": int(occurrence),
                "pixel_fraction": float(occurrence / counts.size)})
    tables.write_csv(analysis_dir / "sampling_visit_counts.csv", visit_rows,
                     VISITS_COLUMNS)

    # The raw sequences, so any statistic can be recomputed without rebuilding
    # the rank volumes (and so a reviewer can check these numbers).
    np.savez_compressed(
        analysis_dir / "sampling_sequences.npz",
        **{regime: sequences[regime] for regime in SAMPLING_REGIMES},
        slice_indices=json.dumps({r: metas[r]["slice_indices"]
                                  for r in SAMPLING_REGIMES}))

    _save_mask_images(analyses, sequences, analysis_dir)
    _save_spectra(analyses, analysis_dir)
    _save_temporal(temporal, spectra, coverage, analyses, analysis_dir)

    findings = {r: {k: v for k, v in analyses[r].items()
                    if not k.startswith("_")} for r in SAMPLING_REGIMES}

    def _mode(mode: str, regime: str, key: str):
        row = next((r for r in mode_rows if r["select_mode"] == mode
                    and r["regime"] == regime), None)
        return (row or {}).get(key)

    # The spectral cost of tile stratification, and — the part that was easy to
    # miss — the same constraint making the WHITE control artificially bluer, so
    # the blue-vs-white contrast is compressed from both ends.
    blue_global = _mode(cfg.SELECT_GLOBAL, "stbn", "low_freq_power_mean")
    blue_tiled = _mode(cfg.SELECT_TILE_RANK, "stbn", "low_freq_power_mean")
    white_global = _mode(cfg.SELECT_GLOBAL, "iid_white", "low_freq_power_mean")
    white_tiled = _mode(cfg.SELECT_TILE_RANK, "iid_white", "low_freq_power_mean")
    stratification = {
        "blue_low_freq_global": blue_global,
        "blue_low_freq_tile_rank": blue_tiled,
        "white_low_freq_global": white_global,
        "white_low_freq_tile_rank": white_tiled,
        "blue_degradation_factor": (
            (blue_tiled / blue_global) if blue_global else None),
        "white_flattery_factor": (
            (white_global / white_tiled) if white_tiled else None),
        "contrast_global": ((white_global / blue_global)
                            if blue_global else None),
        "contrast_tile_rank": ((white_tiled / blue_tiled)
                               if blue_tiled else None),
        "tile_count_std_global": _mode(cfg.SELECT_GLOBAL, "stbn",
                                       "tile_count_std"),
        "tile_count_std_tile_rank": _mode(cfg.SELECT_TILE_RANK, "stbn",
                                          "tile_count_std"),
    }
    return {
        "status": "ok",
        "fraction": fraction,
        "visits": visits,
        "select_mode": cfg.PRIMARY_SELECT_MODE,
        "per_regime": findings,
        "select_mode_comparison": mode_rows,
        "stratification": stratification,
        "coverage": {r: coverage[r] for r in SAMPLING_REGIMES},
        "note": (
            "Statistics are of the 128x128x64 mask volume itself, which tiles the "
            "screen (the kernels index it modulo its dimensions). Over 64 "
            "consecutive visits the STBN walk is a cyclic permutation and uses "
            "each of the 64 slices exactly once, whereas 64 independent draws "
            "use ~40 distinct slices; part of any coverage advantage is that "
            "bookkeeping rather than the spectrum, which is why both are "
            "reported."),
        "artifacts": {
            "summary_csv": "sampling_analysis/sampling_summary.csv",
            "coverage_csv": "sampling_analysis/sampling_coverage.csv",
            "radial_csv": "sampling_analysis/sampling_radial_spectrum.csv",
            "temporal_autocorr_csv":
                "sampling_analysis/sampling_temporal_autocorrelation.csv",
            "temporal_spectrum_csv":
                "sampling_analysis/sampling_temporal_spectrum.csv",
            "visit_counts_csv": "sampling_analysis/sampling_visit_counts.csv",
            "select_mode_csv":
                "sampling_analysis/sampling_select_mode_comparison.csv",
            "sequences_npz": "sampling_analysis/sampling_sequences.npz",
            "figures": ["sampling_analysis/expE_masks_and_visits.pdf",
                        "sampling_analysis/expE_spatial_spectra.pdf",
                        "sampling_analysis/expE_temporal_and_coverage.pdf"],
        },
    }
