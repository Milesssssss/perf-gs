# Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

"""Sections 9 and 10: is the sampler CALIBRATED, and does it stay BLUE?

No GPU, no training. Everything here runs against the host reference in
:mod:`trainer.backend.render.adaptive_stbn` — which the GPU tests pin the device
path to — so the two questions can be answered before a single training hour is
spent.

**Calibration (section 9).** For a fixed probability field p, generate many masks
and measure the empirical inclusion frequency phat_i. It must match p_i. This tests
the FIRST-ORDER marginal in isolation from any blue-noise property, which is why it
is run on four synthetic fields (uniform, a horizontal ramp, one Gaussian hotspot,
two hotspots) rather than on a scene: a synthetic field is known exactly, so a
discrepancy can only come from the sampler.

One result is structural and worth stating before it is measured. With a FIXED
64-slice blue-noise volume, M_{i,t} = 1[U_{i,t} < p_i] is DETERMINISTIC given the
slice — there is no randomness left to average over. So the blue arms' phat_i can
only ever equal the fraction of the volume's 64 per-pixel values below p_i, which
lands on a grid of spacing 1/64 around p_i. That is a quantization floor of ~1/128
on the calibration error, not a bias, and it is reported as such: the IID arm (a
genuine Bernoulli draw each visit) is measured alongside so the two error sources
can be told apart.

**Variable-density blue noise (section 10).** A nonuniform density NECESSARILY puts
low-frequency structure into the raw mask — that is the mean field p itself, not a
defect — so the spectrum of M would report every adaptive method as "less blue"
purely for having a non-flat marginal. The diagnostics therefore run on the
variance-normalized residual

    Z_i = (M_i - p_i) / sqrt(p_i (1 - p_i) + eps)

whose expectation is zero and whose variance is one at every pixel regardless of
the density. What remains in Z is the ARRANGEMENT, which is the only thing the
three uniform sources differ in.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from paper_suite import figures as figs
from paper_suite import tables
from paper_suite.exp_e import radial_profile, temporal_autocorrelation

from trainer.backend.render import adaptive_stbn as ad

from . import config as cfg

# The four static probability fields of section 9. Each is defined as an
# IMPORTANCE field and pushed through the real construction (smooth -> strength ->
# lambda solve), so what is calibrated is the shipped pipeline and not a hand-made
# p that happens to be easy.
FIELDS = ("uniform", "horizontal_ramp", "gaussian_hotspot", "two_hotspots")

SAMPLERS = ("adaptive_iid", "adaptive_indep_blue", "adaptive_stbn")
SAMPLER_LABELS = {
    "adaptive_iid": "Adaptive IID",
    "adaptive_indep_blue": "Adaptive indep. blue",
    "adaptive_stbn": "Adaptive STBN",
}
SAMPLER_STYLE = {
    "adaptive_iid": {"color": "#E69F00", "marker": "s", "linestyle": "--"},
    "adaptive_indep_blue": {"color": "#56B4E9", "marker": "^", "linestyle": "-."},
    "adaptive_stbn": {"color": "#009E73", "marker": "D", "linestyle": "-"},
}

CALIBRATION_COLUMNS = (
    "field", "sampler", "alpha", "rho", "visits", "n_pixels",
    "p_mean", "p_min", "p_max", "saturated_fraction",
    "phat_mean", "calibration_rmse", "calibration_max_abs",
    "calibration_bias", "binomial_rmse_floor", "rmse_over_floor",
)
RESIDUAL_COLUMNS = (
    "field", "sampler", "alpha", "rho", "visits",
    "residual_mean", "residual_std",
    "low_freq_power", "low_freq_power_white_reference",
    "low_freq_ratio_vs_white", "temporal_lag1_autocorrelation",
    "temporal_lag1_expected_independent",
)
RADIAL_COLUMNS = ("field", "sampler", "radial_frequency_normalized", "power_mean")
TEMPORAL_COLUMNS = ("field", "sampler", "lag", "autocorrelation")

# Radial band the "is it blue?" scalar is read from: below a quarter of Nyquist.
LOW_FREQ_CUTOFF = 0.25

# ONE probability colormap AND one absolute range across every figure and
# visualization in the suite (design section 18), so a given colour means the same
# p everywhere — in a synthetic-field panel, in a per-camera panel, and across
# cameras whose p happens to span different intervals. Auto-scaling each panel to
# its own range would make two panels look alike while showing different
# allocations, which is precisely what a shared colormap is for.
P_COLORMAP = "magma"
P_VMIN, P_VMAX = 0.0, 1.0


# ---------------------------------------------------------------------------
# Fields
# ---------------------------------------------------------------------------

def importance_field(name: str, size: int, seed: int = 0) -> np.ndarray:
    """A synthetic screen-gradient magnitude with the requested spatial shape.

    Deliberately noisy at the pixel level with smooth structure on top, matching
    what the real thing looks like: L1's gradient is sign-valued, so the raw
    per-pixel |h| is almost binary and all the usable signal is in its
    neighbourhood average.
    """
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:size, 0:size] / float(size)
    noise = np.abs(rng.standard_normal((size, size))) + 0.25
    if name == "uniform":
        return noise
    if name == "horizontal_ramp":
        return noise * (0.2 + 3.0 * xx)
    if name == "gaussian_hotspot":
        return noise * (0.2 + 8.0 * np.exp(
            -(((yy - 0.5) ** 2 + (xx - 0.5) ** 2) / 0.02)))
    if name == "two_hotspots":
        first = np.exp(-(((yy - 0.3) ** 2 + (xx - 0.25) ** 2) / 0.01))
        second = np.exp(-(((yy - 0.7) ** 2 + (xx - 0.75) ** 2) / 0.008))
        return noise * (0.2 + 6.0 * first + 4.0 * second)
    raise ValueError(f"unknown field {name!r}")


def _spec(sampler: str, *, alpha: float, fraction: float,
          seed: int) -> ad.AdaptiveSpec:
    uniform = {"adaptive_iid": ad.UNIFORM_IID,
               "adaptive_indep_blue": ad.UNIFORM_BLUE_INDEPENDENT,
               "adaptive_stbn": ad.UNIFORM_BLUE_TEMPORAL}[sampler]
    source = ad.SOURCE_WHITE if sampler == "adaptive_iid" else ad.SOURCE_NVIDIA
    return ad.AdaptiveSpec(fraction=fraction, alpha=alpha, sigma=cfg.PRIMARY_SIGMA,
                           downsample=cfg.PRIMARY_DOWNSAMPLE, uniform=uniform,
                           source=source, seed=seed)


# ---------------------------------------------------------------------------
# Mask sequences — mirroring the trainer
# ---------------------------------------------------------------------------

def mask_sequence(sampler: str, p: np.ndarray, spec: ad.AdaptiveSpec, *,
                  visits: int, seed: int) -> np.ndarray:
    """``(visits, H, W)`` boolean, generated the way the trainer generates it.

    ``adaptive_iid`` uses the SHADER's integer hash rather than numpy's RNG, so
    what is measured is the stream the trainer actually samples. The blue arms read
    the same normalized-rank volume the sampler uploads; they differ only in how
    the slice index advances — independently drawn per visit, or walking a camera's
    own visit counter, exactly as ``--stbn-independent`` distinguishes the uniform
    regimes.
    """
    height, width = p.shape
    rng = np.random.default_rng(seed)
    if sampler == "adaptive_iid":
        seeds = rng.integers(1 << 32, size=visits)
        return np.stack([ad.mask_from_uniform(
            ad.hash_uniform(height, width, int(s)), p) for s in seeds])

    volume = ad.uniform_volume(ad.spec_to_mask_spec(spec))
    slices = volume.shape[0]
    if sampler == "adaptive_indep_blue":
        indices = [int(rng.integers(slices)) for _ in range(visits)]
    else:
        phase = int(rng.integers(slices))
        indices = [(phase + step) % slices for step in range(visits)]
    return np.stack([ad.mask_from_uniform(
        ad.wrap_to_screen(volume[index], height, width), p) for index in indices])


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def calibration(masks: np.ndarray, p: np.ndarray) -> tuple[dict, np.ndarray]:
    """Empirical inclusion frequency against p, with the noise floor alongside.

    ``binomial_rmse_floor`` is what a PERFECTLY calibrated independent sampler
    would score with this many visits, ``sqrt(mean(p(1-p))/T)``. Reporting the raw
    RMSE without it would be uninterpretable — the number shrinks with T for a
    correct sampler and the reader cannot tell "calibrated" from "few draws".
    """
    visits = masks.shape[0]
    phat = masks.mean(axis=0)
    error = phat - p
    floor = float(np.sqrt(np.mean(p * (1.0 - p)) / max(visits, 1)))
    rmse = float(np.sqrt(np.mean(error ** 2)))
    return {
        "visits": int(visits),
        "n_pixels": int(p.size),
        "phat_mean": float(phat.mean()),
        "calibration_rmse": rmse,
        "calibration_max_abs": float(np.abs(error).max()),
        "calibration_bias": float(error.mean()),
        "binomial_rmse_floor": floor,
        "rmse_over_floor": (rmse / floor) if floor > 0 else None,
    }, phat


def residual_spectrum(masks: np.ndarray, p: np.ndarray) -> dict:
    """Radial spectrum and low-frequency power of the standardized residual Z.

    Averaged over visits (each visit's 2D spectrum, then the mean), so the number
    describes a typical single mask rather than an accumulation. The white
    reference is the same statistic on independent Bernoulli draws at the SAME p —
    computed rather than assumed, because the analytic value for a finite grid with
    DC removed is not exactly the band's area fraction.
    """
    visits = masks.shape[0]
    z = np.stack([ad.residual(masks[t], p) for t in range(visits)])
    power = np.zeros(p.shape, dtype=np.float64)
    for t in range(visits):
        centered = z[t] - z[t].mean()
        power += np.abs(np.fft.fft2(centered)) ** 2
    power /= visits
    shifted = np.fft.fftshift(power)
    centers, values = radial_profile(shifted, bins=64)

    fy = np.fft.fftfreq(p.shape[0])[:, None]
    fx = np.fft.fftfreq(p.shape[1])[None, :]
    radius = np.sqrt((fy / 0.5) ** 2 + (fx / 0.5) ** 2)
    total = power.sum()
    low = float(power[radius <= LOW_FREQ_CUTOFF].sum() / total) if total > 0 else 0.0
    lags, autocorrelation = temporal_autocorrelation(masks, max_lag=min(16,
                                                                       visits - 1))
    return {
        "residual_mean": float(z.mean()),
        "residual_std": float(z.std()),
        "low_freq_power": low,
        "radial_centers": centers,
        "radial_values": values,
        "temporal_lags": lags,
        "temporal_autocorrelation": autocorrelation,
    }


def _white_low_freq_reference(p: np.ndarray, visits: int, seed: int) -> float:
    rng = np.random.default_rng(seed)
    masks = np.stack([rng.random(p.shape) < p for _ in range(visits)])
    return residual_spectrum(masks, p)["low_freq_power"]


# ---------------------------------------------------------------------------
# The stage
# ---------------------------------------------------------------------------

def run(paths: dict, *, alpha: float = cfg.PRIMARY_ALPHA,
        fraction: float = cfg.PRIMARY_FRACTION,
        size: int = cfg.CALIBRATION_SIZE,
        calibration_visits: int = cfg.CALIBRATION_VISITS,
        residual_visits: int = cfg.RESIDUAL_VISITS,
        seed: int = 20260821) -> dict:
    """Run sections 9 and 10 and write every artifact. Returns a stage record."""
    out = paths["sampling_validation"]
    out.mkdir(parents=True, exist_ok=True)

    asset = ad.spec_to_mask_spec(_spec("adaptive_stbn", alpha=alpha,
                                       fraction=fraction, seed=0))
    if not asset.resolved_path().is_file():
        message = (f"NVIDIA STBN asset absent at {asset.resolved_path()}; run "
                   f"scripts/data/download_stbn.py. The blue arms cannot be "
                   f"measured without it.")
        print(f"[sampling] SKIP: {message}", flush=True)
        return {"status": "skipped", "reason": message}

    calibration_rows: list[dict] = []
    residual_rows: list[dict] = []
    radial_rows: list[dict] = []
    temporal_rows: list[dict] = []
    probability_maps: dict[str, np.ndarray] = {}
    spectra: dict[tuple[str, str], dict] = {}
    phats: dict[tuple[str, str], np.ndarray] = {}

    for field_name in FIELDS:
        importance = importance_field(field_name, size, seed=seed)
        spec = _spec("adaptive_stbn", alpha=alpha, fraction=fraction, seed=0)
        p, diagnostics = ad.probability_map(importance, spec)
        probability_maps[field_name] = p
        print(f"[sampling] {field_name}: p in "
              f"[{diagnostics['p_min']:.4f}, {diagnostics['p_max']:.4f}] "
              f"mean {diagnostics['p_mean']:.8f} (target "
              f"{spec.budget_fraction():.8f}) saturated "
              f"{diagnostics['saturated_fraction'] * 100:.2f}%", flush=True)

        white_reference = _white_low_freq_reference(p, residual_visits, seed + 7)

        for sampler in SAMPLERS:
            sampler_spec = _spec(sampler, alpha=alpha, fraction=fraction, seed=0)
            masks = mask_sequence(sampler, p, sampler_spec,
                                  visits=calibration_visits, seed=seed + 1)
            statistics, phat = calibration(masks, p)
            phats[(field_name, sampler)] = phat
            row = {"field": field_name, "sampler": sampler, "alpha": alpha,
                   "rho": spec.budget_fraction(),
                   "p_mean": diagnostics["p_mean"], "p_min": diagnostics["p_min"],
                   "p_max": diagnostics["p_max"],
                   "saturated_fraction": diagnostics["saturated_fraction"],
                   **statistics}
            calibration_rows.append(row)

            spectrum = residual_spectrum(masks[:residual_visits], p)
            spectra[(field_name, sampler)] = spectrum
            lag1 = (float(spectrum["temporal_autocorrelation"][1])
                    if len(spectrum["temporal_autocorrelation"]) > 1 else None)
            residual_rows.append({
                "field": field_name, "sampler": sampler, "alpha": alpha,
                "rho": spec.budget_fraction(), "visits": residual_visits,
                "residual_mean": spectrum["residual_mean"],
                "residual_std": spectrum["residual_std"],
                "low_freq_power": spectrum["low_freq_power"],
                "low_freq_power_white_reference": white_reference,
                "low_freq_ratio_vs_white": (spectrum["low_freq_power"]
                                            / white_reference
                                            if white_reference > 0 else None),
                "temporal_lag1_autocorrelation": lag1,
                # An independent mask has zero temporal autocorrelation at every
                # nonzero lag; a temporally distributed one goes NEGATIVE.
                "temporal_lag1_expected_independent": 0.0,
            })
            for centre, value in zip(spectrum["radial_centers"],
                                     spectrum["radial_values"]):
                radial_rows.append({"field": field_name, "sampler": sampler,
                                    "radial_frequency_normalized": float(centre),
                                    "power_mean": float(value)})
            for lag, value in zip(spectrum["temporal_lags"],
                                  spectrum["temporal_autocorrelation"]):
                temporal_rows.append({"field": field_name, "sampler": sampler,
                                      "lag": int(lag),
                                      "autocorrelation": float(value)})
            print(f"[sampling]   {sampler:22s} rmse {row['calibration_rmse']:.5f} "
                  f"({row['rmse_over_floor']:.2f}x the independent floor)  "
                  f"max |phat-p| {row['calibration_max_abs']:.4f}  "
                  f"low-freq {spectrum['low_freq_power']:.5f} "
                  f"({spectrum['low_freq_power'] / white_reference:.2f}x white)  "
                  f"lag-1 {lag1:+.4f}", flush=True)

    tables.write_csv(out / "calibration.csv", calibration_rows,
                     CALIBRATION_COLUMNS)
    tables.write_csv(out / "residual_structure.csv", residual_rows,
                     RESIDUAL_COLUMNS)
    tables.write_csv(out / "residual_radial_spectrum.csv", radial_rows,
                     RADIAL_COLUMNS)
    tables.write_csv(out / "residual_temporal_autocorrelation.csv",
                     temporal_rows, TEMPORAL_COLUMNS)
    np.savez_compressed(out / "probability_maps.npz", **probability_maps)

    written = _figures(paths, probability_maps, spectra, phats)

    record = {
        "status": "ok",
        "alpha": alpha,
        "rho_requested": fraction,
        "rho_enforced": _spec("adaptive_stbn", alpha=alpha, fraction=fraction,
                              seed=0).budget_fraction(),
        "size": size,
        "calibration_visits": calibration_visits,
        "residual_visits": residual_visits,
        "seed": seed,
        "fields": list(FIELDS),
        "samplers": list(SAMPLERS),
        "quantization_note": (
            "the blue arms threshold a FIXED 64-slice volume, so M is "
            "deterministic given the slice and phat can only land on a grid of "
            "spacing 1/64 around p — a quantization floor of ~1/128 on the "
            "calibration error, not a bias. The IID arm is a genuine Bernoulli "
            "draw per visit and is measured alongside so the two are separable."),
        "outputs": {
            "calibration_csv": "sampling_validation/calibration.csv",
            "residual_csv": "sampling_validation/residual_structure.csv",
            "radial_csv": "sampling_validation/residual_radial_spectrum.csv",
            "temporal_csv":
                "sampling_validation/residual_temporal_autocorrelation.csv",
            "probability_maps_npz": "sampling_validation/probability_maps.npz",
            "figures": [str(p.relative_to(paths["root"])) for p in written],
        },
        "headline": _headline(calibration_rows, residual_rows),
    }
    (out / "sampling_validation.json").write_text(json.dumps(record, indent=2))
    return record


def _headline(calibration_rows: list[dict], residual_rows: list[dict]) -> dict:
    """The two numbers the report leads with, per sampler, pooled over fields."""
    out: dict[str, dict] = {}
    for sampler in SAMPLERS:
        cal = [r for r in calibration_rows if r["sampler"] == sampler]
        res = [r for r in residual_rows if r["sampler"] == sampler]
        out[sampler] = {
            "calibration_rmse_max": max((r["calibration_rmse"] for r in cal),
                                        default=None),
            "calibration_max_abs": max((r["calibration_max_abs"] for r in cal),
                                       default=None),
            "calibration_bias_max_abs": max((abs(r["calibration_bias"])
                                             for r in cal), default=None),
            "rmse_over_floor_max": max((r["rmse_over_floor"] for r in cal
                                        if r["rmse_over_floor"] is not None),
                                       default=None),
            "low_freq_ratio_vs_white_mean": (
                float(np.mean([r["low_freq_ratio_vs_white"] for r in res]))
                if res else None),
            "temporal_lag1_mean": (
                float(np.mean([r["temporal_lag1_autocorrelation"] for r in res
                               if r["temporal_lag1_autocorrelation"] is not None]))
                if res else None),
        }
    return out


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def _figures(paths: dict, probability_maps: dict, spectra: dict,
             phats: dict) -> list[Path]:
    plt = figs.pyplot()
    if plt is None:
        return []
    out = paths["figures"]
    written: list[Path] = []

    # 1. The probability fields, on ONE shared colour scale so the four panels are
    #    comparable by eye rather than each auto-scaled to its own range — and the
    #    range is the ABSOLUTE [0,1], shared with the per-camera visualizations.
    fig, axes = plt.subplots(1, len(FIELDS),
                             figsize=(figs.FIG_WIDTH_DOUBLE, 2.0))
    image = None
    for ax, name in zip(np.atleast_1d(axes), FIELDS):
        image = ax.imshow(probability_maps[name], cmap=P_COLORMAP, vmin=P_VMIN,
                          vmax=P_VMAX, interpolation="nearest")
        ax.set_title(name.replace("_", " "), fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
    if image is not None:
        fig.colorbar(image, ax=list(np.atleast_1d(axes)), shrink=0.85,
                     label=r"$p_i$")
    written += figs.save(fig, out, "sampling_probability_fields")

    # 2. Calibration: the CONDITIONAL MEAN of phat given p, in equal-count bins of
    #    p, against the diagonal. Binned rather than scattered because 16k points
    #    are an unreadable cloud and the claim is about the conditional mean; the
    #    per-pixel spread is reported as the RMSE in calibration.csv. Error bars
    #    are the standard error WITHIN the bin, so a deviation can be judged
    #    significant or not without leaving the figure.
    fig, axes = plt.subplots(1, len(FIELDS),
                             figsize=(figs.FIG_WIDTH_DOUBLE, 2.2))
    for ax, name in zip(np.atleast_1d(axes), FIELDS):
        p = probability_maps[name]
        low, high = float(p.min()), float(p.max())
        ax.plot([low, high], [low, high], color="#000000", linewidth=0.8,
                zorder=1, label="ideal")
        for sampler in SAMPLERS:
            phat = phats.get((name, sampler))
            if phat is None:
                continue
            centers, means, errors = _binned_conditional_mean(p, phat)
            ax.errorbar(centers, means, yerr=errors, capsize=1.5, linewidth=1.0,
                        markersize=3, label=SAMPLER_LABELS[sampler],
                        **SAMPLER_STYLE[sampler])
        ax.set_title(name.replace("_", " "), fontsize=8)
        ax.set_xlabel(r"$p_i$")
    np.atleast_1d(axes)[0].set_ylabel(r"$\hat p_i$")
    figs.legend_below(fig, np.atleast_1d(axes)[0], ncol=4)
    written += figs.save(fig, out, "sampling_calibration")

    # 3. Radial spectra of the standardized residual Z. The mean field is removed
    #    by construction, so a curve that dips at low frequency is a statement
    #    about ARRANGEMENT and not about the density — which is the whole reason
    #    this is not the spectrum of M.
    fig, axes = plt.subplots(1, len(FIELDS),
                             figsize=(figs.FIG_WIDTH_DOUBLE, 2.2), sharey=True)
    for ax, name in zip(np.atleast_1d(axes), FIELDS):
        for sampler in SAMPLERS:
            spectrum = spectra.get((name, sampler))
            if spectrum is None:
                continue
            ax.plot(spectrum["radial_centers"], spectrum["radial_values"],
                    label=SAMPLER_LABELS[sampler], linewidth=1.0,
                    color=SAMPLER_STYLE[sampler]["color"],
                    linestyle=SAMPLER_STYLE[sampler]["linestyle"])
        ax.axvline(LOW_FREQ_CUTOFF, color="#999999", linewidth=0.6,
                   linestyle=":")
        ax.set_title(name.replace("_", " "), fontsize=8)
        ax.set_xlabel("radial frequency / Nyquist")
    np.atleast_1d(axes)[0].set_ylabel("residual power")
    figs.legend_below(fig, np.atleast_1d(axes)[0], ncol=3)
    written += figs.save(fig, out, "sampling_residual_spectra")

    # 4. Temporal autocorrelation of the residual at fixed pixels. Zero is the
    #    independent baseline; negative means the sampler actively avoids
    #    re-picking a pixel at that spacing, which is the property the temporal
    #    axis exists to have.
    fig, axes = plt.subplots(1, len(FIELDS),
                             figsize=(figs.FIG_WIDTH_DOUBLE, 2.2), sharey=True)
    for ax, name in zip(np.atleast_1d(axes), FIELDS):
        for sampler in SAMPLERS:
            spectrum = spectra.get((name, sampler))
            if spectrum is None:
                continue
            ax.plot(spectrum["temporal_lags"],
                    spectrum["temporal_autocorrelation"],
                    label=SAMPLER_LABELS[sampler], linewidth=1.0, markersize=3,
                    **SAMPLER_STYLE[sampler])
        ax.axhline(0.0, color="#000000", linewidth=0.6)
        ax.set_title(name.replace("_", " "), fontsize=8)
        ax.set_xlabel("visit lag")
    np.atleast_1d(axes)[0].set_ylabel("autocorrelation")
    figs.legend_below(fig, np.atleast_1d(axes)[0], ncol=3)
    written += figs.save(fig, out, "sampling_temporal_autocorrelation")
    return written


def _binned_conditional_mean(p: np.ndarray, phat: np.ndarray, bins: int = 12):
    """``(centers, means, standard_errors)`` of phat in equal-COUNT bins of p.

    Equal-count rather than equal-width: p is concentrated near rho with a thin
    high tail, so equal-width bins would put nearly every pixel in one bin and
    leave the interesting tail with a handful.
    """
    order = np.argsort(p.reshape(-1), kind="stable")
    sorted_p = p.reshape(-1)[order]
    sorted_hat = phat.reshape(-1)[order]
    edges = np.linspace(0, sorted_p.size, bins + 1).astype(np.int64)
    centers, means, errors = [], [], []
    for low, high in zip(edges[:-1], edges[1:]):
        if high <= low:
            continue
        chunk = sorted_hat[low:high]
        centers.append(float(sorted_p[low:high].mean()))
        means.append(float(chunk.mean()))
        errors.append(float(chunk.std() / np.sqrt(chunk.size)))
    return np.asarray(centers), np.asarray(means), np.asarray(errors)
