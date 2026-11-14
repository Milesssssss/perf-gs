# Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

"""Figure helpers: one consistent style, PDF for the paper and PNG for review.

A failing matplotlib logs and skips the figure, never costing the numbers already on disk.
"""
from __future__ import annotations

from pathlib import Path

# Okabe-Ito colour-blind-safe palette, plus a marker and dash per regime so greyscale prints survive.
REGIME_STYLE: dict[str, dict] = {
    "full":       {"color": "#000000", "marker": "o", "linestyle": "-",
                   "label": "Dense (100%)"},
    "iid_white":  {"color": "#E69F00", "marker": "s", "linestyle": "--",
                   "label": "IID white"},
    "indep_blue": {"color": "#56B4E9", "marker": "^", "linestyle": "-.",
                   "label": "Indep. blue noise"},
    "stbn":       {"color": "#009E73", "marker": "D", "linestyle": "-",
                   "label": "STBN"},
}

FRACTION_STYLE: dict[float, dict] = {
    0.10: {"color": "#CC79A7", "marker": "v", "linestyle": ":"},
    0.25: {"color": "#0072B2", "marker": "o", "linestyle": "-"},
    0.50: {"color": "#D55E00", "marker": "s", "linestyle": "--"},
}

PNG_DPI = 160
# Single-column Eurographics width is ~3.3in; most of these are double-column.
FIG_WIDTH_SINGLE = 3.4
FIG_WIDTH_DOUBLE = 7.0


def pyplot():
    """Configured pyplot, or ``None`` when matplotlib is unusable."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        plt.rcParams.update({
            "figure.dpi": 110,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "legend.fontsize": 7,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linewidth": 0.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            # Keep text as text in the PDF so it can be searched and restyled.
            "pdf.fonttype": 42,
        })
        return plt
    except Exception as error:                             # noqa: BLE001
        print(f"[figures] matplotlib unavailable ({error}); figures skipped. "
              f"All numbers are already in tables/ and raw/.", flush=True)
        return None


def save(fig, figures_dir: Path, stem: str, *, also_png: bool = True) -> list[Path]:
    """Save ``fig`` as PDF (+PNG) under ``figures_dir``; never raises."""
    written: list[Path] = []
    figures_dir.mkdir(parents=True, exist_ok=True)
    for suffix, kwargs in ((".pdf", {}), (".png", {"dpi": PNG_DPI})):
        if suffix == ".png" and not also_png:
            continue
        path = figures_dir / f"{stem}{suffix}"
        try:
            fig.savefig(path, **kwargs)
            written.append(path)
        except Exception as error:                         # noqa: BLE001
            print(f"[figures] could not write {path.name}: {error}", flush=True)
    try:
        import matplotlib.pyplot as plt
        plt.close(fig)
    except Exception:                                      # noqa: BLE001
        pass
    return written


def style_for(regime: str) -> dict:
    return REGIME_STYLE.get(regime, {"color": "#777777", "marker": ".",
                                     "linestyle": "-", "label": regime})


def annotate_empty(ax, message: str = "no data") -> None:
    """Mark an axis that has nothing to show, so an empty panel cannot be read as zero."""
    ax.text(0.5, 0.5, message, transform=ax.transAxes, ha="center",
            va="center", fontsize=7, color="#999999")
    ax.set_xticks([])
    ax.set_yticks([])


def zoom_to_data(ax, values, *, pad_fraction: float = 0.08,
                 min_pad: float = 0.05) -> None:
    """Fit the y-axis to the data with spread-relative padding (regimes differ by tenths of a dB)."""
    finite = [v for v in values if v is not None and v == v]
    if not finite:
        return
    low, high = min(finite), max(finite)
    pad = max(min_pad, (high - low) * pad_fraction)
    ax.set_ylim(low - pad, high + pad)


def legend_below(fig, ax, ncol: int = 4) -> None:
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=ncol,
                   bbox_to_anchor=(0.5, -0.02))
