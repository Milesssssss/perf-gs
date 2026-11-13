# Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

"""Regenerate the README's Results figures and tables from the benchmark runs.

Inputs:
  * ``scripts/running/eval_sweep.py`` runs named ``readme_sh{0,3}_{fp32,bf16}``,
    swept over ``--stbn-pixel-fraction`` on the 13 scenes, plus ``readme_conv``
    (held-out PSNR every 100 iterations, test renders saved).
  * ``scripts/shell/_sadgs_run/run_counterparts.sh``: the SADGS reference on the
    same GPU (``outputs/sadgs_counterparts/sh<D>/<scene>``; indoor scenes only).
  * The earlier SADGS SH0 runs on all nine Mip-NeRF 360 scenes
    (``outputs/SADGS_ref_sh0_<scene>``, ``outputs/sadgs_timing/<scene>``): same
    images and holdout, another GPU, so only their quality and splat counts are used.

Outputs: figures in ``assets/``, every cell in ``docs/benchmark_results.csv``, and
draft Markdown tables in ``outputs/readme_results/tables.md``. Missing runs are left
blank, so this can be re-run while the benchmark is going.

The grid, per SH degree and Adam precision (cells are cached, so a rerun resumes)::

    python scripts/running/eval_sweep.py --axis stbn-pixel-fraction \
        --values 1.0,0.5,0.25,0.1,0.02 --name readme_sh0_fp32 --no-plot --extra \
        --sh-degree 0 --adam-moments fp32 --eval-every 0 --preview-every 0 \
        --no-write-ground-truths

Mip-NeRF 360 uses the default ``--downsample 4``, Tanks & Temples ``2`` and Deep
Blending ``1`` (``--scenes`` per group). Dense fp32 cells add ``--run-report
--eval-lpips``; ``readme_conv`` adds ``--run-report --save-test-renders --eval-iters
100 200 ... 3000``.
"""
from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont

REPO = Path(__file__).resolve().parents[2]
TRAIN = REPO / "outputs" / "train"
SADGS = REPO / "outputs" / "sadgs_counterparts"
ASSETS = REPO / "assets"
CSV_OUT = REPO / "docs" / "benchmark_results.csv"
REPORT = REPO / "outputs" / "readme_results"

DATASETS: dict[str, tuple[str, ...]] = {
    "Mip-NeRF 360": ("bicycle", "flowers", "garden", "stump", "treehill",
                     "bonsai", "counter", "kitchen", "room"),
    "Tanks & Temples": ("train", "truck"),
    "Deep Blending": ("drjohnson", "playroom"),
}
SCENES = tuple(scene for scenes in DATASETS.values() for scene in scenes)
FRACTIONS = ("1.0", "0.5", "0.25", "0.1", "0.02")
DENSE = "1.0"
SH_DEGREES = (0, 3)
PRECISIONS = ("fp32", "bf16")
# Metrics that do not depend on the GPU, so the earlier SADGS runs can supply them.
PORTABLE_FIELDS = ("psnr", "ssim", "lpips", "splats")
CONVERGENCE_SCENES = ("bicycle", "kitchen", "train", "drjohnson")
QUALITATIVE_SCENE = "bicycle"
# The whole bicycle in that view, in render pixels (left, top, right, bottom).
QUALITATIVE_BOX = (290, 145, 950, 585)
# The scheduled opacity reset briefly hides most splats (up to -14 dB for one sample);
# the curves skip the samples taken while it recovers.
OPACITY_RESET_ITERS = range(1200, 1400)

# Okabe-Ito hues (SH0 blue, SH3 orange); the SADGS reference is the lighter tint.
COLORS = {("ours", 0): "#0072B2", ("ours", 3): "#E69F00",
          ("sadgs", 0): "#56B4E9", ("sadgs", 3): "#F3CA6B"}
FRACTION_RAMP = {"1.0": "#104281", "0.5": "#1c5cab", "0.25": "#2a78d6",
                 "0.1": "#5598e7", "0.02": "#86b6ef"}
INK = "#222222"
GIB = 1024 ** 3


@dataclass(frozen=True)
class Result:
    psnr: float
    ssim: float | None
    lpips: float | None
    seconds: float | None
    splats: int | None
    vram_gib: float | None


# ---------------------------------------------------------------- loading

def _read_json(path: Path) -> dict | None:
    return json.loads(path.read_text()) if path.is_file() else None


def _gib(value: float | None) -> float | None:
    return None if value is None else value / GIB


def _percent(fraction: str) -> str:
    return f"{float(fraction) * 100:g}%"


def run_dir(name: str, scene: str, fraction: str) -> Path:
    """Run directory eval_sweep.py gives a cell (its value tag replaces '.')."""
    return TRAIN / f"sweep_{name}_{scene}_{fraction.replace('.', 'p')}"


def load_ours(sh: int, precision: str, scene: str, fraction: str) -> Result | None:
    run = run_dir(f"readme_sh{sh}_{precision}", scene, fraction)
    test = _read_json(run / "test_psnr.json")
    if test is None:
        return None
    quality = (_read_json(run / "run_report.json") or {}).get("quality", {})
    return Result(psnr=test["psnr_srgb"], ssim=quality.get("ssim_srgb"),
                  lpips=quality.get("lpips"), seconds=test["train_wall_s"],
                  splats=test["n_gaussians"], vram_gib=_gib(test.get("vram_usage_bytes")))


def load_sadgs(sh: int, scene: str) -> Result | None:
    root = SADGS / f"sh{sh}" / scene
    metrics = (_read_json(root / "results.json") or {}).get("ours_3000")
    if metrics is None:
        return None
    timing = _read_json(root / "timing.json") or {}
    return Result(psnr=metrics["PSNR"], ssim=metrics["SSIM"], lpips=metrics["LPIPS"],
                  seconds=timing.get("loop_wall_s"), splats=timing.get("num_gaussians"),
                  vram_gib=_gib(timing.get("peak_reserved_bytes")))


def load_sadgs_reference(scene: str) -> Result | None:
    root = REPO / "outputs"
    metrics = (_read_json(root / f"SADGS_ref_sh0_{scene}" / "results.json") or {}).get("ours_3000")
    if metrics is None:
        return None
    timing = _read_json(root / "sadgs_timing" / scene / "timing.json") or {}
    return Result(psnr=metrics["PSNR"], ssim=metrics["SSIM"], lpips=metrics["LPIPS"],
                  seconds=None, splats=timing.get("num_gaussians"), vram_gib=None)


def load_curve(scene: str, fraction: str) -> list[dict]:
    path = run_dir("readme_conv", scene, fraction) / "eval_curve.jsonl"
    if not path.is_file():
        return []
    samples = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return [s for s in samples if s["iter"] not in OPACITY_RESET_ITERS]


@dataclass
class Grid:
    """Every loaded result, keyed by (sh, precision, scene, fraction) for ours,
    (sh, scene) for the same-GPU SADGS runs and scene for the SADGS SH0 reference."""
    ours: dict[tuple[int, str, str, str], Result]
    sadgs: dict[tuple[int, str], Result]
    reference: dict[str, Result]

    @classmethod
    def load(cls) -> Grid:
        ours = {}
        for sh in SH_DEGREES:
            for precision in PRECISIONS:
                for scene in SCENES:
                    for fraction in FRACTIONS:
                        result = load_ours(sh, precision, scene, fraction)
                        if result is not None:
                            ours[(sh, precision, scene, fraction)] = result
        sadgs = {(sh, scene): result for sh in SH_DEGREES for scene in SCENES
                 if (result := load_sadgs(sh, scene)) is not None}
        reference = {scene: result for scene in SCENES
                     if (result := load_sadgs_reference(scene)) is not None}
        return cls(ours, sadgs, reference)

    def sadgs_for(self, sh: int, scene: str, field: str) -> Result | None:
        """SH0 quality and splat counts come from the complete earlier reference
        set; times, memory and SH3 from the same-GPU runs."""
        if sh == 0 and field in PORTABLE_FIELDS:
            return self.reference.get(scene)
        return self.sadgs.get((sh, scene))

    def dense(self, sh: int, scene: str, precision: str = "fp32") -> Result | None:
        return self.ours.get((sh, precision, scene, DENSE))


def mean_of(results: list[Result | None], field: str) -> float | None:
    """Mean of ``field`` over every result, or None if any is missing: a mean over
    a partial scene set would not be comparable to a complete one."""
    values = [None if r is None else getattr(r, field) for r in results]
    if not values or any(v is None for v in values):
        return None
    return float(np.mean(values))


# ---------------------------------------------------------------- figures

def _style(ax) -> None:
    ax.grid(axis="y", alpha=0.25, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


def _series(grid: Grid, field: str) -> list[tuple[str, str, str, dict[str, Result | None]]]:
    """(label, color, hatch, per-scene result) for the four per-scene series."""
    series = []
    for sh in SH_DEGREES:
        series.append((f"Ours SH{sh}", COLORS[("ours", sh)], "",
                       {s: grid.dense(sh, s) for s in SCENES}))
    for sh in SH_DEGREES:
        series.append((f"SADGS SH{sh}", COLORS[("sadgs", sh)], "//",
                       {s: grid.sadgs_for(sh, s, field) for s in SCENES}))
    return series


def plot_scene_bars(grid: Grid, field: str, ylabel: str, title: str, fmt: str,
                    out: Path) -> None:
    series = _series(grid, field)
    labels = list(SCENES) + [f"{name}\naverage" for name in DATASETS]
    x = np.arange(len(labels))
    width = 0.8 / len(series)
    fig, ax = plt.subplots(figsize=(20, 6), dpi=130)
    tallest = 0.0
    for index, (label, color, hatch, by_scene) in enumerate(series):
        values = [getattr(by_scene[s], field) if by_scene[s] else None for s in SCENES]
        values += [mean_of([by_scene[s] for s in scenes], field)
                   for scenes in DATASETS.values()]
        heights = [np.nan if v is None else v for v in values]
        offset = (index - (len(series) - 1) / 2) * width
        bars = ax.bar(x + offset, heights, width, label=label, color=color, hatch=hatch,
                      edgecolor="white", linewidth=0.5, zorder=3)
        for bar, value in zip(bars, values):
            if value is None:
                continue
            tallest = max(tallest, value)
            ax.annotate(fmt.format(value), (bar.get_x() + bar.get_width() / 2, value),
                        xytext=(0, 2), textcoords="offset points", ha="center",
                        va="bottom", rotation=90, fontsize=6.5, color=INK)
    ax.axvline(len(SCENES) - 0.5, color="#bbbbbb", lw=1.0, ls="--", zorder=1)
    for boundary in np.cumsum([len(s) for s in DATASETS.values()])[:-1]:
        ax.axvline(boundary - 0.5, color="#e2e2e2", lw=0.8, zorder=1)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, tallest * 1.18 if tallest else 1)
    _style(ax)
    fig.legend(*ax.get_legend_handles_labels(), frameon=False, ncol=len(series),
               loc="upper center", bbox_to_anchor=(0.5, 0.93))
    fig.suptitle(title, y=0.985, fontsize=12)
    fig.subplots_adjust(left=0.04, right=0.995, top=0.85, bottom=0.2)
    fig.savefig(out)
    plt.close(fig)


def plot_vram(grid: Grid, out: Path) -> None:
    """GPU memory per SH degree and Adam precision, averaged over every scene."""
    groups = [f"SH{sh}" for sh in SH_DEGREES]
    bars = [("fp32 Adam", "fp32", COLORS[("ours", 0)]),
            ("bf16 Adam", "bf16", COLORS[("ours", 3)])]
    x = np.arange(len(groups))
    width = 0.8 / len(bars)
    fig, ax = plt.subplots(figsize=(8, 4.8), dpi=130)
    for index, (label, precision, color) in enumerate(bars):
        values = [mean_of([grid.dense(sh, s, precision) for s in SCENES], "vram_gib")
                  for sh in SH_DEGREES]
        heights = [np.nan if v is None else v for v in values]
        offset = (index - (len(bars) - 1) / 2) * width
        drawn = ax.bar(x + offset, heights, width, label=label, color=color,
                       edgecolor="white", linewidth=0.5, zorder=3)
        for bar, value in zip(drawn, values):
            if value is not None:
                ax.annotate(f"{value:.1f}", (bar.get_x() + bar.get_width() / 2, value),
                            xytext=(0, 2), textcoords="offset points", ha="center",
                            va="bottom", fontsize=9, color=INK)
    ax.axhline(8, color="#cc3311", lw=1.0, ls="--", zorder=2)
    ax.annotate("8 GiB", (x[0] - 0.45, 8), xytext=(0, 3), textcoords="offset points",
                ha="left", fontsize=8, color="#cc3311")
    ax.set_xticks(x)
    ax.set_xticklabels(groups)
    ax.set_ylabel("GPU memory (GiB)")
    _style(ax)
    fig.legend(*ax.get_legend_handles_labels(), frameon=False, ncol=2, loc="upper center",
               bbox_to_anchor=(0.5, 0.92), fontsize=9)
    fig.suptitle(f"GPU memory while training, average over the {len(SCENES)} scenes",
                 y=0.985, fontsize=11)
    fig.subplots_adjust(left=0.09, right=0.98, top=0.82, bottom=0.08)
    fig.savefig(out)
    plt.close(fig)


def plot_convergence(out: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 6), dpi=130)
    palette = plt.get_cmap("tab20")
    for index, scene in enumerate(SCENES):
        curve = load_curve(scene, DENSE)
        if curve:
            ax.plot([p["iter"] for p in curve], [p["psnr_srgb"] for p in curve],
                    color=palette(index), label=scene, lw=1.6)
    ax.set_xlabel("iteration")
    ax.set_ylabel("held-out PSNR (dB, sRGB)")
    ax.set_title("Held-out PSNR during training (SH0, every pixel supervised)")
    ax.grid(alpha=0.3)
    ax.legend(ncol=3, fontsize=8, frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def plot_stbn_convergence(out: Path) -> None:
    fig, axes = plt.subplots(1, len(CONVERGENCE_SCENES), figsize=(17, 4.4), dpi=130)
    for ax, scene in zip(axes, CONVERGENCE_SCENES):
        for fraction in ("1.0", "0.25", "0.1", "0.02"):
            curve = load_curve(scene, fraction)
            if curve:
                ax.plot([p["elapsed_s"] for p in curve], [p["psnr_srgb"] for p in curve],
                        color=FRACTION_RAMP[fraction], label=_percent(fraction), lw=1.6)
        ax.set_title(scene, fontsize=11)
        ax.set_xlabel("training time (seconds)")
        ax.grid(alpha=0.3)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
    axes[0].set_ylabel("held-out PSNR (dB)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, title="pixels supervised per step", frameon=False,
               ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.08))
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def _ground_truth(scene: str, image_name: str, size: tuple[int, int]) -> Image.Image:
    folders = [p / scene / sub for p in (REPO / "datasets").iterdir() if p.is_dir()
               for sub in ("images_4", "images")]
    path = next(f / image_name for f in folders if (f / image_name).is_file())
    return Image.open(path).convert("RGB").resize(size, Image.LANCZOS)


def plot_qualitative(out: Path) -> None:
    """One held-out view of QUALITATIVE_SCENE, every panel cropped to QUALITATIVE_BOX:
    ground truth on top, then the final render at each supervised-pixel fraction."""
    panels = []
    for fraction in ("1.0", "0.25", "0.1", "0.02"):
        run = run_dir("readme_conv", QUALITATIVE_SCENE, fraction)
        report = _read_json(run / "run_report.json")
        if report is None:
            return
        view = report["quality"]["per_view"][0]
        render = Image.open(run / "test_renders" / f"cam_{view['cam_idx']:04d}.png").convert("RGB")
        panels.append((f"{_percent(fraction)} pixels, {view['psnr_srgb']:.2f} dB", render))
    panels.insert(0, ("ground truth", _ground_truth(QUALITATIVE_SCENE, view["image_name"],
                                                   panels[0][1].size)))
    tiles = [(label, image.crop(QUALITATIVE_BOX)) for label, image in panels]
    tile_w, tile_h = tiles[0][1].size
    gap, caption = 12, 40
    step_x, step_y = tile_w + gap, caption + tile_h + gap
    # Ground truth centered on the first row, the four renders in a 2x2 grid below.
    origins = [(step_x // 2, 0)] + [(col * step_x, (row + 1) * step_y)
                                    for row in range(2) for col in range(2)]
    sheet = Image.new("RGB", (2 * tile_w + gap, 3 * step_y - gap), "white")
    draw = ImageDraw.Draw(sheet)
    try:
        font = ImageFont.truetype("arial.ttf", 26)
    except OSError:
        font = ImageFont.load_default()
    for (label, tile), (x, y) in zip(tiles, origins):
        draw.text((x + 4, y + 6), label, fill=INK, font=font)
        sheet.paste(tile, (x, y + caption))
    sheet.save(out, quality=92)


# ---------------------------------------------------------------- tables

def _cell(value: float | None, fmt: str) -> str:
    return "" if value is None else fmt.format(value)


def _row(cells: list[str]) -> str:
    return "| " + " | ".join(cells) + " |"


def _table(header: list[str], rows: list[list[str]], align: str = "---:") -> str:
    lines = [_row(header), _row(["---"] + [align] * (len(header) - 1))]
    return "\n".join(lines + [_row(r) for r in rows])


def _scene_rows(grid: Grid, cells) -> list[list[str]]:
    """Per-scene rows, then a bold average row per dataset."""
    rows = [[scene] + cells({scene}) for scene in SCENES]
    for name, scenes in DATASETS.items():
        rows.append([f"**{name}**"] + [f"**{c}**" if c else "" for c in cells(set(scenes))])
    return rows


def scene_table(grid: Grid) -> str:
    """Per scene: our PSNR, time and splat count, then SADGS's PSNR for reference."""
    def cells(scenes: set[str]) -> list[str]:
        ordered = [s for s in SCENES if s in scenes]
        out = [_cell(mean_of([grid.dense(sh, s) for s in ordered], "psnr"), "{:.2f}")
               for sh in SH_DEGREES]
        out += [_cell(mean_of([grid.dense(sh, s) for s in ordered], "seconds"), "{:.1f}")
                for sh in SH_DEGREES]
        out.append(_cell(_millions(mean_of([grid.dense(0, s) for s in ordered], "splats")),
                         "{:.2f}"))
        out += [_cell(mean_of([grid.sadgs_for(sh, s, "psnr") for s in ordered], "psnr"), "{:.2f}")
                for sh in SH_DEGREES]
        return out
    return _table(["Scene", "PSNR SH0", "PSNR SH3", "Time SH0 (seconds)", "Time SH3 (seconds)",
                   "Splats (millions)", "SADGS PSNR SH0", "SADGS PSNR SH3"],
                  _scene_rows(grid, cells))


def cost_table(grid: Grid) -> str:
    """Per dataset and SH degree: training time, splat count and GPU memory."""
    rows = []
    for name, scenes in DATASETS.items():
        for sh in SH_DEGREES:
            runs = [grid.dense(sh, s) for s in scenes]
            rows.append([name, f"SH{sh}", _cell(mean_of(runs, "seconds"), "{:.1f}"),
                         _cell(_millions(mean_of(runs, "splats")), "{:.2f}"),
                         _cell(mean_of(runs, "vram_gib"), "{:.1f}")])
    return _table(["Dataset", "SH", "Time (seconds)", "Splats (millions)", "VRAM (GiB)"], rows)


def dataset_quality_table(grid: Grid) -> str:
    rows = []
    for name, scenes in DATASETS.items():
        for sh in SH_DEGREES:
            runs = [grid.dense(sh, s) for s in scenes]
            rows.append([name, f"SH{sh}", _cell(mean_of(runs, "psnr"), "{:.2f}"),
                         _cell(mean_of(runs, "ssim"), "{:.3f}"),
                         _cell(mean_of(runs, "lpips"), "{:.3f}")])
    return _table(["Dataset", "SH", "PSNR", "SSIM", "LPIPS"], rows)


def _millions(value: float | None) -> float | None:
    return None if value is None else value / 1e6


def stbn_table(grid: Grid, sh: int) -> str:
    rows = []
    for fraction in FRACTIONS:
        row = [_percent(fraction)]
        for scenes in DATASETS.values():
            runs = [grid.ours.get((sh, "fp32", s, fraction)) for s in scenes]
            row += [_cell(mean_of(runs, "psnr"), "{:.2f}"),
                    _cell(mean_of(runs, "seconds"), "{:.1f}")]
        rows.append(row)
    header = ["Pixels / step"]
    for name in DATASETS:
        short = {"Mip-NeRF 360": "M360", "Tanks & Temples": "T&T", "Deep Blending": "DB"}[name]
        header += [f"{short} PSNR", f"{short} time (seconds)"]
    return _table(header, rows)


def precision_table(grid: Grid) -> str:
    rows = []
    for sh in SH_DEGREES:
        for precision in PRECISIONS:
            runs = [grid.dense(sh, s, precision) for s in SCENES]
            row = [f"SH{sh}", precision]
            for scenes in DATASETS.values():
                row.append(_cell(mean_of([grid.dense(sh, s, precision) for s in scenes],
                                         "psnr"), "{:.2f}"))
            row += [_cell(mean_of(runs, "seconds"), "{:.1f}"),
                    _cell(mean_of(runs, "vram_gib"), "{:.2f}")]
            rows.append(row)
    return _table(["SH", "Adam moments", "M360 PSNR", "T&T PSNR", "DB PSNR",
                   "Time (seconds)", "VRAM (GiB)"], rows)


def precision_spread(grid: Grid) -> str:
    """Largest |bf16 - fp32| PSNR gap over every paired cell of the grid."""
    gaps = [abs(grid.ours[(sh, "bf16", s, f)].psnr - grid.ours[(sh, "fp32", s, f)].psnr)
            for sh in SH_DEGREES for s in SCENES for f in FRACTIONS
            if (sh, "bf16", s, f) in grid.ours and (sh, "fp32", s, f) in grid.ours]
    if not gaps:
        return "no paired bf16/fp32 cells yet"
    return (f"{len(gaps)} paired cells: mean |dPSNR| {np.mean(gaps):.3f} dB, "
            f"max {np.max(gaps):.3f} dB")


def _csv_values(result: Result) -> list:
    return [round(v, 4) if isinstance(v, float) else v for v in asdict(result).values()]


def write_summary_csv(grid: Grid, out: Path) -> None:
    """Every cell; ``gpu`` says where its time and memory were measured."""
    with out.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["method", "gpu", "sh", "adam_moments", "scene", "pixel_fraction",
                         *Result.__dataclass_fields__])
        for (sh, precision, scene, fraction), result in sorted(grid.ours.items()):
            writer.writerow(["ours", "RTX 4080", sh, precision, scene, fraction,
                             *_csv_values(result)])
        for (sh, scene), result in sorted(grid.sadgs.items()):
            writer.writerow(["sadgs", "RTX 4080", sh, "fp32", scene, DENSE, *_csv_values(result)])
        for scene, result in sorted(grid.reference.items()):
            writer.writerow(["sadgs", "RTX 3090", 0, "fp32", scene, DENSE, *_csv_values(result)])


# ---------------------------------------------------------------- main

def main() -> int:
    grid = Grid.load()
    if not grid.ours:
        print("no benchmark runs found")
        return 1
    REPORT.mkdir(parents=True, exist_ok=True)
    plot_scene_bars(grid, "psnr", "held-out PSNR (dB, sRGB)",
                    "Held-out PSNR per scene (3000 iterations, every pixel supervised)",
                    "{:.2f}", ASSETS / "psnr_sweep.png")
    plot_scene_bars(grid, "seconds", "training time (seconds)",
                    "Training time per scene (3000 iterations, RTX 4080)",
                    "{:.0f}", ASSETS / "timing_sweep.png")
    plot_vram(grid, ASSETS / "vram.png")
    plot_convergence(ASSETS / "convergence.png")
    plot_stbn_convergence(ASSETS / "stbn_convergence.png")
    plot_qualitative(ASSETS / "stbn_qualitative.jpg")
    write_summary_csv(grid, CSV_OUT)
    tables = [
        "## Per scene", scene_table(grid),
        "## Time and memory", cost_table(grid),
        "## Dataset averages", dataset_quality_table(grid),
        "## STBN SH0", stbn_table(grid, 0),
        "## STBN SH3", stbn_table(grid, 3),
        "## Adam moment precision", precision_table(grid),
        f"bf16 vs fp32 over the grid: {precision_spread(grid)}",
    ]
    (REPORT / "tables.md").write_text("\n\n".join(tables) + "\n", encoding="utf-8")
    print(f"{len(grid.ours)} runs, {len(grid.sadgs)} SADGS runs -> {REPORT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
