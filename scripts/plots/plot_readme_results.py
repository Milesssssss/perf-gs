# Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

"""Regenerate the figures and tables in docs/benchmarks.md from the benchmark runs (missing runs are left blank).

To regenerate only the PSNR and training-time figures from the two published CSVs:
    python scripts/plots/plot_readme_results.py --overview-only

Ours: grid below over all 13 scenes, each at its 3DGS eval resolution (dense fp32 adds --run-report --eval-lpips; readme_conv adds --run-report --save-test-renders --eval-iters 100 200 ... 3000); SADGS: run_counterparts.sh.
    python scripts/running/eval_sweep.py --axis stbn-pixel-fraction --values 1.0,0.5,0.25,0.1,0.02 --name readme_sh0_fp32 --no-plot --extra --sh-degree 0 --adam-moments fp32 --eval-every 0 --preview-every 0 --no-write-ground-truths
"""
from __future__ import annotations

import argparse
import csv
import json
import re
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
OPTICS_CSV = REPO / "docs" / "mipnerf360_3k_results.csv"
OPTICS_RECIPES = {
    "optics-fast": ("Fixed falloff + stereo SH3 (2-seed mean)", "#009E73"),
    "optics-adaptive": ("Adaptive falloff + stereo SH3 (2-seed mean)", "#AA4499"),
}
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
# --preset values -> their Adam moments; balanced is the default, so its runs are the dense fp32 grid.
PRESETS = {"ultrafast": "bf16", "perf": "bf16", "balanced": "fp32", "quality": "fp32"}
CONVERGENCE_SCENES = ("bicycle", "kitchen", "train", "drjohnson")
QUALITATIVE_SCENE = "bicycle"
# The whole bicycle in that view, in render pixels (left, top, right, bottom).
QUALITATIVE_BOX = (290, 145, 950, 585)
# The opacity reset briefly hides most splats (up to -14 dB), so curves skip samples while it recovers.
OPACITY_RESET_ITERS = range(1200, 1400)

# Okabe-Ito hues: SH0 blue, SH3 orange.
BLUE, ORANGE = "#0072B2", "#E69F00"
COLORS = {0: BLUE, 3: ORANGE}
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
    splats: int | float | None
    vram_gib: float | None
    # Image downsample factor the run trained and was evaluated at.
    downsample: int | None


def _read_json(path: Path) -> dict | None:
    return json.loads(path.read_text()) if path.is_file() else None


def _gib(value: float | None) -> float | None:
    return None if value is None else value / GIB


def _percent(fraction: str) -> str:
    return f"{float(fraction) * 100:g}%"


def _sadgs_downsample(run: Path) -> int | None:
    """N of the images_N folder a SADGS run trained on, from its saved arguments."""
    args = run / "cfg_args"
    match = re.search(r"images='images_(\d+)'", args.read_text()) if args.is_file() else None
    return int(match.group(1)) if match else None


def run_dir(name: str, scene: str, fraction: str) -> Path:
    """Run directory eval_sweep.py gives a cell (its value tag replaces '.')."""
    return TRAIN / f"sweep_{name}_{scene}_{fraction.replace('.', 'p')}"


def load_ours(sh: int, precision: str, scene: str, fraction: str) -> Result | None:
    return load_run(run_dir(f"readme_sh{sh}_{precision}", scene, fraction))


def load_run(run: Path) -> Result | None:
    test = _read_json(run / "test_psnr.json")
    if test is None:
        return None
    quality = (_read_json(run / "run_report.json") or {}).get("quality", {})
    cli_args = (_read_json(run / "config.json") or {}).get("cli_args", {})
    return Result(psnr=test["psnr_srgb"], ssim=quality.get("ssim_srgb"),
                  lpips=quality.get("lpips"), seconds=test["train_wall_s"],
                  splats=test["n_gaussians"], vram_gib=_gib(test.get("vram_usage_bytes")),
                  downsample=cli_args.get("downsample"))


def load_sadgs(sh: int, scene: str) -> Result | None:
    root = SADGS / f"sh{sh}" / scene
    metrics = (_read_json(root / "results.json") or {}).get("ours_3000")
    if metrics is None:
        return None
    timing = _read_json(root / "timing.json") or {}
    # Reserving more than the GPU holds pages into host memory, so the loop time is not comparable.
    paged = timing.get("peak_reserved_bytes", 0) > timing.get("gpu_total_bytes", float("inf"))
    return Result(psnr=metrics["PSNR"], ssim=metrics["SSIM"], lpips=metrics["LPIPS"],
                  seconds=None if paged else timing.get("loop_wall_s"),
                  splats=timing.get("num_gaussians"),
                  vram_gib=_gib(timing.get("peak_reserved_bytes")),
                  downsample=_sadgs_downsample(root))


def load_sadgs_reference(scene: str) -> Result | None:
    root = REPO / "outputs"
    run = root / f"SADGS_ref_sh0_{scene}"
    metrics = (_read_json(run / "results.json") or {}).get("ours_3000")
    if metrics is None:
        return None
    timing = _read_json(root / "sadgs_timing" / scene / "timing.json") or {}
    # Measured on another GPU, so only quality and splat count are comparable.
    return Result(psnr=metrics["PSNR"], ssim=metrics["SSIM"], lpips=metrics["LPIPS"],
                  seconds=None, splats=timing.get("num_gaussians"), vram_gib=None,
                  downsample=_sadgs_downsample(run))


def load_curve(scene: str, fraction: str) -> list[dict]:
    path = run_dir("readme_conv", scene, fraction) / "eval_curve.jsonl"
    if not path.is_file():
        return []
    samples = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return [s for s in samples if s["iter"] not in OPACITY_RESET_ITERS]


@dataclass
class Grid:
    """Every loaded result, keyed by (sh, precision, scene, fraction) for ours, (sh, preset, scene)
    for the presets, (sh, scene) for the same-GPU SADGS runs and scene for the SADGS SH0 reference."""
    ours: dict[tuple[int, str, str, str], Result]
    presets: dict[tuple[int, str, str], Result]
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
        presets = {}
        for sh in SH_DEGREES:
            for preset in PRESETS:
                for scene in SCENES:
                    result = (ours.get((sh, "fp32", scene, DENSE)) if preset == "balanced"
                              else load_run(run_dir(f"readme_presets_sh{sh}", scene, preset)))
                    if result is not None:
                        presets[(sh, preset, scene)] = result
        sadgs = {(sh, scene): result for sh in SH_DEGREES for scene in SCENES
                 if (result := load_sadgs(sh, scene)) is not None}
        reference = {scene: result for scene in SCENES
                     if (result := load_sadgs_reference(scene)) is not None}
        return cls(ours, presets, sadgs, reference)

    def dense(self, sh: int, scene: str, precision: str = "fp32") -> Result | None:
        return self.ours.get((sh, precision, scene, DENSE))


def mean_of(results: list[Result | None], field: str) -> float | None:
    """Mean of ``field`` over every result, or None if any is missing (a partial mean is not comparable)."""
    values = [None if r is None else getattr(r, field) for r in results]
    if not values or any(v is None for v in values):
        return None
    return float(np.mean(values))


def _style(ax) -> None:
    ax.grid(axis="y", alpha=0.25, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


def load_published_defaults(path: Path) -> Grid:
    """The historical full-pixel, fp32 baseline for the overview figures."""
    dense = {}
    with path.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            if (row["method"], row["adam_moments"], row["pixel_fraction"]) == ("ours", "fp32", DENSE):
                dense[int(row["sh"]), "fp32", row["scene"], DENSE] = Result(
                    float(row["psnr"]), float(row["ssim"]) if row["ssim"] else None,
                    float(row["lpips"]) if row["lpips"] else None, float(row["seconds"]),
                    int(row["splats"]) if row["splats"] else None,
                    float(row["vram_gib"]) if row["vram_gib"] else None, int(row["downsample"]))
    return Grid(dense, {}, {}, {})


def load_published_optics(path: Path) -> dict[str, dict[str, Result]]:
    """Pair the seeds for each scene and publish the adaptive recipe after both sweeps finish."""
    fields = ("psnr_db", "ssim", "vgg_lpips", "train_loop_s")
    seeds = ("42", "1")
    scenes = DATASETS["Mip-NeRF 360"]
    runs = {}
    if path.is_file():
        with path.open(newline="", encoding="utf-8-sig") as handle:
            for row in csv.DictReader(handle):
                if row["recipe"] not in OPTICS_RECIPES or row["seed"] not in seeds or row["scene"] not in scenes:
                    continue
                key = row["recipe"], row["scene"], row["seed"]
                if key in runs or row["steps"] != "3000":
                    raise ValueError(f"Unexpected or duplicate benchmark cell {key} in {path}")
                values = tuple(float(row[f]) if row[f] else np.nan for f in fields)
                runs[key] = None
                if np.isfinite(values).all():
                    splats = int(row["final_gaussians"]) if row.get("final_gaussians") else None
                    vram_gib = _gib(int(row["vram_usage_bytes"])) if row.get("vram_usage_bytes") else None
                    runs[key] = Result(*values, splats, vram_gib, None)
    published = {}
    for recipe in OPTICS_RECIPES:
        averaged = {}
        for scene in scenes:
            pair = [runs.get((recipe, scene, seed)) for seed in seeds]
            if all(result is not None for result in pair):
                averaged[scene] = Result(*(mean_of(pair, field) for field in Result.__dataclass_fields__))
        if averaged and (recipe == "optics-fast" or len(averaged) == len(scenes)):
            published[recipe] = averaged
    return published


def plot_scene_bars(grid: Grid, field: str, ylabel: str, title: str, fmt: str, out: Path,
                    from_zero: bool = True, *, optics: dict[str, dict[str, Result]] | None = None) -> None:
    """Per-scene bars plus dataset averages: Mip-NeRF 360 on the first row, the other datasets on the second."""
    defaults = [(f"Historical default SH{sh} (1 run)", COLORS[sh], {s: grid.dense(sh, s) for s in SCENES})
                for sh in SH_DEGREES]
    rows = [("Mip-NeRF 360",), ("Tanks & Temples", "Deep Blending")]
    groups = []
    for names in rows:
        row = []
        for name in names:
            row += [(scene, (scene,)) for scene in DATASETS[name]]
            row.append((f"{name} average", DATASETS[name]))
        groups.append(row)
    slots = max(len(row) for row in groups)
    # Keep the text legible when the documentation scales the figure to its width.
    with plt.rc_context({"font.size": 13}):
        fig, axes = plt.subplots(len(rows), 1, figsize=(12, 10), dpi=130, sharey=True,
                                 layout="constrained")
        drawn = []
        for ax, row in zip(axes, groups):
            x = np.arange(len(row))
            series = list(defaults)
            if ax is axes[0] and optics:
                series.extend((*OPTICS_RECIPES[recipe], values) for recipe, values in optics.items())
            width = 0.82 / len(series)
            for index, (label, color, by_scene) in enumerate(series):
                values = [mean_of([by_scene.get(s) for s in scenes], field) for _, scenes in row]
                drawn += [v for v in values if v is not None]
                offset = (index - (len(series) - 1) / 2) * width
                ax.bar(x + offset, [np.nan if v is None else v for v in values], width,
                       label=label, color=color, edgecolor="white",
                       linewidth=0.5, zorder=3)
                for xv, value in zip(x + offset, values):
                    if value is not None:
                        ax.annotate(fmt.format(value), (xv, value), xytext=(0, 3),
                                    textcoords="offset points", ha="center", va="bottom",
                                    rotation=90, fontsize=10 if len(series) > 2 else 12, color=INK)
            for i, (label, scenes) in enumerate(row):
                if len(scenes) > 1:
                    ax.axvline(i - 0.5, color="#bbbbbb", lw=1.0, ls="--", zorder=1)
                    if i + 1 < len(row):
                        ax.axvline(i + 0.5, color="#e2e2e2", lw=0.8, zorder=1)
            ax.set_xticks(x, [label for label, _ in row], rotation=35, ha="right",
                          rotation_mode="anchor")
            ax.set_xlim(-0.6, slots - 0.4)
            ax.set_ylabel(ylabel)
            _style(ax)
        bottom = 0.0 if from_zero else np.floor(min(drawn)) - 1
        # Headroom for the rotated value labels.
        axes[0].set_ylim(bottom, max(drawn) + 0.2 * (max(drawn) - bottom))
        axes[-1].legend(*axes[0].get_legend_handles_labels(), frameon=False, loc="center right", fontsize=11)
        fig.suptitle(title, fontsize=14)
        fig.savefig(out)
    plt.close(fig)


def plot_vram(grid: Grid, out: Path) -> None:
    """GPU memory per SH degree and Adam precision, averaged over every scene."""
    groups = [f"SH{sh}" for sh in SH_DEGREES]
    bars = [("fp32 Adam", "fp32", BLUE), ("bf16 Adam", "bf16", ORANGE)]
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
    """Contact sheet of one cropped held-out view: ground truth on top, the four pixel-fraction renders 2x2 below."""
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


def _cell(value: float | None, fmt: str) -> str:
    return "" if value is None else fmt.format(value)


def _row(cells: list[str]) -> str:
    return "| " + " | ".join(cells) + " |"


def _table(header: list[str], rows: list[list[str]], align: str = "---:") -> str:
    lines = [_row(header), _row(["---"] + [align] * (len(header) - 1))]
    return "\n".join(lines + [_row(r) for r in rows])


def _dataset_runs(grid: Grid, optics: dict[str, dict[str, Result]] | None, name: str):
    scenes = DATASETS[name]
    for sh in SH_DEGREES:
        yield f"Default SH{sh} (1 run)", [grid.dense(sh, s) for s in scenes]
    if name == "Mip-NeRF 360" and optics:
        for recipe, values in optics.items():
            yield OPTICS_RECIPES[recipe][0], [values.get(s) for s in scenes]


def cost_table(grid: Grid, optics: dict[str, dict[str, Result]] | None = None) -> str:
    """Per dataset and SH degree: training time, splat count and GPU memory."""
    rows = []
    for name in DATASETS:
        for label, runs in _dataset_runs(grid, optics, name):
            rows.append([name, label, _cell(mean_of(runs, "seconds"), "{:.1f}"),
                         _cell(_millions(mean_of(runs, "splats")), "{:.2f}"),
                         _cell(mean_of(runs, "vram_gib"), "{:.1f}")])
    return _table(["Dataset", "Recipe", "Train loop (seconds)", "Splats (millions)", "VRAM (GiB)"], rows)


def dataset_quality_table(grid: Grid, optics: dict[str, dict[str, Result]] | None = None) -> str:
    """Per dataset and SH degree: held-out quality and training time."""
    rows = []
    for name in DATASETS:
        for label, runs in _dataset_runs(grid, optics, name):
            rows.append([name, label, _cell(mean_of(runs, "psnr"), "{:.2f}"),
                         _cell(mean_of(runs, "ssim"), "{:.3f}"),
                         _cell(mean_of(runs, "lpips"), "{:.3f}"),
                         _cell(mean_of(runs, "seconds"), "{:.1f}")])
    return _table(["Dataset", "Recipe", "PSNR", "SSIM", "LPIPS", "Train loop (seconds)"], rows)


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


def preset_table(grid: Grid) -> str:
    """Per SH degree and preset: each dataset's PSNR and time, and LPIPS over all scenes."""
    rows = []
    for sh in SH_DEGREES:
        for preset in PRESETS:
            row = [f"SH{sh}", preset]
            for scenes in DATASETS.values():
                runs = [grid.presets.get((sh, preset, s)) for s in scenes]
                row += [_cell(mean_of(runs, "psnr"), "{:.2f}"),
                        _cell(mean_of(runs, "seconds"), "{:.1f}")]
            every = [grid.presets.get((sh, preset, s)) for s in SCENES]
            rows.append(row + [_cell(mean_of(every, "lpips"), "{:.3f}")])
    header = ["SH", "Preset"]
    for short in ("M360", "T&T", "DB"):
        header += [f"{short} PSNR", f"{short} time (seconds)"]
    return _table(header + ["LPIPS"], rows)


def precision_spread(grid: Grid) -> str:
    """Mean and max |bf16 - fp32| PSNR gap over every paired cell of the grid."""
    gaps = [abs(grid.ours[(sh, "bf16", s, f)].psnr - grid.ours[(sh, "fp32", s, f)].psnr)
            for sh in SH_DEGREES for s in SCENES for f in FRACTIONS
            if (sh, "bf16", s, f) in grid.ours and (sh, "fp32", s, f) in grid.ours]
    if not gaps:
        return "no paired bf16/fp32 cells yet"
    return (f"{len(gaps)} paired cells: mean |dPSNR| {np.mean(gaps):.3f} dB, "
            f"max {np.max(gaps):.3f} dB")


def _csv_values(result: Result) -> list:
    return [round(v, 4) if isinstance(v, float) else v for v in asdict(result).values()]


def _summary_rows(grid: Grid):
    for (sh, precision, scene, fraction), result in sorted(grid.ours.items()):
        yield ["ours", "RTX 4080", str(sh), precision, scene, fraction, *_csv_values(result)]
    for (sh, preset, scene), result in sorted(grid.presets.items()):
        if preset != "balanced":  # already written above as the dense fp32 cell
            run = run_dir(f"readme_presets_sh{sh}", scene, preset)
            config = _read_json(run / "config.json") or {}
            fraction = config.get("cli_args", {}).get("stbn_pixel_fraction")
            yield [f"ours --preset {preset}", "RTX 4080", str(sh), PRESETS[preset], scene,
                   fraction, *_csv_values(result)]
    for (sh, scene), result in sorted(grid.sadgs.items()):
        yield ["sadgs", "RTX 4080", str(sh), "fp32", scene, DENSE, *_csv_values(result)]
    for scene, result in sorted(grid.reference.items()):
        yield ["sadgs", "RTX 3090", "0", "fp32", scene, DENSE, *_csv_values(result)]


def write_summary_csv(grid: Grid, out: Path) -> None:
    """Refresh measured fields while retaining published results absent from a local archive."""
    header = ["method", "gpu", "sh", "adam_moments", "scene", "pixel_fraction",
              *Result.__dataclass_fields__]

    def cell_key(row):
        # Preset runs sweep the named preset. Their pixel fraction is recorded metadata.
        return tuple(row[:5] if row[0].startswith("ours --preset ") else row[:6])

    rows = {}
    if out.is_file():
        with out.open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.reader(handle)
            if next(reader, None) != header:
                raise ValueError(f"Unexpected benchmark columns in {out}")
            for row in reader:
                key = cell_key(row)
                if key in rows or len(row) != len(header):
                    raise ValueError(f"Duplicate or malformed benchmark cell {key} in {out}")
                rows[key] = row
    for row in _summary_rows(grid):
        key = cell_key(row)
        if key in rows:
            row = [old if new is None else new for old, new in zip(rows[key], row)]
        rows[key] = row
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows.values())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--overview-only", action="store_true",
                        help="regenerate only psnr_sweep.png and timing_sweep.png from the published CSVs")
    args = parser.parse_args()
    grid = load_published_defaults(CSV_OUT) if args.overview_only else Grid.load()
    if not grid.ours:
        print("no benchmark runs found")
        return 1
    optics = load_published_optics(OPTICS_CSV)
    plot_scene_bars(grid, "psnr", "Held-out PSNR (dB, sRGB)",
                    "Held-out PSNR after 3,000 steps, full pixel supervision",
                    "{:.2f}", ASSETS / "psnr_sweep.png", from_zero=False, optics=optics)
    plot_scene_bars(grid, "seconds", "Training loop (seconds)",
                    "Training-loop time for 3,000 steps on an RTX 4080",
                    "{:.1f}", ASSETS / "timing_sweep.png", optics=optics)
    if args.overview_only:
        print(dataset_quality_table(grid, optics))
        print(cost_table(grid, optics))
        return 0
    REPORT.mkdir(parents=True, exist_ok=True)
    plot_vram(grid, ASSETS / "vram.png")
    plot_convergence(ASSETS / "convergence.png")
    plot_stbn_convergence(ASSETS / "stbn_convergence.png")
    plot_qualitative(ASSETS / "stbn_qualitative.jpg")
    write_summary_csv(grid, CSV_OUT)
    tables = [
        "## Quality and training time", dataset_quality_table(grid, optics),
        "## Time and memory", cost_table(grid, optics),
        "## STBN SH0", stbn_table(grid, 0),
        "## STBN SH3", stbn_table(grid, 3),
        "## Adam moment precision", precision_table(grid),
        f"bf16 vs fp32 over the grid: {precision_spread(grid)}",
        "## Presets", preset_table(grid),
    ]
    (REPORT / "tables.md").write_text("\n\n".join(tables) + "\n", encoding="utf-8")
    print(f"{len(grid.ours)} runs, {len(grid.sadgs)} SADGS runs -> {REPORT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
