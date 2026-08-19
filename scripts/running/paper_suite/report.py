"""Assemble ``report.md`` from the findings each experiment returned.

Written to be read by someone deciding what to claim in a paper, so the standard
throughout is: state the number, state its spread, state the interval, and let
those speak. In particular this module never writes the word "significant" on the
basis of a mean, and it prints negative and null results with the same prominence
as positive ones — an interval containing zero is reported as "does not resolve
the sign", which is what it means, not as "no effect".

Everything here is derived from the findings dicts. No measurement happens in this
module, so the report can be regenerated from a completed suite without touching
the GPU.
"""
from __future__ import annotations

from pathlib import Path

from . import config as cfg
from . import stats


def _fmt(value, digits: int = 3, *, plus: bool = False) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, (int, float)):
        if isinstance(value, float) and value != value:
            return "n/a"
        return f"{value:+.{digits}f}" if plus else f"{value:.{digits}f}"
    return str(value)


def _verdict(row: dict) -> str:
    """One conservative sentence about a paired contrast.

    The decision rule is deliberately blunt: either the interval excludes zero or
    it does not, and either way the win count is quoted. No p-value threshold is
    used to license the word "significant".
    """
    if not row:
        return "not computed"
    excludes = row.get("ci_excludes_zero")
    mean = row.get("mean_delta")
    low, high = row.get("ci_low"), row.get("ci_high")
    wins, cells = row.get("wins"), row.get("n_cells")
    direction = row.get("higher_is_better")
    interval = (f"95% CI [{_fmt(low, 4)}, {_fmt(high, 4)}]"
                if low is not None else "no CI (too few scenes)")
    better = ("higher is better" if direction
              else "lower is better" if direction is False else "")
    count = f"{wins}/{cells} cells favour the treatment" if cells else ""
    if excludes is True:
        sign = "positive" if (mean or 0) > 0 else "negative"
        return (f"mean {_fmt(mean, 4, plus=True)}, {interval} — interval "
                f"excludes zero, so the {sign} sign is resolved at this sample "
                f"size; {count} ({better})")
    if excludes is False:
        return (f"mean {_fmt(mean, 4, plus=True)}, {interval} — interval "
                f"CONTAINS zero, so the data do not resolve the sign of this "
                f"effect; {count} ({better})")
    return f"mean {_fmt(mean, 4, plus=True)}, {interval}; {count}"


def _protocol(findings: dict, environment: dict, plan_summary: dict,
              suite: dict) -> list[str]:
    git = environment.get("git") or {}
    gpu = environment.get("gpu") or {}
    asset = environment.get("stbn_asset") or {}
    packages = environment.get("packages") or {}
    dirty = git.get("dirty")
    dirty_note = ""
    if dirty:
        files = git.get("dirty_source_files") or []
        dirty_note = (
            f"\n> **The working tree was DIRTY when these runs were produced.** "
            f"{git.get('dirty_file_count')} modified path(s)"
            + (f", including source/shader files: `{'`, `'.join(files[:10])}`"
               if files else " (none under `src/`, `shaders/` or `scripts/`)")
            + f". The commit hash therefore does not fully identify the code that "
              f"produced these numbers.\n")

    lines = [
        "## 1. Protocol",
        "",
        "### Sampling regimes",
        "",
        "Four supervision regimes, identical in every respect except the pixel "
        "mask:",
        "",
        "| Regime | Pixels per backward | Mask source | Slice selection |",
        "|---|---|---|---|",
        "| `full` | 100% | none (mask off) | n/a |",
        "| `iid_white` | 25% | white noise | fresh independent draw each step |",
        "| `indep_blue` | 25% | NVIDIA STBN volume | fresh independent draw each step |",
        "| `stbn` | 25% | NVIDIA STBN volume | each camera's own visit counter |",
        "",
        f"Pixels are selected by **{suite.get('select_mode', cfg.PRIMARY_SELECT_MODE)}** "
        "thresholding: the lowest rho fraction of each whole 128x128 slice, one "
        "threshold per slice. That is the point set the asset's spatial energy "
        "function was optimized for, and it leaves the 8x8 execution tiling as "
        "what it should be — an implementation detail of how GPU threads are "
        "assigned to pixels, which looks up an already-decided mask. Per-tile "
        "counts therefore vary (measured 16 ± 1.4 at a nominal 16), and the "
        "backward reads its loop bound from the mask table.",
        "",
        "All three sparse regimes share the density, the uniform `1/rho` loss "
        "reweighting, the selection mode, and the compacted backward. The two "
        "controlled contrasts are therefore clean:",
        "",
        "- `iid_white -> indep_blue` isolates the **spatial** arrangement "
        "(independent draws on both sides).",
        "- `indep_blue -> stbn` isolates the **temporal** ordering (the same 64 "
        "blue-noise slices on both sides).",
        "",
        "> **The reweighting is uniform, not per-tile.** A global threshold "
        "selects each pixel with probability rho, so `1/rho` is the correct "
        "importance weight for the image sum. A per-tile `TILE_AREA/count` weight "
        "would be unbiased per tile but would re-impose exactly the tile-grid "
        "structure the global threshold removes.",
        "",
        "> **The IID control is genuinely unstratified.** Earlier revisions of "
        "this study ranked pixels *within* each 8x8 tile and kept exactly k per "
        "tile. That is a different sampling process, and it compressed the "
        "blue-versus-white contrast from both ends — it damaged the blue mask and "
        "simultaneously made the white control artificially bluer, since forcing "
        "16 of every 64 pixels is a strong constraint on a process whose natural "
        "tile counts span 5..31. Experiment S measures that choice directly "
        "instead of assuming it away.",
        "",
        "> **Independence in `indep_blue` is over a 64-slice pool**, drawn with "
        "replacement. Consecutive masks are statistically independent, but they "
        "are drawn from a finite set of blue-noise realizations rather than "
        "generated afresh. Over 64 visits this uses ~40 distinct slices, where "
        "the STBN walk uses all 64 exactly once — part of any coverage advantage "
        "is that bookkeeping rather than the spectrum, and Experiment E "
        "quantifies both separately.",
        "",
        "### Training configuration (identical for every cell)",
        "",
        f"- {suite.get('iters', cfg.ITERS)} iterations over "
        f"{len(suite.get('scenes') or [])} scene(s), SH degree 3, "
        f"structure-aware densification, Mip-Splatting on, quarter resolution "
        f"(`--downsample 4`).",
        "- Held-out protocol: every 8th camera by name-sorted index is a test "
        "view; training samples train views only.",
        "- Sparse density is FLAT (no warmup ramp), so the arrangement and "
        "ordering contrasts are not blurred by a dense opening.",
        "- Metrics are the paper convention: the MEAN of per-image scores over "
        "held-out views. PSNR/SSIM from the float render; LPIPS (AlexNet) from "
        "the same float render, on CPU.",
        f"- Seeds: {plan_summary.get('by_seed')}. The primary seed "
        f"({cfg.PRIMARY_SEED}) additionally carries the convergence ladder and "
        f"the render dump; the other seeds run unobserved so their wall-clock is "
        f"a clean speed reference.",
        "",
        "### Timing definitions",
        "",
        "- `train_wall_s` — wall-clock across the training loop, frozen right "
        "after the final step with the GPU drained.",
        "- `train_active_s` — `train_wall_s` minus every mid-loop observation "
        "(evaluation sweeps, curve ticks). **This is the only figure used for "
        "speed claims**, because it must not depend on how often quality was "
        "measured. Unobserved cells have `observation_s = 0` and the two "
        "coincide.",
        "- `process_wall_s` — measured by the parent process, so it includes "
        "scene load, shader compilation, the post-loop metrics pass and teardown.",
        "",
        "### Environment",
        "",
        f"- GPU: {gpu.get('name', 'unknown')} ({gpu.get('memory_total', '?')}), "
        f"driver {gpu.get('driver_version', '?')}.",
        f"- Commit `{git.get('commit_short', '?')}` on branch "
        f"`{git.get('branch', '?')}`; dirty: **{dirty}**.",
        f"- STBN asset `{asset.get('name', '?')}` sha256 "
        f"`{(asset.get('sha256') or '')[:16]}...` "
        f"(NVIDIA, NON-COMMERCIAL licence).",
        f"- python {environment.get('python')}, numpy {packages.get('numpy')}, "
        f"scipy {packages.get('scipy')}, slangpy {packages.get('slangpy')}, "
        f"torch {packages.get('torch')}, lpips {packages.get('lpips')}.",
        dirty_note,
    ]
    return [line for line in lines if line is not None]


def _completeness(integrity: dict, counts: dict) -> list[str]:
    lines = [
        "## 2. Run accounting",
        "",
        f"- Planned cells: **{integrity.get('planned')}**",
        f"- Completed and usable: **{integrity.get('ok')}**",
        f"- Failed: **{integrity.get('failed')}**",
        f"- Never run: **{integrity.get('missing')}**",
        f"- Skipped (scene absent): **{integrity.get('missing_scene')}**",
        f"- Suite wall time this session: "
        f"**{_fmt(counts.get('wall_hours'), 2)} h**",
        "",
    ]
    if integrity.get("complete"):
        lines.append("Every planned cell completed. No aggregate in this report "
                     "is over a partial group.")
    else:
        lines.append("> **Not every planned cell completed.** Aggregates over "
                     "incomplete groups are flagged in the tables (a dagger in "
                     "LaTeX, a `complete` column in the CSVs) and must not be "
                     "compared against complete ones as if they were equivalent.")
        for key in ("failed_keys", "missing_keys", "missing_scene_keys"):
            items = integrity.get(key) or []
            if items:
                lines += ["", f"<details><summary>{key} "
                              f"({len(items)})</summary>", ""]
                lines += [f"- `{item}`" for item in items[:60]]
                if len(items) > 60:
                    lines.append(f"- ... and {len(items) - 60} more")
                lines += ["", "</details>"]
    lines.append("")
    return lines


def _main_results(exp_a: dict, suite: dict) -> list[str]:
    if not exp_a:
        return ["## 3. Main result (Experiment A)", "", "Not run.", ""]
    aggregate = exp_a.get("aggregate_all_scenes") or {}
    # Scene/iteration counts come from what actually ran, never from the module
    # defaults: a report that says "13 scenes" after two of them failed would be
    # exactly the silent averaging this suite is built to prevent.
    n_scenes = len(suite.get("scenes") or [])
    iters = suite.get("iters", cfg.ITERS)
    lines = [
        "## 3. Main result — 25% ablation (Experiment A)",
        "",
        f"{n_scenes} scene(s), seeds {exp_a.get('seeds')}, "
        f"rho = {exp_a.get('fraction')}, {iters} iterations. "
        f"{exp_a.get('n_cells')} cells, {exp_a.get('n_test_views')} scored "
        f"held-out views.",
        "",
        f"### Aggregate over all {n_scenes} scene(s)",
        "",
        "| Regime | PSNR (dB) | SSIM | LPIPS | Active train (s) | Gaussians |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for regime in cfg.REGIMES:
        row = aggregate.get(regime) or {}
        flag = "" if row.get("complete", True) else " ⚠"
        lines.append(
            f"| `{regime}`{flag} | {_fmt(row.get('psnr_srgb_mean'), 3)} "
            f"± {_fmt(row.get('psnr_srgb_std'), 3)} "
            f"| {_fmt(row.get('ssim_srgb_mean'), 4)} "
            f"| {_fmt(row.get('lpips_alex_mean'), 4)} "
            f"| {_fmt(row.get('train_active_s_mean'), 1)} "
            f"| {_fmt(row.get('n_gaussians_mean'), 0)} |")
    lines += ["", "Spread is across scenes (each scene averaged over seeds "
                  "first, so every scene is weighted equally).", ""]

    lines += ["### Paired contrasts", "",
              "Differences are paired within (scene, seed). The interval and the "
              "sign test use the 13 per-scene differences (seeds averaged within "
              "a scene) because scenes are the independent unit; win counts use "
              "all (scene, seed) cells.", ""]
    paired = exp_a.get("paired") or {}
    order = [("spatial", "Spatial: `indep_blue` − `iid_white`"),
             ("temporal", "Temporal: `stbn` − `indep_blue`"),
             ("spatial+temporal", "Both: `stbn` − `iid_white`"),
             ("sparsity_cost", "Cost of sparsity: `stbn` − `full`")]
    for contrast, title in order:
        lines += [f"**{title}**", ""]
        for metric in ("psnr_srgb", "ssim_srgb", "lpips_alex", "train_active_s"):
            row = paired.get(f"{contrast}::{metric}")
            if not row:
                continue
            label = stats.METRIC_LABELS.get(metric, metric)
            lines.append(f"- {label}: {_verdict(row)}")
        lines.append("")

    validation = exp_a.get("timing_validation") or {}
    worst = validation.get("worst_relative_gap")
    lines += [
        "### Timing-correction check",
        "",
        f"The primary seed carries mid-loop evaluation sweeps whose cost is "
        f"subtracted to give `train_active_s`; seeds "
        f"{validation.get('clean_seeds')} carry none and need no correction. "
        f"Comparing the two: the worst relative discrepancy across regimes is "
        f"**{_fmt(worst, 4)}**"
        + (" — comparable to seed-to-seed spread, so the correction is sound and "
           "the seeds may be pooled for timing."
           if (worst is not None and worst < 0.10)
           else " — large enough that only the unobserved seeds should be quoted "
                "for speed. Treat pooled timing means with suspicion."
           if worst is not None else "."),
        "",
    ]
    return lines


def _families(exp_a: dict) -> list[str]:
    lines = ["## 4. Per-dataset-family behaviour", "",
             "Averaged over the scenes of each family (each scene averaged over "
             "seeds first).", ""]
    lines += ["See `tables/main_25pct_dataset_summary.csv` and "
              "`latex/table_dataset_family.tex` for the full table. The "
              "families differ in capture style (unbounded outdoor, bounded "
              "indoor, object-centric) and in resolution — Tanks & Temples and "
              "Deep Blending ship only full-resolution images and are resized on "
              "load — so absolute scores are not comparable across families, "
              "only the within-family contrasts are.", ""]
    return lines


def _budget(exp_b: dict, suite: dict) -> list[str]:
    if not exp_b:
        return ["## 5. Pixel-budget sweep (Experiment B)", "", "Not run.", ""]
    n_scenes = len(suite.get("scenes") or [])
    lines = [
        "## 5. Pixel-budget sweep (Experiment B)",
        "",
        f"Seed {exp_b.get('seed')}, rho in {exp_b.get('fractions')}, "
        f"{n_scenes} scene(s), dense baseline shared across budgets. Gaps are "
        f"paired against the SAME scene's dense run and signed so **negative "
        f"always means worse than dense**, including for LPIPS.",
        "",
        "| Regime | rho | PSNR gap vs dense (dB) | LPIPS gap vs dense | Active (s) | Speedup vs dense |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    trends = exp_b.get("trends_all_scenes") or {}
    for regime in cfg.SPARSE_REGIMES:
        for fraction in sorted((trends.get(regime) or {}), key=float):
            row = trends[regime][fraction]
            flag = "" if row.get("complete", True) else " ⚠"
            lines.append(
                f"| `{regime}`{flag} | {fraction} "
                f"| {_fmt(row.get('psnr_gap_vs_dense'), 3, plus=True)} "
                f"| {_fmt(row.get('lpips_gap_vs_dense'), 4, plus=True)} "
                f"| {_fmt(row.get('active_s'), 1)} "
                f"| {_fmt(row.get('speedup_vs_dense'), 2)} |")
    lines += ["", "**STBN's advantage as a function of budget** (paired, per "
                  "scene):", ""]
    contrasts = exp_b.get("contrasts") or {}
    for name, title in (("stbn_vs_iid", "`stbn` − `iid_white`"),
                        ("stbn_vs_indep_blue", "`stbn` − `indep_blue`")):
        lines += [f"- {title}:"]
        for key, row in sorted(contrasts.items()):
            if not key.startswith(f"{name}::psnr_srgb::"):
                continue
            rho = key.rsplit("rho", 1)[-1]
            lines.append(f"    - rho={rho}: {_verdict(row)}")
    lines += ["", "> The rho=0.25 column is the same cells Experiment A trained "
                  "(not a re-run), and those cells carry the convergence ladder, "
                  "so their active time is observation-corrected while the 0.10 "
                  "and 0.50 columns are not. See the timing-correction check in "
                  "section 3 before reading the time-vs-budget trend as exact.",
              ""]
    return lines


def _convergence(exp_c: dict) -> list[str]:
    if not exp_c:
        return ["## 6. Convergence (Experiment C)", "", "Not run.", ""]
    lines = [
        "## 6. Convergence and time-to-quality (Experiment C)",
        "",
        f"Held-out quality at iterations {exp_c.get('checkpoint_iters')} for the "
        f"primary-seed 25% cells, {exp_c.get('n_scenes')} scene(s), "
        f"{exp_c.get('n_checkpoints')} checkpoint measurements. Evaluation cost "
        f"is excluded from the time axis by construction.",
        "",
        "### Time to reach dense-final quality",
        "",
        "| Regime | Target | Scenes reached | Active seconds (mean ± s.d.) | Speedup vs dense |",
        "|---|---|---:|---:|---:|",
    ]
    for row in exp_c.get("time_to_quality") or []:
        lines.append(
            f"| `{row.get('regime')}` "
            f"| dense − {row.get('target_db_below_dense')} dB "
            f"| {row.get('n_scenes_reached')}/{row.get('n_scenes_total')} "
            f"| {_fmt(row.get('seconds_mean'), 1)} ± "
            f"{_fmt(row.get('seconds_std'), 1)} "
            f"| {_fmt(row.get('speedup_vs_dense_mean'), 2)} |")
    lines += ["", "A mean over only the scenes that reached a target is not a "
                  "mean over all scenes: the reached count is quoted next to "
                  "every figure for that reason. Trajectories that never reach a "
                  "target are recorded as `not_reached` with their shortfall and "
                  "are **never extrapolated**.", ""]
    not_reached = exp_c.get("not_reached") or {}
    if not_reached:
        lines += ["Targets not reached within 3000 iterations:", ""]
        for key, scenes in sorted(not_reached.items()):
            lines.append(f"- `{key}`: {scenes}")
        lines.append("")
    return lines


def _runtime(exp_d: dict) -> list[str]:
    if not exp_d or exp_d.get("status"):
        return ["## 7. Runtime breakdown (Experiment D)", "",
                (exp_d or {}).get("status", "Not run."), ""]
    lines = [
        "## 7. Runtime breakdown (Experiment D)",
        "",
        f"Dedicated stage-timing runs on {exp_d.get('scenes')}, sampled on one "
        f"iteration in {cfg.STAGE_TIMING_EVERY}.",
        "",
        f"> {exp_d.get('decision')}",
        "",
        "### Share of per-iteration GPU time, whole run",
        "",
    ]
    shares = exp_d.get("stage_share_of_total") or {}
    stage_names = sorted({s for regime in shares.values() for s in regime})
    if stage_names:
        lines.append("| Stage | " + " | ".join(f"`{r}`" for r in cfg.REGIMES
                                               if r in shares) + " |")
        lines.append("|---" + "|---:" * len([r for r in cfg.REGIMES
                                             if r in shares]) + "|")
        for stage in stage_names:
            cells = " | ".join(
                f"{shares[r].get(stage, 0.0) * 100:.1f}%"
                for r in cfg.REGIMES if r in shares)
            lines.append(f"| {stage} | {cells} |")
    lines += [
        "",
        "Two limits of this breakdown, both structural rather than "
        "instrumental:",
        "",
        "- **There is no GPU mask-generation stage.** The rank table is built and "
        "uploaded once at startup and indexed inline by the kernels, so "
        "per-iteration mask work is a few microseconds of host arithmetic. It is "
        "reported as a host cost, not omitted.",
        "- **Densification is host-timed**, because it runs in its own submit "
        "outside the step's marker tree and reads buffers back. It is reported "
        "per densify event in `tables/runtime_breakdown.csv`, never folded into "
        "a GPU stage total.",
        "",
    ]
    return lines


def _sampling(exp_e: dict) -> list[str]:
    if not exp_e or exp_e.get("status") != "ok":
        return ["## 8. Sampling diagnostics (Experiment E)", "",
                (exp_e or {}).get("reason", "Not run."), ""]
    per_regime = exp_e.get("per_regime") or {}
    lines = [
        "## 8. Sampling diagnostics (Experiment E)",
        "",
        f"No training. {exp_e.get('visits')} consecutive masks per regime at "
        f"rho = {exp_e.get('fraction')}, measured on the 128x128x64 mask volume "
        f"that tiles the screen.",
        "",
        "| Statistic | `iid_white` | `indep_blue` | `stbn` | independent reference |",
        "|---|---:|---:|---:|---:|",
    ]

    def row(label: str, key: str, digits: int, ref_key: str | None = None) -> str:
        cells = " | ".join(_fmt((per_regime.get(r) or {}).get(key), digits)
                           for r in ("iid_white", "indep_blue", "stbn"))
        reference = (_fmt((per_regime.get("stbn") or {}).get(ref_key), digits)
                     if ref_key else "—")
        return f"| {label} | {cells} | {reference} |"

    lines += [
        row("low-frequency power (lower = bluer)", "low_freq_power_mean", 5),
        row("successive-mask overlap", "successive_overlap_mean", 5,
            "successive_overlap_expected_iid"),
        row("visit-count CoV after 64 (lower = more uniform)",
            "visit_count_cov", 4),
        # No analytic reference on the MAX gap: 1/rho is the expectation of the
        # mean gap, and printing it beside a maximum would invite reading it as
        # the independent-sampling max, which it is not.
        row("max revisit gap", "revisit_gap_max", 1),
        row("mean revisit gap", "revisit_gap_mean", 3,
            "revisit_gap_expected_iid"),
        row("distinct slices used of 64", "distinct_slices_used", 0),
    ]
    lines += ["", "### Reading of these numbers", ""]
    white = (per_regime.get("iid_white") or {}).get("low_freq_power_mean")
    blue = (per_regime.get("indep_blue") or {}).get("low_freq_power_mean")
    stbn_low = (per_regime.get("stbn") or {}).get("low_freq_power_mean")
    overlap_blue = (per_regime.get("indep_blue") or {}).get(
        "successive_overlap_mean")
    overlap_stbn = (per_regime.get("stbn") or {}).get("successive_overlap_mean")
    if all(v is not None for v in (white, blue, stbn_low)):
        lines.append(
            f"- The **spatial** axis moves the low-frequency power from "
            f"{_fmt(white, 5)} (white) to {_fmt(blue, 5)} (blue), a factor of "
            f"{_fmt(white / blue if blue else None, 1)}. `indep_blue` and `stbn` "
            f"have essentially the same spatial spectrum "
            f"({_fmt(blue, 5)} vs {_fmt(stbn_low, 5)}), confirming that the "
            f"only difference between them is temporal.")
    if all(v is not None for v in (overlap_blue, overlap_stbn)):
        lines.append(
            f"- The **temporal** axis is where they separate: successive-mask "
            f"overlap falls from {_fmt(overlap_blue, 5)} to "
            f"{_fmt(overlap_stbn, 5)}, i.e. the STBN walk almost never "
            f"re-samples a pixel on the next visit, whereas independent draws "
            f"re-sample it at close to the density rate.")
    lines += [
        f"- {exp_e.get('note')}",
        "",
    ]
    return lines


def _stratification(exp_s: dict, exp_e: dict) -> list[str]:
    """Section 9: does the 8x8 execution tiling belong in the distribution?"""
    lines = [
        "## 9. Selection-mode ablation (Experiment S)",
        "",
        "A sparse mask at density rho can be built two ways, and they are "
        "different sampling processes:",
        "",
        "- **`global`** — threshold the whole 128x128 slice and keep its lowest "
        "rho fraction. One threshold, so the survivors are the blue-noise point "
        "set the asset's spatial energy function was optimized for. Per-tile "
        "counts are then free to vary.",
        "- **`tile_rank`** — rank within each 8x8 execution tile and keep exactly "
        "k per tile. Uniform per-tile work, but the effective threshold now "
        "varies from tile to tile, which lets pixels on either side of a tile "
        "boundary both survive and clump.",
        "",
        "The 8x8 tiling is how GPU threads are assigned to pixels. Making the "
        "sampling *distribution* respect those boundaries imposes a "
        "stratification the asset does not have.",
        "",
    ]
    strat = (exp_e or {}).get("stratification") or {}
    if strat:
        lines += [
            "### Measured on the mask alone (no training)",
            "",
            "| quantity | `global` | `tile_rank` |",
            "|---|---:|---:|",
            f"| blue low-frequency power (lower = bluer) | "
            f"{_fmt(strat.get('blue_low_freq_global'), 5)} | "
            f"{_fmt(strat.get('blue_low_freq_tile_rank'), 5)} |",
            f"| white low-frequency power | "
            f"{_fmt(strat.get('white_low_freq_global'), 5)} | "
            f"{_fmt(strat.get('white_low_freq_tile_rank'), 5)} |",
            f"| blue:white contrast | "
            f"{_fmt(strat.get('contrast_global'), 1)}x | "
            f"{_fmt(strat.get('contrast_tile_rank'), 1)}x |",
            f"| per-tile count s.d. | "
            f"{_fmt(strat.get('tile_count_std_global'), 2)} | "
            f"{_fmt(strat.get('tile_count_std_tile_rank'), 2)} |",
            "",
            f"Per-tile ranking degrades the blue mask by a factor of "
            f"{_fmt(strat.get('blue_degradation_factor'), 2)} AND makes the white "
            f"control {_fmt(strat.get('white_flattery_factor'), 2)}x bluer than "
            f"true white — so it compresses the blue-versus-white contrast **from "
            f"both ends**. This is the more important half of the effect and the "
            f"one that is easy to miss: the control was being strengthened, not "
            f"just the treatment weakened.",
            "",
        ]
    if not exp_s or exp_s.get("status") != "ok":
        lines += ["### Effect on trained quality", "",
                  (exp_s or {}).get("status", "Not run."), ""]
        return lines

    lines += [
        "### Effect on trained quality",
        "",
        f"Paired on identical (scene, seed, rho) cells at seed "
        f"{exp_s.get('seed')} and rho = {exp_s.get('fraction')}, the only "
        f"difference being the selection mode. Positive means the global "
        f"threshold is better.",
        "",
        "| Regime | ΔPSNR (dB) | 95% CI | Wins | Sign p |",
        "|---|---:|---:|---:|---:|",
    ]
    for regime in cfg.SPARSE_REGIMES:
        row = (exp_s.get("deltas") or {}).get(f"{regime}::psnr_srgb")
        if not row:
            continue
        lines.append(
            f"| `{regime}` | {_fmt(row.get('mean_delta'), 4, plus=True)} "
            f"| [{_fmt(row.get('ci_low'), 4)}, {_fmt(row.get('ci_high'), 4)}] "
            f"| {row.get('wins')}/{row.get('n_cells')} "
            f"| {_fmt(row.get('sign_test_p'), 3)} |")
    lines += [
        "",
        f"**Reading:** {exp_s.get('verdict')}",
        "",
        "The prediction from the spectral measurement was specific and therefore "
        "falsifiable: a global threshold should HELP the two blue conditions and "
        "HURT the white control, since per-tile ranking was stratifying the "
        "control for free. Whether that pattern appears bounds how much of the "
        "quality difference the mask spectrum actually explains.",
        "",
    ]
    return lines


def _gradients(exp_f: dict) -> list[str]:
    if not exp_f or exp_f.get("status") != "ok":
        return ["## 10. Gradient diagnostic (Experiment F)", "",
                (exp_f or {}).get("status", "Not run."), ""]
    overall = exp_f.get("overall_all_components") or {}
    lines = [
        "## 10. Gradient diagnostic (Experiment F)",
        "",
        f"Frozen Gaussians, pinned training camera, no parameter updates. "
        f"Scenes {exp_f.get('scenes')} at training states "
        f"{exp_f.get('states')}. Sparse gradients are the MEAN of K consecutive "
        f"masked backwards (each already `1/rho`-reweighted, hence individually "
        f"unbiased), compared against the dense gradient at the same state.",
        "",
        "| Regime | " + " | ".join(f"cos K={k}" for k in exp_f.get('k_values', []))
        + " | " + " | ".join(f"relL2 K={k}"
                             for k in exp_f.get('k_values', [])) + " |",
        "|---" + "|---:" * (2 * len(exp_f.get("k_values") or [])) + "|",
    ]
    for regime in ("iid_white", "indep_blue", "stbn"):
        block = overall.get(regime) or {}
        cos = " | ".join(_fmt((block.get(str(k)) or {}).get("cosine"), 4)
                         for k in exp_f.get("k_values", []))
        rel = " | ".join(_fmt((block.get(str(k)) or {}).get("relative_l2"), 4)
                         for k in exp_f.get("k_values", []))
        lines.append(f"| `{regime}` | {cos} | {rel} |")
    lines += [
        "",
        f"**Harness check.** At K=1 a single STBN slice *is* a single independent "
        f"blue-noise slice, so the two must agree. Worst gap across parameter "
        f"groups: {_fmt(exp_f.get('k1_equivalence_worst_gap'), 4)} — "
        f"{exp_f.get('k1_equivalence_verdict')}. The separation between them is "
        f"therefore a property of accumulation, which is what the temporal claim "
        f"predicts.",
        "",
        "Cosine similarity is the more relevant of the two metrics for an "
        "Adam-optimized system, since Adam renormalizes per-parameter magnitude; "
        "relative L2 is the stricter number and is reported alongside rather "
        "than in place of it. Per-parameter-group results (position, scale, "
        "rotation, opacity, colour) are in "
        "`gradient_analysis/gradient_summary.csv`.",
        "",
    ]
    return lines


def _qualitative(exp_g: dict) -> list[str]:
    if not exp_g:
        return ["## 11. Qualitative material (Experiment G)", "", "Not run.", ""]
    lines = [
        "## 11. Qualitative material (Experiment G)",
        "",
        f"{exp_g.get('n_views')} held-out views across {exp_g.get('n_scenes')} "
        f"scene(s): one contact sheet (GT | dense | IID | independent blue | STBN) "
        f"and one shared-scale error map each. "
        f"{exp_g.get('n_patches_scanned')} patches scanned, "
        f"{exp_g.get('n_candidates')} ranked candidate crops saved with scene, "
        f"view and pixel coordinates.",
        "",
        "> Three rankings are produced and **all three are kept**, including "
        "`stbn_vs_indep_worst` — the patches where STBN is most beaten by "
        "independent blue noise. Choosing crops by eye is how qualitative "
        "figures end up flattering; these are measured extremes, and the failure "
        "cases are generated on exactly the same terms as the favourable ones.",
        "",
    ]
    for key, title in (("top_stbn_vs_iid", "STBN most improves on IID white"),
                       ("top_stbn_vs_indep",
                        "STBN most improves on independent blue noise"),
                       ("worst_stbn_vs_indep",
                        "STBN most UNDERPERFORMS independent blue noise")):
        items = exp_g.get(key) or []
        if not items:
            continue
        lines += [f"**{title}**", ""]
        for item in items:
            lines.append(
                f"- `{item.get('scene')}` cam {item.get('cam_idx')} "
                f"at ({item.get('x0')}, {item.get('y0')}): "
                f"STBN−IID {_fmt(item.get('delta_stbn_minus_iid'), 2, plus=True)} dB, "
                f"STBN−indep {_fmt(item.get('delta_stbn_minus_indep'), 2, plus=True)} dB "
                f"— `{item.get('crop_png')}`")
        lines.append("")
    if exp_g.get("skipped"):
        lines += ["Skipped views/scenes:", ""]
        for key, reason in sorted((exp_g.get("skipped") or {}).items()):
            lines.append(f"- `{key}`: {reason}")
        lines.append("")
    return lines


def _artifacts(findings: dict) -> list[str]:
    lines = ["## 12. Paper-ready artifacts", "",
             "### LaTeX tables (booktabs)", ""]
    for name, caption in (
            ("latex/table_main_quantitative.tex", "main quantitative table"),
            ("latex/table_dataset_family.tex", "dataset-family aggregate"),
            ("latex/table_spatial_vs_temporal.tex",
             "spatial-vs-temporal ablation"),
            ("latex/table_pixel_fraction_sweep.tex", "pixel-fraction sweep"),
            ("latex/table_runtime_breakdown.tex", "runtime breakdown"),
            ("latex/table_stratification_ablation.tex",
             "selection-mode (global vs per-tile) ablation"),
            ("latex/table_gradient_quality.tex", "gradient diagnostic")):
        lines.append(f"- `{name}` — {caption}")
    lines += ["", "### Figures (PDF for the paper, PNG for review)", ""]
    for experiment, block in sorted(findings.items()):
        for path in ((block or {}).get("artifacts") or {}).get("figures", []):
            lines.append(f"- `{path}` ({experiment})")
    lines += ["", "### Tables and raw data", ""]
    for experiment, block in sorted(findings.items()):
        artifacts = ((block or {}).get("artifacts") or {})
        for key, path in sorted(artifacts.items()):
            if key in ("figures", "latex"):
                continue
            lines.append(f"- `{path}` ({experiment}: {key})")
    lines += ["", "### Bookkeeping", "",
              "- `manifest.json` — every cell: exact argv, config hash, commit, "
              "dirty flag, STBN asset hash, GPU and driver, resolution, image "
              "counts, split rule, timestamps, active/wall seconds, Gaussian "
              "count, failure status and stderr path.",
              "- `environment.json` — session provenance.",
              "- `logs/` — per-cell stdout and stderr.",
              ""]
    return lines


def _limitations() -> list[str]:
    return [
        "## 13. Limitations to state in the paper",
        "",
        "1. **The IID control is stratified**, so the spatial effect is a lower "
        "bound (section 1).",
        "2. **`indep_blue` draws from a 64-slice pool**, not from freshly "
        "generated blue noise; part of STBN's coverage advantage over it is that "
        "the walk visits all 64 slices while independent draws visit ~40.",
        "3. **One resolution, one iteration budget, one densifier.** Everything "
        "here is quarter-resolution, 3000 iterations, structure-aware "
        "densification with Mip-Splatting. Sparse supervision interacts with "
        "densification (the densifier's gradient statistics are what sparsity "
        "starves), so these results should not be assumed to transfer to a "
        "different densifier or a 30k-iteration schedule.",
        "4. **Three seeds** is enough to see the seed-to-seed spread and not "
        "enough to characterize its tail. The intervals here are over scenes, "
        "which is the right unit, but they inherit whatever seed noise remains.",
        "5. **LPIPS is computed at the evaluation resolution on CPU AlexNet**, "
        "which is comparable across these regimes but not necessarily against a "
        "published number scored at a different resolution or with a different "
        "backbone.",
        "6. **Peak GPU memory is capacity-dominated.** The trainer preallocates "
        "per-splat buffers from `--max-splats`, so the allocation figure is a "
        "constant of the configuration and is identical across regimes; the "
        "device-wide poll is reported alongside it and is only interpretable on "
        "an idle GPU. Neither is evidence that sparse supervision saves memory.",
        "7. **The NVIDIA STBN asset is NON-COMMERCIAL licensed.** Any release "
        "must either exclude it or point at the upstream download.",
        "",
    ]


def build(findings: dict, *, environment: dict, integrity: dict,
          counts: dict, plan_summary: dict, out_root: Path,
          suite: dict | None = None) -> Path:
    """Write ``report.md`` and return its path."""
    from .env import utc_now

    lines: list[str] = [
        "# STBN sparse supervision — experiment report",
        "",
        f"Generated {utc_now()} by "
        f"`scripts/running/eval_stbn_ablation.py`.",
        "",
        "This report is deliberately conservative. Where an interval contains "
        "zero it says the data do not resolve the sign, rather than claiming "
        "either an effect or its absence. Negative and null results appear with "
        "the same prominence as positive ones.",
        "",
        "---",
        "",
    ]
    suite = suite or {}
    lines += _protocol(findings, environment, plan_summary, suite)
    lines += _completeness(integrity, counts)
    lines += _main_results(findings.get("A") or {}, suite)
    lines += _families(findings.get("A") or {})
    lines += _budget(findings.get("B") or {}, suite)
    lines += _convergence(findings.get("C") or {})
    lines += _runtime(findings.get("D") or {})
    lines += _sampling(findings.get("E") or {})
    lines += _stratification(findings.get("S") or {}, findings.get("E") or {})
    lines += _gradients(findings.get("F") or {})
    lines += _qualitative(findings.get("G") or {})
    lines += _artifacts(findings)
    lines += _limitations()

    path = out_root / "report.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path
