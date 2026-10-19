# Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

"""Is ANY cheap screen-space statistic a usable importance proxy?

The pilot's negative result has one obvious escape: `||h_i||` is a bad proxy for
`|J_i^T h_i|`, so try a better one. That is a hypothesis, and testing it by
implementing proxies one at a time and training five scenes each is expensive. This
measures the thing directly instead.

THE MEASUREMENT. Partition a frozen render into blocks. For each block b, run a
backward supervised ONLY on that block and take the norm of the resulting Gaussian
gradient:

    c_b = || sum_{i in b} J_i^T h_i ||

That is the block-level version of exactly the quantity importance sampling wants
to be proportional to and cannot afford to know. It is measurable because a block
is one mask, not one pixel: 96 blocks is 96 backwards, where 96 pixels-at-a-time
would be a million.

Then correlate every candidate proxy — computed from the render, the target and the
dense `h` alone — against `c_b`. A proxy that cannot predict where the Gaussian
gradient comes from cannot allocate toward it, whatever the sampler does with it.
So this bounds the whole approach rather than one instance of it:

  * high correlation for some proxy -> the pilot failed on its choice of proxy, and
    that proxy is worth implementing;
  * low correlation for ALL of them -> screen-space allocation is dead regardless of
    proxy, and the negative result is about the method, not the parameterization.

WHY A CONSTANT WEIGHT IS FINE. The loss applies `stbnWeight(budget)` =
TILE_AREA/budget to every supervised pixel. Passing the same budget for every block
makes that a single global factor, which cancels in a correlation — so no
reweighting machinery is needed and the measurement stays a pure comparison of
block contributions.

MEASUREMENT ONLY. No optimizer step is taken and the state is verified unchanged by
checksum, exactly as the gradient study does it.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

_SCRIPTS = Path(__file__).resolve().parents[2]
for _sub in ("running", "debug", "plots"):
    _p = str(_SCRIPTS / _sub)
    if _p not in sys.path:
        sys.path.append(_p)

from paper_gradient_study import (SCALAR_GROUPS, StagedGradReader,  # noqa: E402
                                 state_fingerprint)

from trainer.backend.render import stbn as stbn_module            # noqa: E402
from trainer.config.config import (DensificationProperties,       # noqa: E402
                                   IterationContext,
                                   IterationContextualVariables, StbnVariables)
from trainer.contracts.codegen import (TILE_AREA, TILE_HEIGHT,    # noqa: E402
                                       TILE_WIDTH)
from trainer.frontend.app import TrainingApplication              # noqa: E402
from trainer.frontend.cli import build_parser                     # noqa: E402

# Blocks along each axis. 12x8 = 96 blocks is the compromise: fine enough that a
# block is smaller than the sigma=8 smoothing scale the real proxy operates at
# (a 1236x821 frame gives ~103x103 px blocks — coarser than sigma, but any finer
# and the per-block gradient is too noisy to correlate against), coarse enough
# that 96 backwards per camera is affordable.
BLOCKS_X = 12
BLOCKS_Y = 8

# The budget uniform passed for every block. Any value in [1, TILE_AREA) works and
# only scales every c_b by the same constant; 16 matches the study's rho=0.25 so
# the shader takes the same code path it does everywhere else.
BLOCK_BUDGET = 16

GROUPS = ("position", "rotation", "opacity", "color_sh0")


# ---------------------------------------------------------------------------
# Candidate proxies
# ---------------------------------------------------------------------------

def _grad_magnitude(field: np.ndarray) -> np.ndarray:
    """``||∇f||`` per pixel by central differences, edge-replicated.

    Edge-replicated rather than zero-padded: a zero pad would manufacture a large
    spatial gradient at the frame border and put spurious importance there.
    """
    dy = np.gradient(field, axis=0)
    dx = np.gradient(field, axis=1)
    return np.sqrt(dx * dx + dy * dy)


def proxies(h: np.ndarray, rendered: np.ndarray,
            target: np.ndarray) -> dict[str, np.ndarray]:
    """Per-PIXEL candidate importance fields, all computable before the backward.

    Each is a statistic of the render, the target and the dense screen gradient —
    i.e. of things the loss pass already produced. Anything needing the Jacobian
    would defeat the purpose.
    """
    h_rgb = h[..., :3]
    magnitude = np.sqrt((h_rgb ** 2).sum(axis=-1))
    error = rendered[..., :3] - target[..., :3]
    error_magnitude = np.abs(error).sum(axis=-1)
    luminance = rendered[..., :3].mean(axis=-1)
    # Signed channels, kept separate so the block aggregate can be a COHERENT
    # (vector) sum rather than an absolute one: cancellation within a block is
    # exactly what an absolute sum cannot see, and J_i^T h_i sums coherently.
    return {
        # The shipped proxy.
        "abs_h": magnitude,
        "signed_h_r": h_rgb[..., 0],
        "signed_h_g": h_rgb[..., 1],
        "signed_h_b": h_rgb[..., 2],
        # Squared: weights the tail harder, which is what a variance-optimal
        # allocation for a SUM of squares would want.
        "h_squared": magnitude ** 2,
        # The proposed replacement: the residual's spatial VARIATION, which is what
        # position and rotation gradients integrate across a splat footprint.
        "grad_h": _grad_magnitude(magnitude),
        # Same idea one step earlier, on the raw residual rather than its gradient.
        "grad_error": _grad_magnitude(error_magnitude),
        # The error-magnitude fallback the design allows when the screen gradient is
        # unavailable. Measured here so the fallback is compared rather than assumed
        # worse.
        "abs_error": error_magnitude,
        # Image detail alone — needs no loss pass at all, so if it correlated it
        # would be the cheapest possible proxy.
        "grad_render": _grad_magnitude(luminance),
        # A flat field: the control. Its correlation with c_b measures how much of
        # the variation is just "blocks contain different numbers of live pixels".
        "uniform": np.ones_like(magnitude),
    }


def block_slices(height: int, width: int) -> list[tuple[slice, slice, tuple]]:
    """Tile-ALIGNED block partition.

    Aligned to TILE_WIDTH/TILE_HEIGHT because the packed mask is per tile: a block
    boundary inside a tile would make that tile partially selected, and the
    per-tile count would then no longer be 0 or TILE_AREA. Harmless for the
    backward, but it would make "supervised only on block b" untrue at the seams.
    """
    tiles_x = -(-width // TILE_WIDTH)
    tiles_y = -(-height // TILE_HEIGHT)
    xs = np.linspace(0, tiles_x, BLOCKS_X + 1).astype(int)
    ys = np.linspace(0, tiles_y, BLOCKS_Y + 1).astype(int)
    out = []
    for by in range(BLOCKS_Y):
        for bx in range(BLOCKS_X):
            if xs[bx + 1] <= xs[bx] or ys[by + 1] <= ys[by]:
                continue
            out.append((slice(ys[by] * TILE_HEIGHT,
                              min(ys[by + 1] * TILE_HEIGHT, height)),
                        slice(xs[bx] * TILE_WIDTH,
                              min(xs[bx + 1] * TILE_WIDTH, width)),
                        (bx, by)))
    return out


# ---------------------------------------------------------------------------
# Screen-sized single-block masks
# ---------------------------------------------------------------------------

def block_mask_spec(width: int, height: int) -> stbn_module.StbnMaskSpec:
    """A one-slice spec at the TILE-PADDED frame size.

    ``source="white"`` because the spec's source is irrelevant here — the packed
    table is supplied directly by :func:`stbn_module.pack_selection` and the volume
    is never read. The nvidia source would refuse these dimensions (it is pinned to
    the published 128x128x64), which is the check working as intended.
    """
    padded_w = -(-width // TILE_WIDTH) * TILE_WIDTH
    padded_h = -(-height // TILE_HEIGHT) * TILE_HEIGHT
    return stbn_module.StbnMaskSpec(
        width=padded_w, height=padded_h, slices=1,
        source=stbn_module.SOURCE_WHITE, select_mode=stbn_module.SELECT_GLOBAL,
        fraction=BLOCK_BUDGET / TILE_AREA)


def upload_block_masks(device, spec, blocks, height, width) -> list:
    """One uploaded table per block. All resident at once, so a block measurement
    is a pointer swap rather than an upload."""
    tables = []
    for rows, cols, _ in blocks:
        selected = np.zeros((1, spec.height, spec.width), dtype=bool)
        selected[0, rows, cols] = True
        packed = stbn_module.pack_selection(selected, spec)
        tables.append(stbn_module.upload_packed_table(device, packed, spec))
    return tables


# ---------------------------------------------------------------------------
# Frozen-state passes
# ---------------------------------------------------------------------------

def _context(pipeline, sh_degree, hardness, variables):
    contextual = IterationContextualVariables(
        hardness_value=hardness, sh_degree=sh_degree, contextual_stbn=variables,
        densification_variables=DensificationProperties(
            structure_tensor_texture=None, structure_kwargs={}))
    return IterationContext(step_number=pipeline.iteration + 1, camera_index=0,
                            camera_entry=None, learning_rates=None, events=None,
                            iteration_contextual_variables=contextual)


def loss_and_backward(pipeline, camera, *, sh_degree, hardness, variables):
    """Forward + loss + backward with the grads cleared first.

    The forward is re-run every time. Not redundant: the backward does not leave
    the forward's raster checkpoints reusable, so only the first backward after a
    forward is valid (documented at length in paper_gradient_study).
    """
    context = _context(pipeline, sh_degree, hardness, variables)
    inner = context.iteration_contextual_variables
    with pipeline.recording("proxy_probe"):
        pipeline._clear_grads()
        pipeline._run_forward(camera, inner)
        pipeline._run_loss(inner)
        pipeline._run_backward(context)


def group_norms(grads: dict[str, np.ndarray]) -> dict[str, float]:
    out = {}
    for group, components in SCALAR_GROUPS.items():
        total = 0.0
        for name in components:
            values = grads[name].astype(np.float64, copy=False)
            total += float(np.dot(values, values))
        out[group] = float(np.sqrt(max(total, 0.0)))
    out["reported"] = float(np.sqrt(sum(out[g] ** 2 for g in GROUPS)))
    return out


# ---------------------------------------------------------------------------
# Correlation
# ---------------------------------------------------------------------------

def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Rank correlation, computed by hand to avoid a scipy dependency.

    Rank rather than Pearson is the primary statistic: an allocation only needs the
    proxy to ORDER blocks correctly, not to be linear in the contribution.
    """
    def rank(values: np.ndarray) -> np.ndarray:
        order = np.argsort(values, kind="stable")
        ranks = np.empty(values.size, dtype=np.float64)
        ranks[order] = np.arange(values.size, dtype=np.float64)
        # Average ties, so a constant field gives a defined (zero-variance) rank.
        _, inverse, counts = np.unique(values, return_inverse=True,
                                       return_counts=True)
        sums = np.zeros(counts.size)
        np.add.at(sums, inverse, ranks)
        return (sums / counts)[inverse]

    ra, rb = rank(np.asarray(a, float)), rank(np.asarray(b, float))
    if ra.std() < 1e-12 or rb.std() < 1e-12:
        return 0.0
    return float(np.corrcoef(ra, rb)[0, 1])


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.std() < 1e-12 or b.std() < 1e-12:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def allocation_gain(contribution: np.ndarray, proxy: np.ndarray) -> float:
    """Fraction of the ORACLE allocation gain a proxy would capture.

    For a fixed budget the variance of an inverse-probability-weighted sum is
    minimized by ``p_b ∝ c_b`` (proportional allocation), and the resulting
    variance is ``(sum c_b)^2 / B`` against ``N/B * sum c_b^2`` for uniform. So the
    oracle's reduction factor is ``(sum c)^2 / (N sum c^2)``, and a proxy achieves
    whatever ``p ∝ proxy`` gives. Reported as the fraction of the oracle's gain
    recovered, in [0, 1] — 0 means the proxy is no better than uniform, 1 means it
    is as good as knowing c_b exactly.
    """
    c = np.asarray(contribution, float)
    w = np.asarray(proxy, float)
    n = c.size
    if n == 0 or c.sum() <= 0:
        return 0.0
    uniform = n * float((c ** 2).sum())
    oracle = float(c.sum()) ** 2
    if w.sum() <= 0:
        return 0.0
    w = w / w.sum()
    # Variance of the IPW estimator with p_b ∝ w_b: sum_b c_b^2 / w_b (up to the
    # shared budget factor), which is the Cauchy-Schwarz bound the oracle attains.
    achieved = float(((c ** 2) / np.maximum(w, 1e-30)).sum())
    if uniform <= oracle:
        return 0.0
    return float(np.clip((uniform - achieved) / (uniform - oracle), 0.0, 1.0))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = build_parser()
    parser.add_argument("--state-iters", type=int, nargs="*", default=[500])
    parser.add_argument("--cameras", type=int, default=4)
    parser.add_argument("--out", type=str, required=True)
    args = parser.parse_args()

    for name, value in (("no_viewer", True), ("profile_run", False),
                        ("capture_external", False), ("preview_every", 0),
                        ("log_every", 0), ("eval_every", 0), ("eval_iters", None),
                        ("psnr_curve", False), ("render_all_every", 0),
                        ("final_test_psnr", False),
                        ("compute_final_metrics", False), ("run_report", False),
                        ("save_test_renders", False), ("stage_timing_every", 0),
                        ("write_ground_truths", False), ("stbn_adaptive", False)):
        setattr(args, name, value)
    args.iters = max(args.iters, max(args.state_iters))

    app = TrainingApplication(args)
    app.prepare_training_context()
    loop = app.build_training_loop()
    loop.stbn_density = None                    # advance DENSELY
    pipeline = app._pipeline
    scene = app._sceneContext
    width, height = scene.width, scene.height

    blocks = block_slices(height, width)
    spec = block_mask_spec(width, height)
    tables = upload_block_masks(pipeline.device, spec, blocks, height, width)
    dense_table = pipeline._stbn
    scene_name = Path(args.scene).name
    print(f"[proxy] {scene_name} {width}x{height}: {len(blocks)} blocks "
          f"({BLOCKS_X}x{BLOCKS_Y}), states={args.state_iters}", flush=True)

    train_indices = list(scene.train_indices)
    step = max(1, len(train_indices) // max(args.cameras, 1))
    cameras = train_indices[::step][:args.cameras]

    rows: list[dict] = []
    for state_iter in sorted(args.state_iters):
        context = loop.advance_to_capture(state_iter)
        sh_degree = context.iteration_contextual_variables.sh_degree
        hardness = context.iteration_contextual_variables.hardness_value
        before = state_fingerprint(pipeline)
        count = int(before.get("n_gaussians") or 0)
        reader = StagedGradReader(pipeline, count)
        started = time.perf_counter()

        for cam_idx in cameras:
            pipeline.set_target_texture(app.target_textures[cam_idx])
            camera = scene.colmapScene.images[cam_idx].camera

            # Dense pass: h for the whole image, and the render, for the proxies.
            pipeline._stbn = dense_table
            loss_and_backward(pipeline, camera, sh_degree=sh_degree,
                              hardness=hardness, variables=None)
            pipeline.runtime.wait_for_idle()
            h = np.asarray(pipeline.buffers.dL_dC_texture.to_numpy(),
                           dtype=np.float64)
            rendered = np.asarray(pipeline.buffers.output_image.to_numpy(),
                                  dtype=np.float64)
            target = np.asarray(pipeline._active_target.to_numpy())
            if target.dtype == np.uint8:
                target = target.astype(np.float64) / 255.0
            fields = proxies(h, rendered, np.asarray(target, dtype=np.float64))

            for index, (block_rows, block_cols, coords) in enumerate(blocks):
                pipeline._stbn = tables[index]
                loss_and_backward(
                    pipeline, camera, sh_degree=sh_degree, hardness=hardness,
                    variables=StbnVariables(slice_index=0,
                                            pixels_per_tile=BLOCK_BUDGET))
                norms = group_norms(reader.read_all())
                row = {"scene": scene_name, "checkpoint": state_iter,
                       "camera_id": int(cam_idx), "block_x": coords[0],
                       "block_y": coords[1], "n_gaussians": count,
                       "n_pixels": int((block_rows.stop - block_rows.start)
                                       * (block_cols.stop - block_cols.start))}
                row.update({f"contribution_{k}": v for k, v in norms.items()})
                sums = {name: float(field[block_rows, block_cols].sum())
                        for name, field in fields.items()}
                # The coherent proxy: || sum_{i in b} h_i ||, from the signed
                # channel sums. Formed here rather than as a per-pixel field
                # because it is only defined for a BLOCK.
                sums["coherent_h"] = float(np.sqrt(
                    sums.pop("signed_h_r") ** 2 + sums.pop("signed_h_g") ** 2
                    + sums.pop("signed_h_b") ** 2))
                row.update({f"proxy_{name}": v for name, v in sums.items()})
                rows.append(row)
            print(f"[proxy]   {scene_name}@{state_iter} cam{cam_idx}: "
                  f"{len(blocks)} blocks in "
                  f"{time.perf_counter() - started:.0f}s", flush=True)

        after = state_fingerprint(pipeline)
        unchanged = (before.get("gaussians_checksum")
                     == after.get("gaussians_checksum"))
        print(f"[proxy] {scene_name}@{state_iter}: n_g={count}, "
              f"state_unchanged={unchanged}", flush=True)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "scene": scene_name, "blocks_x": BLOCKS_X, "blocks_y": BLOCKS_Y,
        "block_budget": BLOCK_BUDGET, "groups": list(GROUPS),
        "cameras": [int(c) for c in cameras],
        "state_iters": sorted(args.state_iters),
        "measurement": (
            "c_b = ||sum_{i in b} J_i^T h_i||, measured by a backward supervised "
            "only on block b. The loss's constant TILE_AREA/budget weight scales "
            "every block identically and so cancels in every correlation."),
        "rows": rows,
    }, indent=2))
    print(f"[proxy] wrote {out_path} ({len(rows)} block observations)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

PROXY_ORDER = ("abs_h", "h_squared", "grad_h", "abs_error", "grad_error",
               "coherent_h", "grad_render", "uniform")
PROXY_LABELS = {
    "abs_h": "||h|| (shipped)",
    "h_squared": "||h||^2",
    "grad_h": "||grad h||",
    "abs_error": "|C-T| (design fallback)",
    "grad_error": "||grad |C-T|||",
    "coherent_h": "||sum h|| (coherent)",
    "grad_render": "||grad render|| (no loss pass)",
    "uniform": "uniform (control)",
}
SUMMARY_COLUMNS = ("scene", "proxy", "n_cells", "rank_correlation",
                   "rank_correlation_sd", "allocation_gain",
                   "variance_vs_uniform", "equal_variance_rho")


def summarize(paths, scenes, rho: float = 0.25) -> dict:
    """Correlate every proxy against the measured block contribution, per scene.

    Reported per scene and never pooled. A cross-scene mean would hide the finding
    that matters most here: the ranking of proxies is NOT stable between an outdoor
    and an indoor scene, so no single global choice is supported.

    ``equal_variance_rho`` is the decision-relevant number. Uniform sampling's
    estimator variance scales as 1/rho, so a proxy that cuts variance to ``f`` times
    uniform reaches the SAME variance at ``f * rho`` — i.e. it says what budget the
    allocation would buy back. The ORACLE row is the ceiling any proxy could reach.
    """
    from paper_suite import tables as _tables

    out_dir = paths["gradient_validation"]
    rows: list[dict] = []
    per_scene: dict[str, dict] = {}
    for scene in scenes:
        path = out_dir / f"proxy_{scene}_full.json"
        if not path.is_file():
            continue
        payload = json.loads(path.read_text())
        blocks = payload["rows"]
        cells = sorted({(r["checkpoint"], r["camera_id"]) for r in blocks})
        entry: dict[str, dict] = {}
        for proxy in PROXY_ORDER + ("ORACLE",):
            corr, gain, factor = [], [], []
            for checkpoint, camera in cells:
                sub = [r for r in blocks if r["checkpoint"] == checkpoint
                       and r["camera_id"] == camera]
                c = np.asarray([r["contribution_reported"] for r in sub], float)
                oracle_f = ((c.sum() ** 2) / (c.size * (c ** 2).sum())
                            if c.sum() > 0 else 1.0)
                if proxy == "ORACLE":
                    corr.append(1.0)
                    gain.append(1.0)
                    factor.append(oracle_f)
                    continue
                p = np.asarray([r[f"proxy_{proxy}"] for r in sub], float)
                corr.append(_spearman(p, c))
                g = allocation_gain(c, p)
                gain.append(g)
                factor.append(1.0 - g * (1.0 - oracle_f))
            record = {
                "scene": scene, "proxy": proxy, "n_cells": len(cells),
                "rank_correlation": float(np.mean(corr)),
                "rank_correlation_sd": (float(np.std(corr, ddof=1))
                                        if len(corr) > 1 else 0.0),
                "allocation_gain": float(np.mean(gain)),
                "variance_vs_uniform": float(np.median(factor)),
                "equal_variance_rho": float(np.median(factor)) * rho,
            }
            rows.append(record)
            entry[proxy] = record
        per_scene[scene] = entry
    if not rows:
        return {"status": "missing", "reason": "no proxy_*_full.json payloads"}
    _tables.write_csv(out_dir / "proxy_correlation.csv", rows, SUMMARY_COLUMNS)
    record = {
        "status": "ok", "rho": rho, "scenes": sorted(per_scene),
        "blocks": f"{BLOCKS_X}x{BLOCKS_Y}",
        "measurement": (
            "c_b = ||sum_{i in b} J_i^T h_i||, measured by a backward supervised "
            "only on block b. Correlations are WITHIN an image; pooling across "
            "cameras would mostly measure that different views have different "
            "total gradient, which no per-image allocation can exploit."),
        "equal_variance_rho_note": (
            "uniform variance scales as 1/rho, so a proxy cutting variance to f x "
            "uniform matches it at f*rho. This is a GRADIENT-VARIANCE bound, not a "
            "PSNR prediction: the pilot moved PSNR only ~0.09 dB for a 4.8% change "
            "in gradient error, so treat it as an upper bound on what would show "
            "up in dB."),
        "per_scene": per_scene,
        "outputs": {"csv": "gradient_validation/proxy_correlation.csv"},
    }
    (out_dir / "proxy_probe.json").write_text(json.dumps(record, indent=2))
    return record
