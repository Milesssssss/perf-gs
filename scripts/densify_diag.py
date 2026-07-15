"""Densification diagnostics for the SAD-GS path.

A single entry point, :func:`capture_densify_tick`, reads back the GPU state
AROUND one densification tick and returns a JSON-serializable dict of metrics
that answer: is densification runaway / untargeted, and is growth spatially
concentrated (starving optimization in dense regions)?

Deliberately readback-heavy — it calls wait_for_idle and pulls whole buffers to
CPU — so it is OPT-IN (``--densify-diag``) and only runs on the ~17 densify
ticks of a 3k-iter run, not the hot per-iter path.

Metrics per tick:
  pre_count / post_count / growth_factor  — the raw growth trajectory.
  action histogram (keep/split/clone/prune) over the PRE-densify slots.
  fanout: distribution of prod = n_x*n_y*n_z over tentative splits (are we
    getting 512-children blowups, or targeted 2-4x splits?).
  eta gate: fraction of visible gaussians that pass the split trigger, plus
    high_ratio / n_total percentiles (is the multiview gate actually gating,
    or does ~everything trigger?).
  spatial concentration: occupancy grid over gaussian means (Gini + top-cell
    share of the population) — directly tests "runaway growth in some areas".
"""

from __future__ import annotations

import numpy as np

ACTION_NAMES = ("keep", "split", "clone", "prune")


def _counter(pipeline) -> int:
    return int(pipeline.buffers.num_gaussians_counter.to_numpy().view(np.uint32)[0])


def _u32(buf, n: int) -> np.ndarray:
    return np.frombuffer(buf.to_numpy().tobytes(), dtype=np.uint32)[:n]


def _i64(buf, n: int) -> np.ndarray:
    return np.frombuffer(buf.to_numpy().tobytes(), dtype=np.int64)[:n]


def _gaussian_means(pipeline, n: int) -> np.ndarray:
    g = np.frombuffer(pipeline.buffers.gaussians.to_numpy().tobytes(),
                      dtype=np.float32).reshape(-1, 14)[:n]
    return g[:, 0:3]


def _gini(counts: np.ndarray) -> float:
    """Gini coefficient of a set of nonneg counts (0 = uniform, ->1 = all in
    one bin). Standard sorted-cumulative formula."""
    c = np.sort(counts.astype(np.float64))
    n = c.size
    if n == 0 or c.sum() == 0:
        return 0.0
    idx = np.arange(1, n + 1)
    return float((np.sum((2 * idx - n - 1) * c)) / (n * c.sum()))


def _spatial_concentration(means: np.ndarray, grid: int = 64) -> dict:
    """Occupancy stats over a `grid`^3 box spanning the gaussian means.
    Reports how concentrated the population is: Gini over occupied structure,
    and the share of gaussians in the densest 1% of occupied cells."""
    if means.shape[0] == 0:
        return {"gini": 0.0, "top1pct_cell_share": 0.0, "occupied_cells": 0,
                "max_cell_pop": 0}
    lo = means.min(axis=0)
    hi = means.max(axis=0)
    ext = np.maximum(hi - lo, 1e-6)
    q = np.clip(((means - lo) / ext * grid).astype(np.int64), 0, grid - 1)
    lin = (q[:, 0] * grid + q[:, 1]) * grid + q[:, 2]
    cell_pop = np.bincount(lin, minlength=grid ** 3)
    occ = cell_pop[cell_pop > 0]
    n = means.shape[0]
    # Share of population living in the densest 1% of OCCUPIED cells.
    k = max(1, int(np.ceil(0.01 * occ.size)))
    top = np.sort(occ)[::-1][:k].sum()
    return {
        "gini": round(_gini(occ), 4),
        "top1pct_cell_share": round(float(top) / n, 4),
        "occupied_cells": int(occ.size),
        "max_cell_pop": int(cell_pop.max()),
    }


def capture_pre(pipeline, *, structure: bool, width: int, height: int,
                structure_cfg=None, grad_thresh_internal: float | None = None,
                dense_scale: float | None = None) -> dict:
    """Snapshot BEFORE the densify runs. Reads eta / grad / scale state and
    computes what the decision WOULD do (mirrors the shader gates) so we can
    report gate selectivity and fan-out independent of the wait-free path.

    Returns a partial dict; pass it to :func:`capture_post` to complete."""
    pipeline.device.wait_for_idle()
    b = pipeline.buffers
    n = _counter(pipeline)
    out: dict = {"pre_count": n}

    means = _gaussian_means(pipeline, n)
    out["spatial_pre"] = _spatial_concentration(means)

    # World-space max-scale percentiles (depth-independent proxy for projected
    # L, and the quantity warmup SPLITTING is supposed to shrink). The
    # threshold-sweep causal test keys on this: lower warmup split threshold ->
    # more large blobs split -> smaller max_scale arriving at the structure
    # pass -> smaller eta -> smaller fan-out. Cheap: one buffer read.
    g_all = np.frombuffer(b.gaussians.to_numpy().tobytes(),
                          dtype=np.float32).reshape(-1, 14)[:n]
    max_scale_all = np.exp(g_all[:, 3:6].astype(np.float64)).max(axis=1)
    def _p(a, ps=(10, 25, 50, 75, 90, 99)):
        return {f"p{p}": round(float(np.percentile(a, p)), 5) for p in ps}
    out["world_max_scale"] = {**_p(max_scale_all), "max": round(float(max_scale_all.max()), 4)}
    out["opacity_pre"] = _opacity_pcts(pipeline, n)

    if not structure:
        return out

    # --- Recreate the structure decision gates on CPU (read-only) ---
    n_total = _u32(b.eta_n_total, n).astype(np.float64)
    n_high = _u32(b.eta_buffers["n_high"], n).astype(np.float64)
    n_low = _u32(b.eta_buffers["n_low"], n).astype(np.float64)
    max_x = _u32(b.eta_buffers["max_x"], n).astype(np.float64) / 65536.0
    max_y = _u32(b.eta_buffers["max_y"], n).astype(np.float64) / 65536.0
    max_z = _u32(b.eta_buffers["max_z"], n).astype(np.float64) / 65536.0
    vis = _u32(b.visibility_count_accum, n).astype(np.float64)
    grad_mag = _i64(b.grad_mean_2d_mag_accum, n).astype(np.float64) / 65536.0

    seen = n_total > 0
    inv_total = np.where(seen, 1.0 / np.maximum(n_total, 1.0), 0.0)
    high_ratio = n_high * inv_total
    low_ratio = n_low * inv_total
    avg_grad = np.where(vis > 0, grad_mag / np.maximum(vis, 1.0), 0.0)

    tau_split = structure_cfg.tau_split
    tau_prune = structure_cfg.tau_prune
    is_grad_high = avg_grad >= (grad_thresh_internal or 0.0)
    triggered = (high_ratio > tau_split) & is_grad_high & seen
    is_prune = ((low_ratio > tau_prune) & seen)  # opacity prune added in shader

    # Per-axis split factor n_k = clip(ceil(eta**power), 1, cap).
    power = structure_cfg.split_power
    cap = int(structure_cfg.max_split_factor)

    def nk(e):
        raw = np.ceil(np.power(np.maximum(e, 0.0), power))
        return np.clip(raw, 1, cap).astype(np.int64)

    nx, ny, nz = nk(max_x), nk(max_y), nk(max_z)
    prod = (nx * ny * nz)

    # Clone vs split by scale.
    g = np.frombuffer(b.gaussians.to_numpy().tobytes(),
                      dtype=np.float32).reshape(-1, 14)[:n]
    scale = np.exp(g[:, 3:6].astype(np.float64))
    max_scale = scale.max(axis=1)
    is_small = max_scale <= (dense_scale or 0.0)

    tent_split = triggered & (~is_small) & (prod > 1)
    tent_clone = triggered & (is_small | (prod <= 1))

    # Scalar eta_max = max over the 3 axes (the quantity classified against
    # ETA_HIGH=1 and fed to the fan-out). Its percentiles over SEEN gaussians
    # are the direct link between projected size and the split factor.
    eta_max_scalar = np.maximum(max_x, np.maximum(max_y, max_z))
    seen_n = int(seen.sum())
    out["eta_gate"] = {
        "n_seen": seen_n,
        "frac_seen": round(seen_n / max(n, 1), 4),
        "frac_triggered_of_seen": round(int(triggered.sum()) / max(seen_n, 1), 4),
        "frac_prune_of_seen": round(int(is_prune.sum()) / max(seen_n, 1), 4),
        "high_ratio_p50": round(float(np.percentile(high_ratio[seen], 50)) if seen_n else 0.0, 4),
        "high_ratio_p90": round(float(np.percentile(high_ratio[seen], 90)) if seen_n else 0.0, 4),
        "frac_high_ratio_gt_tau": round(float(np.mean(high_ratio[seen] > tau_split)) if seen_n else 0.0, 4),
        "eta_max_p50": round(float(np.percentile(eta_max_scalar[seen], 50)) if seen_n else 0.0, 3),
        "eta_max_p90": round(float(np.percentile(eta_max_scalar[seen], 90)) if seen_n else 0.0, 3),
        "n_total_p50": round(float(np.percentile(n_total[seen], 50)) if seen_n else 0.0, 2),
        "n_total_p90": round(float(np.percentile(n_total[seen], 90)) if seen_n else 0.0, 2),
    }
    # Fan-out (prod) distribution over tentative splits: are splits targeted
    # (2-8) or exploding (up to cap^3 = 512)?
    ps = prod[tent_split]
    if ps.size:
        out["fanout"] = {
            "n_tent_split": int(tent_split.sum()),
            "n_tent_clone": int(tent_clone.sum()),
            "prod_mean": round(float(ps.mean()), 2),
            "prod_p50": int(np.percentile(ps, 50)),
            "prod_p90": int(np.percentile(ps, 90)),
            "prod_max": int(ps.max()),
            "children_requested": int(ps.sum() - ps.size),  # tail growth
            "frac_at_cap": round(float(np.mean(ps >= cap ** 3)), 4),
        }
    else:
        out["fanout"] = {"n_tent_split": 0, "n_tent_clone": int(tent_clone.sum()),
                         "children_requested": 0}
    return out


def _opacity_pcts(pipeline, n: int) -> dict:
    """Activated-opacity (sigmoid) percentiles over the first n gaussians.
    Confirms the 0.1 structure floor culled the low tail and the 0.8 cap
    clamped the high tail — the two signatures we measured in the reference
    (ref it500-post: opacity p10=0.13, p90=p99=max=0.8)."""
    g = np.frombuffer(pipeline.buffers.gaussians.to_numpy().tobytes(),
                      dtype=np.float32).reshape(-1, 14)[:n]
    logit = g[:, 10].astype(np.float64)
    a = 1.0 / (1.0 + np.exp(-logit))
    ps = (1, 10, 50, 90, 99)
    return {f"p{p}": round(float(np.percentile(a, p)), 4) for p in ps} | {"max": round(float(a.max()), 4)}


def assert_opacity_bundle(pipeline, *, floor, cap_enabled, cap, tol: float = 0.02):
    """Diagnostic check that the structure pass HONORED the resolved opacity
    floor/cap on the live (compacted) buffer. Reads back activated opacity and
    asserts:

        min live opacity >= floor - tol      # structure prune floor applied
        max live opacity <= cap   + tol      # post-densify cap applied (if on)

    Asserts against the CONFIGURED values (not the reference 0.1 / 0.8), so a
    default run (floor 0.005, cap disabled) never false-fails — the assert only
    fires if the bundle was configured but did not take effect (e.g. the cap
    dispatch hit a stale buffer, or the floor constant never reached the shader).
    Returns (min_opacity, max_opacity, messages)."""
    pipeline.device.wait_for_idle()
    n = _counter(pipeline)
    g = np.frombuffer(pipeline.buffers.gaussians.to_numpy().tobytes(),
                      dtype=np.float32).reshape(-1, 14)[:n]
    a = 1.0 / (1.0 + np.exp(-g[:, 10].astype(np.float64)))  # sigmoid(logit)
    amin, amax = float(a.min()), float(a.max())
    msgs = []

    ok_floor = amin >= floor - tol
    msgs.append(f"floor>={floor}: min={amin:.4f} [{'OK' if ok_floor else 'FAIL'}]")
    assert ok_floor, (f"structure opacity FLOOR not applied: min live opacity "
                      f"{amin:.4f} < {floor} - {tol} (t_opacity did not reach the "
                      f"decide shader, or prune ran on the wrong buffer)")

    if cap_enabled:
        ok_cap = amax <= cap + tol
        msgs.append(f"cap<={cap}: max={amax:.4f} [{'OK' if ok_cap else 'FAIL'}]")
        assert ok_cap, (f"opacity CAP not applied: max live opacity {amax:.4f} > "
                        f"{cap} + {tol} (cap dispatch skipped or targeted the "
                        f"stale source buffer before the ping-pong swap)")
    else:
        msgs.append(f"cap: disabled (max={amax:.4f})")

    return amin, amax, msgs


def capture_post(pipeline, pre: dict, *, structure: bool) -> dict:
    """Snapshot AFTER densify + morton. Completes the pre dict with the
    realized growth + post-densify spatial concentration + opacity dist
    (verifies the cap/floor fired)."""
    pipeline.device.wait_for_idle()
    n = _counter(pipeline)
    pre["post_count"] = n
    pre["growth"] = n - pre["pre_count"]
    pre["growth_factor"] = round(n / max(pre["pre_count"], 1), 4)
    means = _gaussian_means(pipeline, n)
    pre["spatial_post"] = _spatial_concentration(means)
    pre["opacity_post"] = _opacity_pcts(pipeline, n)
    return pre
