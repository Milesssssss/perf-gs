# SADGS — Structure-Aware Densification Cheat Sheet

> Lyu et al., *Faster 3DGS Convergence via Structure-Aware Densification* (arXiv 2604.28016).
> This repo's implementation. Enable with `--densify-mode structure_aware`.
> All names below are the actual code identifiers, with type and contents.

## The whole idea in 3 lines

1. For each Gaussian compute `eta_x`, `eta_y`, `eta_z` — how coarse each projected axis is vs. local image detail.
2. Tally per-Gaussian votes over many views (`eta_n_high` / `eta_n_low` out of `eta_n_total`).
3. `split_anisotropic` cuts a Gaussian into a grid of children along the axes whose `eta` is too high.

Everything else is GPU plumbing (wait-free = no CPU readback mid-densify), not the idea.

---

## Performance and diagnosing slow densification ticks

A training densification tick is **decision/split/prune + tracking clears + Morton
sort + Mip-filter recomputation**, not just the SAD split shader. Structure fields
are already GPU-resident; they are not uploaded again at each scoring view.

The implementation avoids three sources of unnecessary work:

- **Capacity-wide copy-back:** the shared scratch arena now commits only live
  records, using a GPU count. SH Adam state remains coefficient-major SoA, so each
  component's live prefix is copied at its original capacity stride. At SH3 and
  10M capacity, the former four copy-backs moved 6.2 GB per compaction or Morton
  pass, regardless of how few splats were live. No extra full-size scratch is used.
- **Serial child grids:** the decision compacts SPLIT/CLONE parents using wave
  reservations. One cooperative workgroup emits each selected parent's grid;
  invariant geometry is hoisted and capacity-overflow children are not computed.
  Child geometry, split thresholds, SH inheritance, and fresh optimizer moments
  are unchanged. Tail order is still intentionally unspecified.
- **Capacity-wide Mip workgroups:** the Mip pass dispatches only the GPU's live
  count, uses wave reductions, and clears only newly dead filter slots after its
  initial tail initialization. It still tests all training cameras, preserving
  the original filter calculation exactly.

Overbudget warnings still report requested growth, now including clones. The
seven full-capacity eta/gradient readbacks and CPU percentile analysis only run
when `--densify-diag` is explicitly enabled; they can transfer 320 MB at 10M
capacity and should not be used for ordinary performance measurements.

Enable `--stage-timing-every 25` for sampled training-step timings **and** separate
timings of every mutation event. The run writes `mutation_timing.jsonl`, and the
stage-timing summary includes `mutation_timings`. Each event reports GPU markers
for SAD/filler densification, tracking, Morton, and Mip; `host_complete_ms` includes
queue backlog and readbacks. Nested markers must not be added to their parents.
Instrumentation completion waits are opt-in and reported separately; normal
training adds no timing queries or extra completion waits.

Dataset-free probes are in `scripts/debug/bench_structure_densify.py` and
`scripts/debug/bench_mip_filter.py`. `--compare-reference` interleaves optimized
and committed-reference shaders on the **same** resources. The former compares
child geometry after ignoring unordered tail reservations; the latter checks
bit-exact filter outputs. Use realistic live/capacity ratios and camera counts.
GPU residency, backend, and background load can strongly affect wall time; an
isolated small-population result does not establish the duration of a full tick.

### Full-training A/B result (2026-10-08)

The full-loop launcher `scripts/debug/bench_training_densify.py` (since removed
along with the reference densifier; see git history) restored only the
old densification paths from source snapshots at commit `1a904565`. Both variants
used the current renderer, optimizer, sampling code, and identical allocations;
no working-tree files were reverted. `summarize_training_densify.py` checked the
recorded settings and source fingerprints before computing the comparison.

Completed four full **3,000-iteration** bicycle runs in before/after/after/before
order: Vulkan, seed 0, SADGS camera sampling, SH0, Mip enabled, NVIDIA global STBN
at fraction 0.1, **618×411 resolution and 2M capacity**. Setup and final three-view
quality checks were outside the timed loop; no GPU timing/profiling queries ran.

| Training-loop time | Before (two runs) | After (two runs) |
|---|---:|---:|
| Active seconds, mean | 12.9049 | 12.1058 |
| Active seconds, range | 12.6724–13.1374 | 12.0930–12.1186 |
| Raw wall seconds, mean | 12.9598 | 12.1627 |

Measured saving: **0.7991 s / 6.19% less active training time**, or **1.066×**
throughput. Raw wall time improves by 6.15%; the small difference is measured CPU
reporting/event-check overhead (about 0.055 s/run), not image evaluation. Final
populations differ by about 44 splats on average out of 2M. The fixed three-view
PSNR sample differs by +0.018 dB on average; this is not a full quality evaluation
or a bit-parity claim. Two samples per variant do not establish significance.

**This is not a measured result for the default 10M-capacity configuration.** That
attempt was stopped because roughly 13 GB of the 16 GB GPU was occupied before
launch, making even ordinary early iterations take about 0.6 s. Other GPU work
was left untouched. The reduced configuration completes without that severe
memory contention; do not extrapolate its percentage to the default workload.

---

## The per-Gaussian score variables (`score_structure_aware.slang`)

Computed per scoring view, one thread per Gaussian. All `float` unless noted.

| Variable | Type | Contents |
|---|---|---|
| `mean_vs` | `float3` | Gaussian center in view space |
| `mean_px` | `float2` | Gaussian center in pixels |
| `A` | `float3x3` | `J·W·R·diag(scale)`; its 3 columns are the projected principal axes |
| `len_vx`,`len_vy`,`len_vz` | `float` | pixel length of each projected axis = `sqrt(u² + v² + COV_DILATION)` |
| `sbar` | `float3` | structure tensor `(Sxx, Sxy, Syy)` sampled at the footprint |
| `lambda1` | `float` | top eigenvalue of `sbar` (local edge energy) |
| `inv_lambda_min` | `float` | `sqrt(max(lambda1,0)) + EPS` — the reciprocal of the finest local wavelength |
| `eta_x`,`eta_y`,`eta_z` | `float` | `len_v* * inv_lambda_min` — **axis-length-in-pixels ÷ finest-detail-period**, per axis |
| `eta_max_scalar` | `float` | `max(eta_x, eta_y, eta_z)` — drives the per-view vote |

**Reading `eta_x/y/z`:** dimensionless ratio. `= 1` → axis matched to the texture; `> 1` → axis
spans multiple detail-cycles (under-resolved → split); `≤ 0.1` → far finer than any local detail
(wasteful → prune). Per-axis so the split can target only the offending direction.

Constants (`float`): `ETA_HIGH = 1.0`, `ETA_LOW = 0.1`, `EPS = 1e-8`, `COV_DILATION = 1e-8`.

### Inputs the score shader reads
| Name | Type | Contents |
|---|---|---|
| `structure_tensor` | `Texture2D<float4>` | the detail-ruler field, `(Sxx, Sxy, Syy, 0)` per pixel |
| `tile_count_per_gaussian` | `StructuredBuffer<uint>` | this view's footprint tile count; `>0` = visible (gate) |
| `grad_2d_x_iter`,`grad_2d_y_iter` | `StructuredBuffer<int64_t>` | this iter's 2D-grad seed (fixed-point); per-view grad gate |
| `opacity_threshold` | `uniform float` | `sigmoid(opacity)` must exceed this (0.05) |
| `per_view_grad_thresh` | `uniform float` | per-view screen-grad floor (0 disables) |
| `pass_idx` | `uniform uint` | view/iter index; decorrelates jitter |
| `max_splats` | `uniform uint` | buffer capacity |

---

## The vote accumulators (`EtaResources`, `buffers.py`)

Built over the window because each view gives only one sample. All `uint[max_splats]`, indexed by Gaussian slot id.

| Buffer | Type | Contents |
|---|---|---|
| `eta_n_total` | `uint[max_splats]` | # views this Gaussian was validly scored in → **ratio denominator** |
| `eta_n_high` | `uint[max_splats]` | # views where `eta_max_scalar > ETA_HIGH` (split votes) |
| `eta_n_low` | `uint[max_splats]` | # views where `eta_max_scalar <= ETA_LOW` (prune votes) |
| `eta_max_x`,`eta_max_y`,`eta_max_z` | `uint[max_splats]` | running **per-axis** max of `eta_*`, fixed-point |

Written in the score shader via `InterlockedMax(eta_max_x[i], uint(max(eta_x,0)*FIXED_SCALE), _)` and
`InterlockedAdd` on the counts.

Gotchas:
- **Ratios, not raw counts** are thresholded → invariant to how many views saw the Gaussian.
- `eta_max_scalar` (scalar) drives the vote; `eta_max_x/y/z` (per-axis) drive the split *amount*.
- `eta_max_*` are **`uint` fixed-point** = `uint(eta * FIXED_SCALE)`, `FIXED_SCALE = 65536.0`
  (`fixed_point.slang`). Atomics can't float-max, so decode with plain `float(eta_max_x[i]) / FIXED_SCALE`
  — **NOT `fromFixed`** (that's `int64`, used only for the grad buffers).

`EtaResources` also holds a full scratch mirror (`scratch_components`, `scratch_n_total`) for the
carry-through-compaction (see bottom), plus `swap()` to flip primary↔scratch.

---

## Three clocks (all host-driven)

| Cadence | Runs | Flag |
|---|---|---|
| every `sad_accum_every` (~10) | **score** current view → fold one vote into the accumulators | `--sad-accum-every` |
| **every** iter | `accumulate_iter`: fold `grad_2d_*_iter` into `grad_mean_2d_mag_accum` + `visibility_count_accum` | (shared w/ simple densify) |
| every `sad_structure_every` (~500) | **decide + split/prune** (consume the votes) | `--sad-structure-every` |

Ordering: score runs **after** backward (grad seeds filled) but **before** they're cleared.

---

## The decision (`decide_actions_structure.slang`, `_tentative_decide`)

Named variables it computes per Gaussian:

| Variable | Type | Contents |
|---|---|---|
| `seen` | `bool` | `eta_n_total[i] > 0` |
| `inv_total` | `float` | `seen ? 1/max(eta_n_total[i],1) : 0` |
| `high_ratio` | `float` | `eta_n_high[i] * inv_total` |
| `low_ratio` | `float` | `eta_n_low[i] * inv_total` |
| `avg_grad` | `float` | `fromFixed(grad_mean_2d_mag_accum[i]) / visibility_count_accum[i]` |
| `is_grad_high` | `bool` | `avg_grad >= grad_threshold_internal` |
| `eta_x`,`eta_y`,`eta_z` | `float` | decoded per-axis maxima = `float(eta_max_x[i]) / FIXED_SCALE`, … |
| `nkx`,`nky`,`nkz` | `uint` | `_nk_from_eta(eta_*, split_power, max_split_factor)` = `clamp(ceil(pow(eta_*, split_power)), 1, max_split_factor)` |
| `prod` | `uint` | `nkx * nky * nkz` |

Logic:
```
PRUNE   if (low_ratio > tau_prune AND seen) OR opacity < t_opacity
trigger if (high_ratio > tau_split) AND is_grad_high AND seen
    CLONE if max_scale <= dense_scale OR prod <= 1     # small / degenerate
    SPLIT otherwise → write split_factor_x/y/z[i] = nkx/nky/nkz
else KEEP
```

Outputs:
| Name | Type | Contents |
|---|---|---|
| `action` | `RWStructuredBuffer<uint>` | per Gaussian: `0 KEEP, 1 SPLIT, 2 CLONE, 3 PRUNE` |
| `split_factor_x`,`split_factor_y`,`split_factor_z` | `RWStructuredBuffer<uint>` | per-axis child counts for SPLIT |

### Single dispatch (no capacity demotion)
| Buffer | Type | Contents |
|---|---|---|
| `densify_children_sum` | `uint[1]` | wave-summed `prod-1` per split and `1` per clone = requested tail growth |

`decideActionsStructure` is one kernel: it decides KEEP/SPLIT/CLONE/PRUNE per Gaussian and atomically
sums `densify_children_sum`. There is **no** capacity clamp / demotion. If a window's requested splits
exceed the buffer headroom, `split_anisotropic` drops the overflow children at the tail and the host
reads back `densify_children_sum` after the tick to **warn** (`requested > headroom`). We surface an
overbudget splat explosion so you can respond (tune `--sad-tau-split` / `--sad-max-split-factor` /
`--max-splats`) rather than silently downsizing splits.

---

## Anisotropic split (`split_anisotropic.slang`, `_grid_child`)

Lay an `nkx × nky × nkz` grid in the parent's local frame. Named variables:

| Variable | Type | Contents |
|---|---|---|
| `n` | `uint3` | `max(split_factor_*, 1)` per axis |
| `nf` | `float3` | `n` as float |
| `coord` | `float3` | centered grid index: `coord.k = idx.k - 0.5*(n.k-1)`, or 0 when `n.k==1` |
| `sep` | `float3` | `(scale / nf) * SQRT12`, `SQRT12 = sqrt(12)` (variance-preserving spacing) |
| `child.mean` | `float3` | `parent.mean + mul(R, coord * sep)` |
| `child.log_scale` | `float3` | `parent.log_scale - ks_scale_power * log(nf)` → `scale.k / n.k^ks_scale_power` |

- `rot_quat`, `opacity`, SH: inherited unchanged. No random jitter (deterministic grid).
- `ks_scale_power` (`uniform float`): 1.0 = variance-preserving (Mip-NeRF 360); 1.2 (T&T / Deep Blending).
- `pre_count_buf` (`uint[1]`): Gaussian count before this densify; loop bound so new children aren't re-split.
- child 0 overwrites the parent slot; the rest fill a contiguous tail block (one `InterlockedAdd`);
  over-capacity children are dropped.
- Then **prune is fused in**: `build_prune_mask` → scan → `scatter_compact` → new count.
  (This tail is what `--densify-mode prune_only` reuses on its own.)

---

## Hyperparameters (`StructureConfig` / `DensifyConfig`)

| Flag / field | Type | Default | Contents |
|---|---|---|---|
| `tau_split` | `float` | 0.8 | SPLIT if `high_ratio` exceeds this |
| `tau_prune` | `float` | 0.8 | PRUNE if `low_ratio` exceeds this |
| `grad_threshold` | `float` | 1e-5 | `is_grad_high` floor (host scales into `grad_threshold_internal` by ×`width*height*3`) |
| `split_power` | `float` | 0.5 | exponent in `_nk_from_eta`; 0.5 → `ceil(sqrt(eta))` |
| `max_split_factor` | `int` | 8 | per-axis cap on `nk*` (so `prod` ≤ 512) |
| `ks_scale_power` | `float` | 1.0 | child scale exponent `scale.k / n.k^ks_scale_power` |
| `dense_scale` | `float` | `dense*extent` | CLONE (≤) vs SPLIT (>) cutoff on `max_scale`, world units |
| `t_opacity` | `float` | 0.005 | opacity PRUNE floor |
| `opacity_cap` | `float` | 0.8 | post-densify clamp on survivor opacity |
| `n_scoring_views` | `int` | 10 | explicit view count (offline/test path only) |
| `--sad-accum-every` | `int` | 10 | score every N iters |
| `--sad-structure-every` | `int` | 500 | decide/split every N iters |

`DensifyConfig` also carries `t_big_scale`, `t_opacity_reset_value`, `reset_relative`, `reset_decay`
(opacity-reset behavior, shared with simple densify).

---

## `carry_eta` — surviving compaction / Morton re-sort

The accumulators are **slot-indexed**, but compaction and Morton re-sort renumber slots → votes would
point at the wrong Gaussian. `scatterCompactEta` / `mortonScatterEta` permute all six accumulators
through the **same index map** as the primitives; `EtaResources.swap()` flips primary↔scratch atomically
with the Gaussian buffers.

Controller rule (`densification.py`):
```python
sad_carry_eta = (mode == STRUCTURE_AWARE and not do_structure_step)
```
- **Structure tick** (decision fired): don't carry, **clear** after (`clearStructureScores`) — window
  consumed, slots changed, start fresh.
- **Filler / periodic Morton** (between decisions): **carry** — the multi-view window must survive to
  keep accumulating toward the next decision.

> "Filler": in `structure_aware` mode, non-structure densify ticks still run gradient-based
> `densify_simple` (AbsGS) as a filler, carrying eta through. The real anisotropic split only happens
> every `sad_structure_every`.

---

## File map

| File | Role |
|---|---|
| `shaders/structure/structure_tensor.slang` | build the `structure_tensor` field (`Sxx,Sxy,Syy`) |
| `shaders/densify/score_structure_aware.slang` | compute `eta_*`, accumulate the vote buffers |
| `shaders/densify/decide_actions_structure.slang` | votes → `action` + `split_factor_*` (Pass1/Threshold/Pass2) |
| `shaders/densify/split_anisotropic.slang` | materialize the child grid |
| `shaders/common/fixed_point.slang` | `FIXED_SCALE = 65536.0`, `toFixed`/`fromFixed` (`int64`) |
| `src/trainer/backend/densify/structure_tensor.py` | GPU builder of the `structure_tensor` field |
| `src/trainer/backend/densify/structure_densifier.py` | host `StructureDensifier`: score accumulation + densify pass |
| `src/trainer/config/config.py` | `DensifyConfig`, `StructureConfig` |
| `src/trainer/backend/memory/buffers.py` | `EtaResources` allocation + `swap()` |
| `src/trainer/frontend/densification.py` | controller: cadence + `sad_carry_eta` rule |
