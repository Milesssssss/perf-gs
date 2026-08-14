# STBN sparse pixel supervision

> Spatiotemporal blue noise from <https://github.com/NVIDIA-RTX/STBN>
> (Wolfe et al., *Scalar Spatiotemporal Blue Noise Masks*, arXiv:2112.09629 /
> EGSR 2022), re-ranked per rasterizer tile.
> **Off by default.** Enable with `--stbn-pixel-fraction <rho>`.
> All names below are actual code identifiers.

## The whole idea in 4 lines

1. Each backward supervises only a fraction `rho` of the pixels; the surviving
   per-pixel terms are scaled by `1/rho`, so the gradient stays an **unbiased**
   estimate of the full-image gradient and the LR needs no retuning.
2. *Which* pixels: a spatiotemporal blue-noise mask, so each visit's pixels are
   well spread across the image **and** a pixel's samples are well spread across
   the repeat visits of its camera.
3. The mask advances per **camera visit**, not per iteration — successive
   evaluations of the same image are the sequence the temporal axis was built for.
4. The backward walks a **compacted list** of the `k = rho * 64` sampled pixels
   per tile, so its pixel-loop trip count drops with `rho` instead of staying at
   `TILE_AREA`.

## Quick start

```bash
# once: fetch the mask (NON-COMMERCIAL license, cached in .stbn/, never committed)
python scripts/data/download_stbn.py

# ~10% of pixels, full density for the first 300 iters
python scripts/running/v2_train.py --scene datasets/360_v2/bicycle \
    --stbn-pixel-fraction 0.1 --stbn-warmup-iters 300

# the control: same count, same code path, arrangement replaced by white noise
python scripts/running/v2_train.py ... --stbn-pixel-fraction 0.1 --stbn-source white
```

---

## First measurements (read this before believing in it)

bicycle, 1500 iters, `--densify-mode structure_aware --sh-degree 0
--mip-splatting --seed 42`, RTX 3090, held-out test PSNR (paper metric):

| config | wall | it/s | PSNR | gaussians |
|---|---|---|---|---|
| dense (`rho=1`) | 18.4 s | 81.5 | **23.106** | 7.54 M |
| STBN `rho=0.25` | 14.6 s | 102.8 | 22.647 | 7.09 M |
| STBN `rho=0.10` | 13.9 s | 107.8 | 21.959 | 6.74 M |
| white `rho=0.10` | 13.9 s | 108.2 | 21.769 | 6.80 M |

**The speedup is real: -21% wall at `rho=0.25`, -24% at `rho=0.10`.** It saturates
early because only the backward's per-pixel work shrinks — projection backward,
SH, Adam, the sort and the forward do not, and SSIM pass 1 does not either.

**The quality cost is also real**, and larger than the design hoped: -0.46 dB at
`rho=0.25` and -1.15 dB at `rho=0.10` at matched iterations. On PSNR vs
*wall-clock* (`--eval-every 250`, `elapsed_s` excludes eval time) `rho=0.25` is
roughly break-even for the first ~500 iterations and slightly behind after that;
`rho=0.10` is behind throughout. So on this configuration sparse supervision does
**not** yet pay for itself.

Two things in those numbers are worth more than the headline:

- **Blue noise beats its control by +0.19 dB** at the same `rho`, same per-tile
  count, same code path. The arrangement is doing measurable work, which is the
  premise the whole design rests on.
- **The gaussian count drops 10%** (6.74 M vs 7.54 M) at `rho=0.10`. That is the
  densifier growing less, not Adam converging worse — exactly the failure mode
  predicted for this regime, and the reason `--stbn-warmup-iters` exists.

Caveat that likely dominates all of the above: a 1500-iteration run is spent
almost entirely *inside* the densify window (100..1900), which is the worst case
for sparse supervision. The regime this should be tested in is a long run where
the sparse steps land in the refinement tail. `--stbn-warmup-iters 400` moved
`rho=0.10` by only +0.04 dB here, which says a 400-iteration hedge is not enough
when densification runs to 1900 — try a warmup that covers the window, or the
variable-density idea in *Known limits*.

The sweep worth running next is `rho in {0.5, 0.25}` at 7k-30k iterations, on
PSNR vs wall-clock.

---

## Why blue noise, with numbers

All measured on the shipped asset after tile ranking, at `rho = 6/64 = 9.4%`
(pinned by `tests/test_stbn.py`):

| property | statistic | STBN | white noise | ideal |
|---|---|---|---|---|
| spatial spread | low-frequency spectral power | **0.0065** | 0.0345 | lower |
| temporal spread | consecutive-visit hits per pixel | **0.000** | 0.549 | lower |
| exposure fairness | std of per-pixel visit frequency | **0.020** | 0.037 | lower |
| per-tile count | min..max pixels per 8x8 tile | **6..6** | 6..6 | exact |

The temporal row is the one that justifies a *spatio-temporal* mask rather than
one static blue-noise mask reused on every visit: at low density a pixel
essentially never gets sampled on two consecutive visits of the same camera.

## What we change about the published mask

### 1. Tile stratification (exactly k per tile)

Thresholding the raw mask globally gives each tile a random count around
`rho * 64`. On NVIDIA's slice 0 at `rho = 0.1`, per-tile counts run **2..11**
around a mean of 6.5 — one tile gets a fifth of the intended gradient budget
while its neighbour gets nearly double.

So `rank_volume_from_scalar` re-ranks the mask **within each 8x8 tile of each
slice**, making every tile a permutation of `0..63`. Then `rank < k` selects
exactly `k` pixels per tile, everywhere, on every slice. This buys three things:

- the `TILE_AREA/k` reweighting is **exact**, not an expectation;
- the per-tile gradient budget is uniform, so no tile is starved;
- the count is known at compile time, so the backward can size its staged LDS
  arrays (`STBN_MAX_TILE_PIXELS`) and shrink them with `rho`.

Ranking only reorders *within* a tile, and a blue-noise mask's per-tile histogram
is already near-uniform, so what it removes is the count jitter, not the blue
noise — the table above is measured *after* stratification.

### 2. Ranks, not scalars

Because each tile is now a permutation we store the rank (0..63), not the
original 0..255 value. Selection is the exact integer compare `rank < k`, and
the selected sets are **nested** (`{rank<k} ⊆ {rank<k'}` for `k<k'`), so a
density schedule walks a progressive sequence instead of re-rolling an unrelated
mask at every phase change.

---

## Where the mask is applied

### The loss defines it (`losses/l1_l2_loss.slang`, `losses/ssim_loss.slang`)

An unsampled pixel gets `dL_dC = 0`; a sampled one gets its gradient scaled by
`TILE_AREA/k`. That single decision propagates everywhere:

- **the backward may skip unsampled pixels** because they would contribute
  exactly zero — `x + 0.0 == x`, so skipping is exact, not approximate;
- **every backward variant** (per-bucket, per-tile per-splat, per-pixel) inherits
  the reweighting without knowing the mask exists;
- **the densifier's gradient statistics** (`grad_2d_{x,y}_iter`, an abs-sum) come
  out on the full-backward scale automatically, since `|s·g| = s·|g|`, so its
  thresholds do not need rescaling.

**SSIM masks pass 2, not pass 1.** Pass 1's maps describe how the loss at each
window *centre* responds to local statistics, and a sampled pixel's 11x11 window
legitimately spans unsampled neighbours — starving pass 1 would change the SSIM
gradient itself. What is subsampled is the chain-rule sum over pixels,

```
dL/dtheta = sum_q (dL/dx_q) (dx_q/dtheta)
```

keeping `k` of `TILE_AREA` of its terms and scaling by `TILE_AREA/k`. That is an
unbiased estimate of the **full** SSIM gradient, with higher variance — which is
why the mask is unbiased for any image loss, not just per-pixel ones. Pass 1
therefore still runs over every pixel and its cost is *not* reduced.

### The backward exploits it (`backward_persplat.slang`)

Both entry points loop over a **visit index** `v`, not a pixel index:

```hlsl
for (uint v = 0u; v < visit_count; ++v)
{
    uint p = gs_stbn_visit[warp_idx][v];   // the v-th sampled pixel
    uint checkpoint_slot = v;              // its checkpoint, compacted to slot v
    ...
}
```

At `k=6` that is 6 loop iterations instead of 64. Two properties make this
cheap and safe:

- **Warp-uniform.** All 32 lanes of a bucket own splats of the *same tile*, so
  the visit list is warp-uniform and every lane skips the same pixels together.
  No divergence, and the wave scans that reconstruct transmittance stay in
  lockstep. (This is the reason a *tile*-aligned mask matters for this
  rasterizer specifically.)
- **Smaller LDS.** `STAGED_PIXELS` becomes `STBN_MAX_TILE_PIXELS`, so the
  per-warp checkpoint stage shrinks with `rho`. Shared memory is this kernel's
  occupancy limiter (see the note in `backward_persplat.slang`), so that is a
  second, independent win.

`tileMaxProcessed` (`live_buckets.slang`) also reduces over sampled pixels only:
at a low budget a tile's deepest ray is usually an *unsampled* pixel, and
counting it would keep buckets alive whose every contribution is zero.

### The reference path

`STBN_ENABLED` is a compile-time define, and the loss masks at **runtime**. That
gives a correct-but-slow configuration for free: mask on, compacted backward off
— the backward keeps its exact full-pixel loop, reaches the unsampled pixels,
finds zeros and adds nothing. `tests/test_stbn_gradients.py` compares the two
head-to-head, which is the correctness gate for the compaction. Hub runs also
land here, because shader defines are fixed before the panel exists.

---

## The temporal axis is per-camera

```
slice_index = (visit_count[camera] + phase[camera]) % slices
```

- **`visit_count`, not `iteration`.** The mask distributes a pixel's values along
  its temporal axis; the sequence that axis should track is "successive
  evaluations of the same image". Indexing by iteration would hand a given camera
  an essentially arbitrary slice on every visit and throw the property away.
- **A fixed per-camera `phase`.** Without it every view samples slice 0's pixels
  on its first visit, slice 1's on its second, and so on — coupling the entire
  dataset to one sequence. The phase is a multiplicative mix of the camera index,
  deliberately *not* a draw from the run's RNG, which would shift the camera
  sampling order frozen by `tests/test_camera_sampling.py`.

Lives in `TrainingLoop._stbn_variables` (called exactly once per iteration — it
mutates the visit counter).

## The density schedule

`StbnDensitySchedule` ramps `rho` from 1.0 down to the target, log-linearly, over
`--stbn-warmup-iters` (same shape as `ExtinctionHardnessSchedule`, descending).

The reason to bother: sparse supervision is hardest on exactly the thin,
high-frequency structures — leaves, grass, fences, thin branches, small specular
features, tiny late-training Gaussians — that early densification is trying to
create. A splat covering a handful of pixels can get **zero** samples on a given
visit. Parameter optimization tolerates that (the next visit's samples fill in);
the *densifier's* statistics are the more fragile consumer, since a structure
that is never sampled is a structure it never learns to split for. Starting dense
is the cheap hedge. `--stbn-warmup-iters 300` with `--densify-from 100` keeps the
first few densify ticks fully supervised.

---

## Knobs

| flag | default | notes |
|---|---|---|
| `--stbn-pixel-fraction` | `1.0` | `rho`. 1.0 = off, bit-identical. Runtime uniform. |
| `--stbn-warmup-iters` | `0` | Steps ramping 1.0 -> `rho`. |
| `--stbn-max-fraction` | `0` (auto) | Compile-time staged capacity. Auto = the run's own density, or 1.0 when a warmup ramps from full. |
| `--stbn-source` | `nvidia` | `nvidia` / `white` (control) / `file`. |
| `--stbn-file` | `""` | A `(slices, height, width)` `.npy` scalar volume. |
| `--stbn-seed` | `0` | For `white` only; separate from `--seed` so the mask can be re-rolled without moving the camera order. |

Rejected combinations (`validate_training_config`):

- `--stbn-pixel-fraction < 1` **and** `--backward-pixel-stride > 1`: both decide
  which pixels the backward visits, and together they compose two subsamplings
  into a density neither flag states while the reweighting compensates for only
  one. The shader `#error`s on the define pair too.
- an explicit `--stbn-max-fraction` below the density the run actually samples.

## Licensing

The default mask is NVIDIA's published asset, under the *NVIDIA Source Code
License for Spatiotemporal Blue Noise Texture Generation*: **non-commercial
research/evaluation use only**, and redistribution must carry the license plus
attribution. Therefore:

- the download is **opt-in** (`scripts/data/download_stbn.py`) and the feature is
  off by default;
- assets land in `.stbn/` which is **gitignored** — vendoring them into a shared
  repo is a redistribution, and that is a call for a human to make;
- the license text and a provenance `NOTICE.md` are written next to the data.

For commercial use, `--stbn-source file` accepts any scalar STBN volume (e.g. one
built from that SDK's ScalarApp under whatever terms apply to you).
`--stbn-source white` needs no asset but is the *control*, not an equivalent —
it measures what the blue-noise arrangement is worth, and per the table above the
arrangement is most of the point.

## Where things live

| file | role |
|---|---|
| `src/trainer/backend/render/stbn.py` | mask spec, ranking, packing, disk cache, upload, quality metrics |
| `shaders/common/stbn.slang` | table layout + selection helpers, `STBN_ENABLED` / `STBN_MAX_TILE_PIXELS` |
| `shaders/losses/*.slang` | where the mask is applied (zero + `TILE_AREA/k`) |
| `shaders/rasterize/tile_binned/backward_persplat.slang` | compacted visit loop |
| `shaders/rasterize/tile_binned/live_buckets.slang` | live-bucket reduction over sampled pixels |
| `src/trainer/config/config.py` | `StbnConfig`, `StbnVariables` |
| `src/trainer/frontend/schedule.py` | `StbnDensitySchedule` |
| `src/trainer/frontend/app.py` | `resolve_stbn_config`, `ShaderSpecialization`, `_stbn_variables` |
| `scripts/data/download_stbn.py` | asset fetch + license handling |
| `tests/test_stbn.py` | mask invariants + blue-noise quality |
| `tests/test_stbn_gradients.py` | reference-vs-compacted equivalence, unbiasedness |

## Known limits

- **SSIM pass 1 is not sparsified** (see above), so it remains a fixed cost.
- **The estimator is unbiased when the slice index is uniform** over the mask's
  64 slices — exactly true once a camera has been drawn a multiple of 64 times.
  Short of that, per-pixel exposure over one cycle spreads ~±2% absolute around
  `rho` (about half of white noise's spread).
- **The rendering literature's convergence argument does not transfer directly.**
  STBN's temporal-integration story assumes repeated estimates of roughly the
  same quantity; during training the Gaussians change between visits. The spatial
  stratification and temporal coverage arguments still hold, but treat the
  quality result as an empirical question, not a guarantee.
- **Densification is the thing to watch** at low `rho`, not smooth-surface PSNR —
  and the measured 10% gaussian-count deficit says it is already being felt. The
  next levers, in order: a warmup that covers the whole densify window; feeding
  the densifier's `grad_2d_*_iter` statistics from a denser mask than the Adam
  gradient uses; and variable-density STBN, where the per-tile budget follows
  reconstruction error (sky 2%, texture 10%, foliage 30%) at a fixed average —
  which would combine TurboGS-style error-guided effort with STBN's coverage.
  A lower flat `rho` is not one of the levers.
