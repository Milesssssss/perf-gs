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

# the measured sweet spots: 0.5 is free, 0.25 is the knee (see Measured results)
python scripts/running/v2_train.py --scene datasets/360_v2/bicycle \
    --stbn-pixel-fraction 0.25

# the control: same count, same code path, arrangement replaced by white noise
python scripts/running/v2_train.py ... --stbn-pixel-fraction 0.25 --stbn-source white

# re-measure the whole trade-off yourself (9 scenes x 5 densities, ~2.5 h)
python scripts/running/eval_sweep.py \
    --axis stbn-pixel-fraction --values 1,0.75,0.5,0.25,0.1
```

---

## Measured results

**All 9 Mip-NeRF 360 scenes, 3000 iters, quarter-res, structure-aware densify,
mip-splatting, seed 42, RTX 3090, held-out test PSNR (the paper metric).** Full
data in `outputs/eval_sweep/stbn_density/` (reproduce with
`scripts/running/eval_sweep.py --axis stbn-pixel-fraction --values 1,0.75,0.5,0.25,0.1`):

| rho | PSNR (avg of 9) | vs dense | train time | speedup |
|---|---|---|---|---|
| 1.0 (off) | 26.463 dB | — | 40.4 s | 1.00x |
| 0.75 | 26.420 dB | **-0.04** | 37.9 s | 1.07x |
| 0.50 | 26.423 dB | **-0.04** | 35.2 s | **1.15x** |
| 0.25 | 26.285 dB | -0.18 | 32.9 s | **1.23x** |
| 0.10 | 25.842 dB | -0.62 | 31.8 s | 1.27x |

Three things this says:

- **Down to rho=0.5 the mask is free.** -0.04 dB is inside run-to-run noise, and
  0.75 and 0.5 are indistinguishable from each other; several scenes come out
  *ahead* (flowers +0.04, treehill +0.03, stump +0.01). 15% of training time for
  nothing measurable.
- **rho=0.25 is the knee** — 0.18 dB for 1.23x. Past it the trade inverts:
  rho=0.10 costs 3.5x more quality (-0.62 dB) to buy almost no extra speed
  (1.23x -> 1.27x). There is no reason to run below 0.25 as a flat density.
- **The speedup saturates** because only the backward's per-pixel work shrinks.
  The forward, SH, Adam, the radix sort and SSIM pass 1 are all untouched, and
  they are most of a step. 1.27x is roughly the ceiling for this design; going
  further means a sparse FORWARD (see *Known limits*).

Per-scene, the cost at rho=0.10 lands hardest on the indoor scenes — kitchen
-1.00, garden -0.86, bonsai -0.83 — and barely registers on the sparse outdoor
ones (stump -0.40, treehill -0.17). At rho=0.25 the worst scene is kitchen at
-0.47 dB and four of nine are within 0.05 dB of dense.

**Blue noise beats its own control.** On bicycle at rho=0.10, STBN scored +0.19 dB
over `--stbn-source white` at identical density, identical per-tile count and
identical code path — so the arrangement, not just the sparsity, is doing work.
That is the premise the whole design rests on, and it is the A/B to re-run if the
mask is ever regenerated.

Caveat on the earlier single-scene numbers: a 1500-iteration bicycle-only run put
rho=0.25 at -0.46 dB, which is 2.5x worse than the 9-scene 3000-iteration result.
A 1500-iter run is spent almost entirely *inside* the densify window (100..1900),
which is the worst case for sparse supervision — short single-scene runs
systematically overstate the cost. Judge this on the table above, not on a quick
bicycle run.

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
- the count is known per tile, so the backward can walk a compacted list of
  exactly `k` entries instead of testing all `TILE_AREA` of them.

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
- **NOT smaller LDS**, despite what an earlier version of this doc and a stale
  comment in `backward_persplat.slang` claimed. Shrinking the staged arrays was
  supposed to be a second, independent occupancy win. At the 8x8 tile
  (`NUM_WARPS = 2`) those arrays total under 2 KB of the ~100 KB shared memory per
  SM, so it buys nothing measurable; the claim was inherited from the 16x16 era,
  where `gs_checkpoints` alone was 16 KB. `STBN_MAX_TILE_PIXELS` therefore stays
  at a full tile and `--stbn-max-fraction` is a hand-only diagnostic for
  re-testing that. All of the speedup is the loop.

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
| `--stbn-max-fraction` | `0` (off) | DIAGNOSTIC. Shrinks the backward's staged arrays; measured to buy nothing at an 8x8 tile (see above). Not in the panel. |
| `--stbn-source` | `nvidia` | `nvidia` / `white` (control) / `file`. |
| `--stbn-file` | `""` | A `(slices, height, width)` `.npy` scalar volume. |
| `--stbn-seed` | `0` | For `white` only; separate from `--seed` so the mask can be re-rolled without moving the camera order. |

Rejected combinations (`validate_training_config`):

- `--stbn-pixel-fraction < 1` **and** `--backward-pixel-stride > 1`: both decide
  which pixels the backward visits, and together they compose two subsamplings
  into a density neither flag states while the reweighting compensates for only
  one. The shader `#error`s on the define pair too.
The staged arrays hold a full tile and the readers clamp to it, so no density a
schedule passes through can outgrow them — there is nothing else to validate.

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

- **1.27x is the ceiling for this design.** Only the backward's per-pixel work is
  sparsified, and the measured speedup is already flat between rho=0.25 and 0.10.
  Going further needs a sparse FORWARD — each pixel walks its own splat chain
  independently, so skipping unsampled pixels would cut the blend by ~90% at
  rho=0.1 (roughly a second 8 ms of a 23 ms step). Two things block it: SSIM
  pass 1 reads the full rendered image in an 11x11 neighbourhood, so a sparse
  forward is only sound at `--lambda-ssim 0`; and `output_image` is also the
  preview/eval/viewer surface, so unsampled pixels would need to hold something
  sane. A cheaper partial win first: the forward writes `raster_checkpoints` for
  all 64 pixels per bucket and the backward now reads only k of them.
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
- **Densification is the thing to watch** at low `rho`, not smooth-surface PSNR.
  Across the 9-scene sweep the gaussian count falls monotonically with density
  (4.98 M dense -> 4.80 M at rho=0.1), and the scenes that lose the most PSNR at
  rho=0.1 are the detailed indoor ones (kitchen -1.00, garden -0.86, bonsai
  -0.83) rather than the sparse outdoor ones. That is the densifier seeing noisier
  gradient statistics, not Adam converging worse. Levers, in order: a warmup
  covering the whole densify window; feeding the densifier's `grad_2d_*_iter`
  statistics from a denser mask than the Adam gradient uses; and variable-density
  STBN, where the per-tile budget follows reconstruction error (sky 2%, texture
  10%, foliage 30%) at a fixed average — combining TurboGS-style error-guided
  effort with STBN's coverage. A lower flat `rho` is not one of the levers: the
  sweep shows it buys no time.

## Follow-up: variable-density STBN

The last lever above is now implemented and measured — as a smooth per-PIXEL
marginal rather than a per-tile budget, since a per-tile allocation would be a
different sampling process for the same reasons `tile_rank` is. See
[Importance-guided STBN sparse backward sampling](adaptive_stbn_importance_sampling.md)
for the construction, the flags and the results, including the negative ones. It is
strictly additive: every regime described on this page behaves exactly as it did
before, and at `alpha = 0` the adaptive path reproduces this page's mask and
gradient bit for bit.
