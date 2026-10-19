# Importance-guided STBN sparse backward sampling

> Spends the same sparse backward budget preferentially where the screen-space
> loss gradient is large, without touching the blue-noise field and without
> changing the training objective.
> **Off by default.** Enable with `--stbn-adaptive` on top of
> `--stbn-pixel-fraction <rho> < 1`.
> Strictly additive: the `full`, `iid_white`, `indep_blue` and `stbn` regimes of
> [STBN sparse pixel supervision](stbn_pixel_sampling.md) are unchanged and remain
> the reference baseline. All names below are actual code identifiers.

## The whole idea in 5 lines

1. The existing sparse regimes pick `rho N` pixels with a **uniform** marginal;
   only the *arrangement* of the chosen set differs between them.
2. This adds the other axis — a **per-pixel** inclusion probability `p_i` driven
   by the screen-space loss gradient — while keeping the arrangement machinery
   untouched.
3. `M_{i,t} = 1[U_{i,t} < p_i]`, where `U` is the **unmodified** STBN scalar. The
   importance map moves the *threshold*; the blue noise decides how the resulting
   samples are arranged in space and time.
4. `lambda` is solved so `sum_i p_i = rho N` **exactly**, so an adaptive arm and
   its uniform control spend the same expected pixels.
5. The gradient handed to the backward is `h'_i = M_i h_i / p_i`. Conditional on
   the forward state `E[h'_i] = h_i`, so the Gaussian gradient stays an
   **unbiased** estimate of the dense one and the objective is unchanged.

The conceptual split, and the one thing to remember:

> **the importance field decides where computation goes; the blue-noise field
> decides how those samples are distributed.**

## Quick start

```bash
# once: fetch the blue-noise volume (NON-COMMERCIAL license, cached in .stbn/)
python scripts/data/download_stbn.py

# adaptive STBN at the locked configuration
python scripts/running/v2_train.py --scene datasets/360_v2/bicycle \
  --stbn-pixel-fraction 0.25 --stbn-source nvidia \
  --stbn-adaptive --stbn-importance-alpha 0.75 --stbn-importance-sigma 8

# the two controls, same probability map, different uniform sequence
#   ... --stbn-adaptive --stbn-adaptive-uniform iid              --stbn-source white  --stbn-independent
#   ... --stbn-adaptive --stbn-adaptive-uniform blue_independent --stbn-source nvidia --stbn-independent
```

The full experiment suite:

```bash
.venv/Scripts/python.exe scripts/running/adaptive_stbn/run.py --stages all
```

## The pipeline

Runs **between** the image-space loss and the Gaussian backward. Neither is
modified.

| step | where | what |
|---|---|---|
| `s_i = \|\|dL/dC_i\|\|_2` | `importanceReduce` | RGB norm of the **dense** loss gradient, box-averaged onto the coarse grid |
| `s~ = G_sigma * s` | `importanceBlurX/Y` | separable Gaussian at coarse resolution, edge-clamped |
| stats | `importanceStats` / `importanceResolve` | `mean(s~)`, `min`, `max`; derives the histogram range; clears state |
| `a_i` at full res | `importanceHistogram` | `(1-alpha) + alpha s~_i/(mean(s~)+eps)`, bilinearly reconstructed, and histogrammed |
| `lambda` | `solveLambda` | bisection on `sum_i min(1, lambda a_i) = rho N` over the histogram |
| `p`, `M`, `h'`, table | `applyAdaptive` | one 64-thread group per screen tile: samples, reweights `dL_dC` in place, emits the packed mask |

Three of the eight dispatches touch the full image; the rest work on the coarse
grid or a 257-entry histogram. Measured on bicycle at 1236x821: **0.13 ms** for
the importance passes plus **0.06 ms** for the apply, against a 0.70 ms backward
raster — reported as its own column, never folded into a speedup claim.

### Why the loss runs dense

`h_i` has to exist for the whole image before the mask can be built, so the loss
pass is invoked with the mask **off** (which is the pre-existing dense code path,
byte for byte) and a separate pass applies the adaptive mask afterwards. The cost
is real: the photometric gradient is evaluated for all `N` pixels instead of
`rho N`. It is image-space work, not rasterizer work, which is why the primary
performance metric stays the **backward raster time**.

It also means no fallback proxy is needed. An error-magnitude stand-in for `h_i`
would only be justified if the exact screen-space gradient were unavailable before
the expensive backward; here it is available, so the preferred mode is the only one
implemented.

### Why the backward needs no changes at all

The existing packed table (`shaders/common/stbn.slang`) carries the selected
**set**, not a ranking or a threshold: `slot | compacted << 8 | count << 16`. A
nonuniform marginal is expressible in it directly, and the backward already reads
its per-tile count *from the table* because a global STBN threshold varies it too.
So the adaptive path just hands the backward a screen-sized, single-slice table
with `stbn_dims` = the tile-padded frame size and `stbn_slice = 0`. The table is
allocated at the padded size precisely so that
`stbnBlockBase(tile origin, padded dims, 0) == tile_idx * TILE_AREA`.

## The three regimes

All three share the **identical** probability map; only the uniform sequence
differs, which is what makes the ladder a controlled comparison.

| `--stbn-adaptive-uniform` | `U_i` | mirrors |
|---|---|---|
| `iid` | fresh integer-hash uniform per pixel per visit | (new) |
| `blue_independent` | blue-noise field, slice drawn independently per step | `indep_blue` |
| `blue_temporal` | same volume, slice = the camera's own visit counter | `stbn` |

`--stbn-adaptive-biased` is a labelled **diagnostic control**, never the proposed
method: it samples from `p_i` but omits the `1/p_i` weight, i.e. deliberately
optimizes an error-weighted objective. It still applies the *global* `1/rho` so its
gradient magnitude matches the unbiased arm — otherwise it would also change the
effective learning rate and the meaning of the densifier's absolute
`--grad-thresh`, confounding the question it exists to ask.

## Design decisions worth knowing

**`alpha = 0` is bitwise the existing mask.** `U` is built as the per-slice
*global normalized rank* `(rank + 0.5)/(H*W)`, so `U < k/64` selects exactly the
set `stbn.global_selected_mask` keeps — the asset's 8-bit quantization cannot
land the threshold mid-level. The solver answers a constant field analytically, so
`p_i = k/64` exactly and `1/p_i` is literally the `TILE_AREA/k` the loss shaders
apply. Both the mask and the `dL_dC` texture are bit-identical to the non-adaptive
path (`tests/test_adaptive_stbn.py`).

**The budget is solved for the *quantized* rho.** `pixels_per_tile(rho)/64`, which
is the density the uniform table actually realizes — at `rho=0.10` that is
6/64 = 0.09375, not 0.10. Solving for the requested value instead would hand the
adaptive arm 6.7% more supervised pixels than its control.

**Clamping without correcting is wrong, and the tests say so.** `min(1, rho a_i)`
loses the clipped mass and undershoots the density by however much was clipped —
silently, and by more the peakier the field.
`test_saturation_is_reached_and_still_budgeted` asserts that the naive baseline
undershoots, so it would fail if it were ever substituted for the bisection.

**`sigma` means output-image pixels.** The field is evaluated at
`--stbn-importance-downsample` resolution purely for cost; `coarse_sigma`
compensates the box-downsample and the bilinear reconstruction so the composed
operator still smooths at `sigma` full-res pixels (measured within 2% at the
default). Without that compensation a sigma sweep would be measuring the
downsample. The grid is deliberately **not** the 8x8 rasterizer tile and not the
128x128 STBN texture; `test_importance_field_has_no_tile_periodic_structure`
measures the tile-frequency component rather than assuming it away.

**`lambda` is bracketed from 0, not from rho.** `rho` is the exact answer when
`mean(a) = 1` and nothing saturates, but a float32 mean can land a hair above 1 and
put the root just below `rho`. Sixty bisection steps resolve it either way.

**Blue-noise quality is read off the residual, not the mask.** A nonuniform density
necessarily puts low-frequency content into `M` — that content *is* `p`. The
diagnostics use `Z_i = (M_i - p_i)/sqrt(p_i(1-p_i) + eps)`, whose mean is 0 and
variance 1 at every pixel regardless of the density.

## Measured

From `outputs/adaptive_stbn/report.md`, which is regenerated from the artifacts and
is the authority. The headline findings:

- **Calibration.** Empirical inclusion frequency matches `p_i` to the sampling
  noise floor on all four synthetic fields, saturating ones included; the budget
  constraint holds to 1e-8.
- **Structure preserved.** Under a nonuniform density the blue arms still suppress
  residual low-frequency power by ~10x against white, and only the temporal arm
  carries a strongly negative lag-1 autocorrelation. It degrades as the field gets
  peakier — saturated pixels have zero residual variance and punch holes in the
  point set.
- **Unbiased in practice.** On frozen states the accumulated norm ratio approaches
  1 and the relative error falls as `K^-1/2`.
- **But the allocation does not improve gradient quality at K=1.** Inverse-
  probability weighting raises single-mask variance, and `||h_i||` is a weak proxy
  for `||J_i^T h_i||`. See the report's failure-case section.
- **Blue noise still helps on top of it.** With the identical probability map,
  adaptive IID > adaptive independent blue > adaptive STBN in error at every `K`.

## Flags

| flag | default | meaning |
|---|---|---|
| `--stbn-adaptive` | off | enable importance-guided sampling |
| `--stbn-importance-alpha` | 0.75 | uniform/importance mixture; 0 = exactly uniform, must be < 1 |
| `--stbn-importance-sigma` | 8.0 | smoothing scale in output-image pixels; 0 = raw per-pixel field |
| `--stbn-importance-downsample` | 4 | resolution divisor for the field (cost only) |
| `--stbn-adaptive-uniform` | `blue_temporal` | `iid` / `blue_independent` / `blue_temporal` |
| `--stbn-adaptive-biased` | off | diagnostic control: drop the `1/p_i` weight |

Rejected combinations (all with a message saying why):
`--stbn-adaptive` without a sparse fraction; `alpha >= 1`; a `blue_*` uniform whose
`--stbn-independent` disagrees with it; a `blue_*` uniform with
`--stbn-source white`; `--stbn-warmup-iters > 0`; and `--stbn-max-fraction`, which
would cap the backward's compacted visit list below what a hot tile can select and
silently truncate exactly the tiles the importance map targeted.

## Code map

| file | role |
|---|---|
| `src/trainer/backend/render/adaptive_stbn.py` | the math, and the **host reference** the GPU path is pinned to |
| `src/trainer/backend/render/adaptive_sampler.py` | GPU resources, dispatch order, diagnostics drain |
| `shaders/losses/adaptive_importance.slang` | the eight kernels |
| `src/trainer/config/config.py` | `AdaptiveStbnConfig` |
| `tests/test_adaptive_stbn.py` | equivalence, budget, unbiasedness, packing, guards |
| `scripts/running/adaptive_stbn/` | the experiment suite (sampling, gradients, visuals, training, report) |
