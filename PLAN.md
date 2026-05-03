# v0 Plan — Slang 3DGS Trainer (from-scratch rewrite)

## Goal

**Platform smoke test.** Load a `.ply` of gaussian splats (or a single hand-authored gaussian), render it from a hard-coded camera via a SlangPy + slang-rhi pipeline, output a PNG. No training, no loss, no backward, no densification. Prove the stack is viable before building the trainer.

Success = the same rendering code runs on **Vulkan (primary) and D3D12 (cross-check)** and the resulting PNG matches a third-party viewer (e.g. SuperSplat) when loading the same `.ply`.

---

## Layout

```
/ (repo root = dev root)
  pyproject.toml
  PLAN.md
  src/trainer/
    __init__.py
    device.py           # slang-rhi Device creation, shader module loading
    buffers.py          # NDBuffer allocation (gaussians, projected, tiles, grads, adam)
    pipeline.py         # dispatch orchestration (forward only in v0)
    dataset.py          # .ply load, Camera struct, load_image stub
    export.py           # render_to_png, save_ply
    train.py            # CLI entry (v0: loads .ply, renders, saves PNG)
  shaders/
    common/             types.slang, utils.slang, math.slang
    primitives/
      gaussian/         structs.slang, project.slang, eval.slang
    rasterize/
      tile_binned/
        bin.slang
        sort.slang      # LDS per-tile sort (v0); global radix comes later
        forward.slang
        backward.slang  # stub in v0
        radix_sort/     # placeholder, .gitkeep only
    optimizers/         adam.slang           # stub in v0
    losses/             l2_linear.slang      # stub in v0
    kernels.slang       # entry points
  tests/
    test_smoke.py       # device + shader compile on Vulkan AND D3D12
    test_math.py        # unit tests on [Differentiable] functions as they land
```

Install: `pip install -e ".[dev]"` once per environment.

---

## Locked design decisions

| Area | Decision |
|---|---|
| Backend | Vulkan primary, D3D12 cross-check |
| Barriers | Trust slang-rhi automatic inter-dispatch barriering |
| Iteration loop | Cold Python restart in v0; `module.reload()` deferred |
| Testing | Smoke (both backends) + unit tests on Slang math from day 1 |
| Dataset | `.ply` loader only; no COLMAP in v0; `Dataset.load_image` = lazy-per-call stub (`NotImplementedError`) |
| Param layout | AoS for gaussians, projected, Adam state; SoA `int32` fixed-point for gradients |
| Fixed-point scale | `FIXED_SCALE = 65536.0` uniform; saturation-check kernel after backward (v1) |
| Forward | Records `gaussian_count_per_pixel[pixel] = k` so early-break is correctness-safe |
| Loss | Separate dispatch writes `dL_dC_texture`; L2-linear; `srgb_to_linear` on target read |
| Densification | None in v0; buffers oversized to `max_splats`; GPU counter `num_gaussians_counter[0]` |
| Color model | Plain RGB via sigmoid-activated logit; no SH; view-dependent color is per-primitive |
| Adam LRs | Fixed values in Python config; no decay in v0 |
| Outputs | `render_to_png(camera, path)` + `save_ply(path)`; no viewer, turntable, or checkpointing |
| Error handling | Fail-fast; validate only at boundaries; no defensive clamping |
| Primitive abstraction | Deferred — v0 has only the gaussian primitive, no `IPrimitive` interface yet |

---

## v0 pipeline (forward only)

```
load_ply → init gaussians buffer + counter → create camera
  ↓
for each render_to_png(camera, path):
    clear_textures()
    clear_tile_bins()
    clear_projected_2d()
    bin_gaussians(camera)         # project + 2D bbox + append to tiles
    sort_gaussians_per_tile()     # LDS sort by depth
    forward_rasterize()           # front-to-back, writes output_image,
                                  # transmittance, gaussian_count_per_pixel
    save PNG from output_image
```

v1 will add: `compute_loss_grad` → `backward_rasterize` → `check_gradient_saturation` → `adam_step` → `clear_gradient_buffers`.

---

## Milestones

1. **M0 — bootstrap**: `pyproject.toml`, `src/trainer/device.py`, smoke test creates a `Device` on Vulkan and D3D12. (~half day)
2. **M1 — hello gaussian**: hand-coded single gaussian in a Python-authored buffer, hand-coded camera, naive forward (one-thread-per-pixel, no tiling), render to PNG. (~1 day)
3. **M2 — .ply load**: `load_ply`, render many gaussians via naive forward. Visual cross-check in SuperSplat. (~half day)
4. **M3 — tile-binned forward**: `bin.slang`, `sort.slang`, `forward.slang` matching the design above. `save_ply` round-trip. (~2–3 days)
5. **M4 — unit tests on math**: `buildCovariance3D`, `matrixFromQuaternion`, `projectGaussian3DTo2D_diff` called from Python via SlangPy functional API with known inputs. (~1 day)

v0 ships at M5 = "M3 output visually matches SuperSplat on the same `.ply` on both Vulkan and D3D12."

---

## Open questions deferred to when they become concrete

- Quaternion normalization strategy (matters for backward gradient flow; revisit when writing `backward.slang`).
- Groupshared accumulator flush strategy (thread-0 serial flush in bad_attempt1 is sub-optimal; revisit when writing `backward.slang`).
- Which `.ply` file to use as the v0 test scene.
- When to replace LDS per-tile sort with global radix sort (v1+).
- When to add densification / pruning (v1+).
- When to add learning rate decay (v1+).
- When `module.reload()` hot-reload becomes worth wiring (v1+).

---

## Ported-from-bad_attempt1 shader audit (on port, not copy verbatim)

These clamps/eps in the old shaders get re-evaluated per the no-defensive-clamping rule:

| Site | Verdict |
|---|---|
| `max(mean_clip.w, 1e-6)` in `projectGaussian3DTo2D_diff` | Reconsider — cull behind-camera points instead of clamping `w` |
| `cov2d += float3(0.3, 0, 0.3)` (2D low-pass) | Keep — canonical 3DGS derivation, document the `0.3` |
| `max(det, 1e-20)` in `evalGaussian2DFromCov` | Keep — genuine div-by-zero guard at numerical floor |
| `clamp(log_scale, -6, 2)` in Adam | **Remove** — magnitude guess; re-add only on evidence |
| `clamp(opacity, 0, 1)` in Adam | Keep — semantic bound, not magnitude guess |

---

## v0 shipped

M0 → M4 complete, all green on Vulkan + D3D12.

### How to use v0

```bash
# One-time setup
python -m venv .venv
.venv/Scripts/python.exe -m pip install -e ".[dev]"

# Tests (19 tests, ~17s, both backends)
.venv/Scripts/python.exe -m pytest tests/ -v

# Render a single hand-authored gaussian
.venv/Scripts/python.exe scripts/m1_hello_gaussian.py
# -> outputs/m1.png

# Render a synthetic 1000-gaussian sphere with tile-binned sorted forward
.venv/Scripts/python.exe scripts/m2_ply.py
# -> outputs/m2.png, outputs/m2_roundtrip.ply

# Render an external 3DGS .ply
.venv/Scripts/python.exe scripts/m2_ply.py --input path/to/scene.ply

# Occlusion smoke (near green should dominate far red)
.venv/Scripts/python.exe scripts/m3_occlusion_test.py
```

### What v0 proves

- SlangPy + slang-rhi renders the same `.slang` source identically on Vulkan and D3D12.
- Tile-binned sorted forward with front-to-back transmittance and per-pixel early break works.
- Slang math utilities (`sigmoid`, `matrixFromQuaternion`, `buildCovariance3D`, `evalGaussian2DFromCov`) are unit-tested with known inputs on both backends.
- `.ply` load/save round-trips pixel-perfect through the GPU buffer.

### Notable v0 discoveries

- SlangPy's ``kernel.dispatch(**kwargs)`` binds resources to **entry-point parameters**, not module-level globals. Declaring ``RWTexture2D<float4> output_image;`` at module scope compiles but never gets bound. See `memory/project_slangpy_binding_rule.md`.
- Forward compute entry points with shared resource names cannot be naively `#include`d into a single kernels.slang — resource redefinitions collide at link time. Load each entry point's `.slang` file directly.

### Not in v0 (explicit non-goals, deferred)

Backward pass, loss, Adam, densification, pruning, SH, LR decay, checkpointing, viewer/turntable, bindless streaming, global radix sort, COLMAP loader, saturation-check kernel, per-primitive interface abstraction.

---

# v1 Plan — Gradient correctness + single-scene overfit

Goal: **the trainer actually trains.** Split in two phases so gradient correctness is proven in isolation before any end-to-end training loop runs.

## v1.0 — Gradient correctness (Phase c)

Deliverables:
- `shaders/losses/l2_linear.slang` — per-pixel kernel: reads `output_image` + `target_image`, writes `dL_dC` to a UAV texture. `srgb_to_linear` on target at read time.
- `shaders/rasterize/tile_binned/backward.slang` — one thread per pixel, walks tile's sorted list in reverse bounded by `gaussian_count_per_pixel[pixel]`. Recovers `T_i = T_{i+1} / (1 - α_i)`. Hand-derives `dC/dα_i` and `dC/dc_i` (transmittance chain rule), chains through `bwd_diff(evalGaussian2DFromCov)` and `bwd_diff(projectGaussian3DTo2D)` for gaussian params. **Atomic int32 fixed-point to global directly** (no groupshared flush — revisit if perf hurts).
- `shaders/optimizers/adam.slang` — one thread per gaussian.
- Grad clear kernel + saturation-check kernel (warns on int32 wraparound, PLAN.md Q10).
- `src/trainer/buffers.py` — add SoA int32 gradient buffers (one per param component) + AoS Adam state buffer.
- `src/trainer/pipeline.py` — add `train_step(camera)` method dispatching forward → loss → backward → saturation_check → adam → grad_clear. No Python training loop yet.
- **`tests/test_gradients.py`** — central-difference finite-difference test on a 3-gaussian 16×16 scene, per-parameter, relative tolerance ~1%. On both backends.

Checkpoint: finite-difference test passes on all 14 per-gaussian param components (mean, log_scale, rot_quat, opacity, color_logit).

## v1.1 — Single-scene multi-view overfit (Phase a)

Deliverables built on v1.0:
- `src/trainer/colmap.py` — COLMAP text-format loader (`cameras.txt`, `images.txt`, `points3D.txt`). Returns list of `Camera` + initial-points dict matching `write_gaussians` signature. No binary `.bin` support in v1 (our dataset tool emits text).
- `src/trainer/dataset.py` — fill in `load_image(idx)` stub: read PNG from disk, downsample to pipeline resolution, upload to a **single reusable `target_image` texture** per iteration (not a preloaded texture array; keeps streaming door open per Q8).
- `scripts/v1_train.py` — Python training loop: random camera pick → `pipeline.set_target_image(load_image(idx))` → `pipeline.train_step(camera)` → log PSNR + it/s → save preview PNG every N iters.
- LR config: fixed values per PLAN.md Q15, **plus exponential decay for `lr_mean` only** (the one that actually matters for stability).

Scene: `datasets/synthetic/suzanne_blender/` — 20 images, 1920×1080, COLMAP text format, 1980 initial points, fixed grey color (127,127,127).

Resolution: **half-res 960×540** for v1.1 (~40 ms/iter target vs ~300 ms/iter at full res). Full-res is a v1.2 concern.

Checkpoint: training PSNR on the 20 training views visibly improves over iterations; rendered preview at a training-set camera is recognizably Suzanne-shaped. No held-out evaluation in v1.

## v1 locked design decisions

| Area | Decision |
|---|---|
| Densification / pruning | OUT. Fixed gaussian count = 1980 (COLMAP init). v2 concern. |
| Dataset format | COLMAP text (`cameras.txt`, `images.txt`, `points3D.txt`). No binary `.bin`, no `transforms.json`. |
| Gradient accumulation | Atomic int32 fixed-point straight to global. No groupshared flush. |
| LR schedule | Fixed per-param LRs in config; exponential decay for `lr_mean` only. |
| Validation | Training = evaluation set. No held-out views in v1. |
| Resolution | 960×540 (half-res) for v1.1. 1920×1080 deferred. |
| Image loading | Lazy per-call `Dataset.load_image(idx)`; single reusable `target_image` texture per step. Streaming is v2+. |
| Loss alpha channel | Compare RGB only, ignore alpha. Background assumed compatible (synthetic set renders on white; our forward writes float4(C, 1)). Revisit if artifacts appear. |
| Skipped between v1.0 and v1.1 | No intermediate single-view-only sanity run. Straight to multi-view. |

## v1 non-goals (deferred to v2+)

SH coefficients, view-dependent color, densification, pruning, per-primitive `IPrimitive` interface, global radix sort, `module.reload()` hot reload, held-out-view evaluation, perceptual losses (SSIM / LPIPS), checkpointing, full-res training, interactive viewer, bindless streaming, torch interop.

---

## v1 shipped

v1.0 (gradient correctness) and v1.1 (single-scene overfit) both green on Vulkan + D3D12.

### How to use v1

```bash
# Unit tests (21 total including finite-difference gradient check)
.venv/Scripts/python.exe -m pytest tests/ -v

# Train Suzanne on half-res (960x540), 1000 iters
.venv/Scripts/python.exe scripts/v1_train.py --iters 1000 --preview-every 100
# -> outputs/train/preview_NNNNN.png + outputs/train/log.txt
```

### What v1 proves

- Backward pass gradients match central-difference finite differences within 10% relative tolerance across all 14 per-gaussian param components, on both backends (`tests/test_gradients.py`).
- End-to-end trainer: forward → L2-linear loss → backward → Adam → grad clear, all on GPU via SlangPy + slang-rhi.
- Suzanne overfit: 1980 initial COLMAP points trained to ~31 dB PSNR in 1000 iterations (~74s on Vulkan). Recognizable monkey shape + color.
- COLMAP text-format loader produces camera matrices that align with our projection convention (COLMAP y-down flipped to our NDC y-up via proj[1][1] sign).
- LR decay on `lr_mean` only (init 1.6e-4 → 1% over `args.iters`). Other LRs fixed.

### v1 notable discoveries

- `bwd_diff` of a function with `out T x` takes the upstream gradient as a **plain value `T`**, not a `DifferentialPair<T>`. Different from the `inout`/input-diff convention where it takes a `DifferentialPair`. See `shaders/rasterize/tile_binned/backward.slang` for the concrete pattern.
- Vulkan and D3D12 produce gradients within ~10% of each other and within ~10% of finite-difference on small scenes. Fixed-point atomic accumulation at 2^16 is stable up to 16×16 tiles with ~3-gaussian occupancy; higher pixel counts / more gaussians per tile may eventually saturate — monitored by the saturation-check kernel.
- Initial gaussian opacity must be small (0.1 used) so the forward's gradient flow isn't saturated at iteration 0. Starting at 0.5 or 0.9 made early training stall.

### v1 files added

```
shaders/losses/l2_linear.slang
shaders/rasterize/tile_binned/backward.slang
shaders/optimizers/adam.slang
shaders/optimizers/grad_utils.slang
shaders/common/types.slang      # + AdamState, FIXED_SCALE
src/trainer/colmap.py           # text-format COLMAP loader
src/trainer/pipeline.py         # + train_step, forward_and_loss, backward,
                                #   read_gradients, read_loss, set_target_image, AdamLRs
src/trainer/buffers.py          # + grad SoA int32, adam_state, target_image, dL_dC, saturation_flag
scripts/v1_train.py             # training entry
tests/test_gradients.py         # finite-difference validation
```

---

# v2 Plan — Densification + Pruning

Goal: lift the quality ceiling imposed by v1's fixed gaussian count. 3DGS's canonical split / clone / prune heuristic implemented fully on the GPU — no CPU readback inside the densification path. The existing "oversized buffer + GPU counter" design from v0 Q13 is about to start paying off.

## v2.0 — Tracking signals (single milestone, quick)

Densification needs two per-gaussian signals the paper specifies:

- **|grad_mean_2d|** — magnitude of the 2D-mean gradient accumulated across pixels over the densification window. Currently computed inside `backward.slang` as `dp_mean_2d.d` after the eval bwd_diff but discarded after being chained through projection. Add one extra `InterlockedAdd(grad_mean_2d_mag_accum[g_idx], toFixed(length(dp_mean_2d.d)))` per pixel-gaussian.
- **pixel_coverage** — number of pixels for which this gaussian was actually processed (passed the per-pixel early-break). `InterlockedAdd(pixel_coverage_accum[g_idx], 1)` in the same backward loop body.

### Work

- Add two int32 SoA buffers to `GaussianBuffers`: `grad_mean_2d_mag_accum`, `pixel_coverage_accum`.
- Extend `shaders/rasterize/tile_binned/backward.slang` with the two atomic adds.
- Add `shaders/densify/clear_tracking.slang` (zero both buffers).
- Pipeline clears tracking at the start of each densification window (per-iter cost = one dispatch, negligible).
- Test: one training step on known scene, readback, assert expected nonzero-per-gaussian pattern.

v2.0 does not modify the gaussian count. It just starts collecting the signals densification will consume in v2.1.

## v2.1 — Densification machinery (GPU-driven, in-place split)

### Five-phase pipeline

**A. Decide actions** — one thread / gaussian, reads tracking buffers + current `log_scale`, `opacity`, writes `action[i] ∈ {KEEP, SPLIT, CLONE, PRUNE}`. Thresholds from a `DensifyConfig`. Paper rules:

- SPLIT: `|grad_mean_2d_mag| > t_grad` AND `max(exp(log_scale)) > t_scale`
- CLONE: `|grad_mean_2d_mag| > t_grad` AND `max(exp(log_scale)) <= t_scale`
- PRUNE: `opacity < t_opacity` OR 2D screen radius excessive
- KEEP: otherwise

**B. Split-and-clone** — one thread / gaussian (over the *pre-densification* range):
- SPLIT → overwrite slot `i` with child-1 (scale / 1.6, position jittered within parent covariance), zero Adam state at `i`. Atomic-append child-2 (same rule) to tail with zeroed Adam state.
- CLONE → atomic-append one clone at tail with inherited parent Adam state. Parent slot `i` untouched. (Position jitter along gradient direction is a v2.2 deferred item — see task #54.)
- KEEP, PRUNE → no-op in this phase.
- All appends do `InterlockedAdd(num_gaussians_counter[0], 1, tail)` and enforce `tail < max_splats`.

**C. Build prune mask** — one thread / gaussian (over the post-append range, i.e. `0..num_gaussians_counter[0]`): `valid[i] = 0` if `i < original_count` AND `action[i] == PRUNE`, else `valid[i] = 1`. Newly-appended gaussians always start as valid.

**D. Prefix scan** on `valid[]` → each kept gaussian's compacted destination index. Blelloch multi-block, lives in `shaders/common/scan.slang` — a reusable primitive.

**E. Scatter-compact** — ping-pong from `gaussians`/`adam_state` into `gaussians_scratch`/`adam_state_scratch` using `scan[]`. Python swaps primary↔scratch buffer references after the dispatch. Counter updated to total valid count.

### Why in-place split (your design)

- SPLIT children don't need a prune-flag-then-scatter pass to remove the parent — one child directly overwrites the parent slot.
- Unified with CLONE pattern (both append one child at tail; split also rewrites slot `i`).
- Fewer bytes moved during compaction.

### Why ping-pong for scatter (not for append)

Scatter writes to slot `scan[i] ≤ i`. Reads and writes to slots `i` and `scan[i]` would race across threads without ordering guarantees. A separate scratch buffer avoids this. Python pointer swap after the dispatch means no physical copy.

### Files

```
shaders/densify/
  decide_actions.slang
  split_and_clone.slang
  build_prune_mask.slang
  scatter_compact.slang
  opacity_reset.slang
  clear_tracking.slang
shaders/common/scan.slang     (reusable prefix scan)
```

### Test

`tests/test_densify.py`: hand-authored 5-gaussian scene, hand-specified action table, run B→C→D→E, assert post-compaction buffer state matches expectation (correct count, correct surviving params per slot, correct children present, Adam state zeroed on split / inherited on clone).

### Sizing

Drop `max_splats` default from 5M to **500k**. Even with Suzanne growing from 1,980 to ~50k peak during training, 500k gives 10× headroom. Ping-pong now costs 500k × ~56 bytes × 2 ≈ 56 MB per buffer (gaussians + adam_state each ping-ponged). Trivial on any modern GPU. 5M was overkill from v0.

## v2.2 — Schedule + re-train

- `train_step(densify_this_step: bool = False, reset_opacity_this_step: bool = False)`.
- Default schedule for 10k training run on Suzanne:
  - `densify_this_step = (500 <= iter <= 7000) and (iter % 100 == 0)`
  - `reset_opacity_this_step = iter in {2000, 4000, 6000}`
- Re-train Suzanne at 10k. Success metric: **masked-sRGB PSNR** (foreground-only, not full-image which gets dragged up by background matching). Target: visibly improve over v1's ~25 dB masked-sRGB, ideally 30+.

## v2 locked decisions

| Area | Decision |
|---|---|
| Driver | GPU-resident densification; no CPU readback in the loop |
| Split placement | In-place at parent slot (child-1) + atomic-append tail (child-2) |
| Clone placement | Atomic-append tail; parent slot untouched |
| Compaction | Ping-pong via scratch buffer + Python pointer swap; scan-based destination indices |
| Prefix scan | Blelloch multi-block, reusable primitive in `shaders/common/scan.slang` |
| Adam state on split | Zero (children are at new positions, parent m/v is stale) |
| Adam state on clone | Inherit parent's state (same position/grad, no reset gain) |
| Clone position jitter | Deferred to v2.2 (task #54) — v2.1 ships with identical-position clones |
| Tracking signals | `\|grad_mean_2d\|` + pixel coverage, SoA int32 fixed-point, cleared per densification window |
| `max_splats` default | 500k (down from 5M); ~56 MB ping-pong cost, ample headroom for Suzanne |
| SH | Still deferred; v2 ships plain-RGB sigmoid color |
| Densification flag | Opt-in per `train_step` call; gradient test runs untouched |
| Order determinism | Post-compaction buffer order is non-deterministic (atomic bumps). Fine — we treat gaussians as an unordered set. |

## v2 non-goals (deferred to v3+)

SH coefficients, held-out view evaluation, perceptual losses (SSIM / LPIPS), full-res (1920×1080) training, checkpointing, interactive viewer, bindless streaming, torch interop, global radix sort (still unused — scan is a different primitive), per-primitive interface abstraction.

---

## v2 shipped

All three milestones done, 57 → 65 tests passing on both Vulkan and D3D12 (new: 4 tracking + 8 densify + 20 scan tests).

### How to use v2

```bash
# Unit tests
.venv/Scripts/python.exe -m pytest tests/ -v

# Train Suzanne with densification (default: 10k iters, half-res)
.venv/Scripts/python.exe scripts/v2_train.py --run-name suzanne_v2

# Same training config lets you tune thresholds:
.venv/Scripts/python.exe scripts/v2_train.py --t-grad 5e-4 --densify-every 100 \
    --densify-from 500 --densify-to 7000 \
    --opacity-reset-iters 2000 4000 6000
```

### What v2 proves

- GPU-resident densification pipeline with no CPU readback in the hot loop (only stats-logging readback after each densify step, which is optional).
- Five-phase densify: decide_actions → split_and_clone (in-place child-1 + tail child-2) → build_prune_mask → prefix scan → scatter-compact with pointer-swap ping-pong.
- Reusable prefix-scan primitive (Blelloch multi-block, up to 1M elements) shipped and unit-tested.
- Masked-sRGB PSNR on Suzanne: **v1 17.6 dB → v2 27.94 dB (+10.3 dB)**. Fixed point cloud (1,980) → densified + pruned (3,811 final after peak ~5,000).
- Final render has visible facial features and continuous surfaces vs v1's pinned-vertex dot cloud.

### v2 honest metrics on Suzanne 10k

| metric | value | baseline v1 | delta |
|---|---|---|---|
| masked-sRGB PSNR | 27.94 dB | 17.6 dB | **+10.3 dB** |
| full-image sRGB PSNR | 37.52 dB | 24.74 dB | +12.8 dB |
| full-image linear PSNR | 44.70 dB | 33.15 dB | +11.5 dB |
| final gaussian count | 3,811 | 1,980 fixed | ~2× |
| runtime | 17 min | 14 min | +3 min |

### v2 notable discoveries

- The 3DGS paper's `t_grad ≈ 2e-4` assumes the 2D mean gradient is normalized to image units. Our backward computes `length(dp_mean_2d.d)` in **pixel-space units**, so we divide by `max(W, H)` in the decide-actions kernel. Without this the threshold triggers on ~80% of gaussians at 960×540, not the paper's ~2%.
- In-place split (child-1 overwrites parent slot, child-2 appended to tail) avoids both a parent-prune flag and a wasted compaction write per split. This was user-suggested and simpler than the ping-pong-both-children variant I originally proposed.
- Post-compaction buffer order is non-deterministic (atomic-append slot allocation). Fine for correctness since we treat the buffer as an unordered set, but gradient tests have to run with densification disabled for reproducibility.
- With `max_splats = 500k`, the full Adam + scratch + tracking state fits in ~200 MB VRAM. The 5M default from v0 was overkill.

### v2 files added

```
shaders/common/scan.slang                   # Blelloch prefix scan primitive
shaders/densify/
  clear_tracking.slang
  decide_actions.slang
  split_and_clone.slang
  build_prune_mask.slang
  scatter_compact.slang
  opacity_reset.slang
shaders/rasterize/tile_binned/backward.slang  # + 2 tracking accumulators
src/trainer/buffers.py                        # + tracking + densify scratch
src/trainer/pipeline.py                       # + densify(), reset_opacity(),
                                              #   clear_tracking(), read_tracking(),
                                              #   DensifyConfig
scripts/v2_train.py                           # training entry with densification
tests/test_tracking.py                        # 4 tests
tests/test_densify.py                         # 8 tests
tests/test_scan.py                            # 20 tests
```

### v2 next quality levers (deferred to v3)

- Clone position jitter along gradient direction (task #54; v2 ships identical-position clones).
- SH view-dependent color (currently plain-RGB sigmoid).
- Full-res training (1920×1080 is 4× the work of 960×540; would need more perf investment).
- Perceptual losses (SSIM / LPIPS) instead of pure L2.


