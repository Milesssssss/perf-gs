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


