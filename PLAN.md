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

