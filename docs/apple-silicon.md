# Apple Silicon (Metal)

The trainer's shaders stay in Slang. On Apple Silicon, Slang targets Metal through slangpy's Metal device (`--backend metal`). The training loop, presets, densification, Adam, loss, and stop rule are the same code as the Vulkan and D3D12 runs.

## What is selected

| Host | Default `--backend` |
| --- | --- |
| Apple Silicon macOS (`arm64`) | `metal` |
| Intel Mac, Linux, Windows | `vulkan` |

An explicit `--backend vulkan`, `--backend d3d12`, or `--backend metal` always wins. Preset flags do not change: `--preset ultrafast`, `balanced`, `quality`, and `quality+` keep their official meaning. For Quality+, `--iters 0` (the preset default) still trains until the training-view PSNR plateau, and a positive `--iters N` is still a fixed step budget that stops with `iteration_limit`.

`--profile-run` is the Nsight Graphics capture path and still forces Vulkan. It is not a Metal training mode.

## Install

The Metal sources were checked with **Slang 2026.5.2**, which ships in **slangpy 0.42.0**. That release publishes macOS wheels for Python 3.9–3.13 tagged `macosx_15_0_arm64`, so the known-good stack is:

- Apple Silicon Mac
- macOS 15 or newer (the wheel tag)
- Python 3.11 (same minor version as the published Linux runs; 3.12 also has a wheel)
- Xcode Command Line Tools, which provide Apple's `metal` compiler (`xcode-select --install`)

From the repository root:

```sh
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e ".[dev]" "slangpy==0.42.0"
```

`slangpy==0.42.0` is pinned on purpose. A newer slangpy may retarget Metal differently; the trainer prints a warning on macOS when the installed slangpy is not 0.42.0.

## Train

The same commands as the other backends. Metal is the default on this machine, so it does not need to be repeated:

```sh
python -m trainer.frontend.app --scene datasets/tandt/truck --no-viewer --save-ply
python -m trainer.frontend.app --scene datasets/360_v2/room --preset quality+ --no-viewer --save-ply
python -m trainer.frontend.app --hub
```

`--save-ply` still writes `outputs/train/<run-name>/output.ply`. `--run-name`, `--downsample`, and the indoor default (downsample 2 for indoor Mip-NeRF 360 scenes, 4 otherwise) are unchanged. The interactive hub uses the same device selection. `scripts/debug/view_ply.py` does too.

To force Vulkan (MoltenVK) on a Mac that has it:

```sh
python -m trainer.frontend.app --scene datasets/tandt/truck --backend vulkan --no-viewer --save-ply
```

## What Metal changes inside the shaders

The Slang sources are shared. Two lowering details exist only so those sources compile to Metal and keep the same integer results:

- Gradient accumulators stay int64 fixed-point. Vulkan and D3D12 still use one 64-bit `InterlockedAdd`. Metal has no 64-bit atomic add on any Apple GPU, so Metal adds the same two's-complement value as two little-endian 32-bit atomics plus the carry from the low half. Integer addition does not depend on order, and no kernel reads an accumulator it is still updating.
- `WavePrefixCountBits` is not available in Slang's Metal compute target. The structure-densify reservation uses it. Metal builds the same exclusive popcount from `WaveActiveBallot` (lane 0 is the low bit). Vulkan and D3D12 still call `WavePrefixCountBits`. That kernel also keeps every lane in the simdgroup alive until the wave ops, because Metal simdgroup instructions are undefined if a lane has already returned. Lanes past the live splat count contribute "not selected" and zero growth, which is what an inactive lane contributed before.
- Kernels that use wave builtins cannot take loose `uniform` scalars. Slang 2026.5.2 then emits those scalars as a by-value struct on the Metal kernel, and Apple's `metal` compiler rejects it (`uint` cannot be used in a struct of resources). The radix sort (`upsweep`, `scan`, `downsweep`), per-splat backward, 3D mip filter, and structure-densify decision pass the same numbers in a `ConstantBuffer`, which Slang emits as `constant T* [[buffer(N)]]`. Vulkan and D3D12 receive that same constant buffer through slangpy. Kernels that do not use wave ops still use ordinary uniforms.
- slangpy 0.42.0's Metal command recorder accepts `dispatchComputeIndirect` and then ignores it (`IComputePassEncoder::dispatchComputeIndirect is not supported`). Every indirect launch in the trainer (radix sort upsweep/downsweep, tile binning, per-bucket backward, Adam, densify/split/compact, mip filter, active-copy) goes through one helper. On Metal that helper submits the work that wrote the three group counts, reads those counts, and dispatches the same kernel with the same group counts. Vulkan and D3D12 still use the hardware indirect path. The extra readback makes Metal steps slower; it does not change the kernels or the counts.

Apple GPUs use 32-wide simdgroups, which matches the rasterizer's 32-splat bucket. The 64-lane backward path is unchanged and simply does not split a bucket on these GPUs.

## Checks that do not need a Mac

With the Slang compiler that matches slangpy 0.42.0 (`slangc` on `PATH`, or `SLANGC=/path/to/slangc`):

```sh
python -m pytest -q tests/test_backend_select.py tests/test_fixed_atomic.py tests/test_metal_shaders.py tests/test_cli_defaults.py
```

`tests/test_metal_shaders.py` compiles every compute entry to Metal (balanced/quality defines, the ultrafast STBN define, and the optional splat-kernel / strided-backward defines) and to SPIR-V. It fails if Metal emits a 64-bit atomic or a buffer binding above slot 30. GPU tests still skip when their device is missing, including Metal on a machine with no Metal runtime.

## Not yet run on a Mac

This port was not trained on Apple hardware. Before trusting a quality number:

- slangpy will ask Apple's `metal` compiler to turn the generated MSL into a metallib. The known rejection (a by-value uniform struct on wave kernels, including radix-sort `upsweep`) is fixed in the generated MSL: those uniforms are `constant T* [[buffer(N)]]`. Apple's compiler was not run in this environment, so a short training step is still the confirmation.
- A short real scene (`--preset balanced --iters` a few hundred, `--no-viewer --save-ply`) is the first useful check that indirect dispatch, read-write textures, and the int64 emulation run.
- Held-out PSNR against a Vulkan run of the same scene, seed, and preset is the check that the sums match in practice. Nothing here replaces that.
- The hub window and `view_ply.py` use slangpy's Metal surface. That path is in the macOS slangpy 0.42.0 library and was not opened here.
- There is no DXGI memory budget on macOS, so `--vram-budget-gib 0` does not clamp `--max-splats` from an OS query. Pass a budget if you need one.
