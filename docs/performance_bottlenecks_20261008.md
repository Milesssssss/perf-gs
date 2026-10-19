# Performance investigation — 2026-10-08

## Scope and method

**Report only. No production Python or Slang files were changed.** Diagnostic
overrides were process-local and restored; production source fingerprints still
match the preceding full-training comparison.

- RTX 4080, 16,376 MiB VRAM; about 6.7 GiB device memory already used before the
  investigation, compared with approximately 13 GiB during the earlier slow run.
- Bicycle, 194 cameras, 1237×822, seed 0, SADGS camera order, Mip enabled, NVIDIA
  global STBN fraction 0.1. Headless, evaluation/log/preview rendering disabled.
- **3M capacity**, not the default 10M, to avoid deliberately overcommitting VRAM.
  Mid-training captures at iteration 1560 had about 1.64M live splats. Capacity
  limits affect the trajectory, so these are not default-10M performance numbers.
- GPU timestamp samples: 48 repeated fixed-camera, zero-LR steps for SH0 and
  SH3. CPU profile: 64 additional steps. These diagnostic timings are not full-run
  throughput measurements; CPU tracing and per-sample waits add overhead.
- A complete 3,000-iteration SH3 run was also sampled every 50 steps, including
  the separate mutation submits. Its profiled wall time is not normal throughput.
- Candidate measurements below use identical real inputs or explicitly labelled
  synthetic inputs. They are experiments, not production changes or claims of
  end-to-end improvement.

## What currently costs time

Median GPU time at the captured SH3 state:

| Non-overlapping work | ms |
|---|---:|
| SH-rest Adam | 1.24 |
| Forward radix sorting | 1.15 |
| Raster backward **including projection backward** | 0.93 |
| Image loss | 0.75 |
| Forward rasterization | 0.71 |
| SH forward | 0.61 |
| Main Adam | 0.60 |
| SH backward | 0.39 |
| Pair emission | 0.35 |
| Cluster culling | 0.21 |
| Six gradient-seed clears | 0.20 |

The non-overlapping root scopes sum to approximately 7.7 ms for SH3 and 5.3 ms
for SH0 at this particular camera/state. CPU command recording is approximately
2.4 ms for SH0 and 2.9 ms for SH3 in the CPU profile; it normally overlaps GPU
execution. Binding/dispatch dominates that CPU time, not camera-selection RNG.

The full SH3 profile's late window confirms the pattern: SH forward averages
1.17 ms, SH-rest Adam 1.36 ms, radix 0.96 ms, loss 0.74 ms. Backward is important,
but is **not the only or necessarily largest optimization target** with sparse
supervision. No hardware-counter capture was made, so precise claims about
occupancy, register spilling, or individual stall reasons remain unverified.

Current 3M-capacity mutation measurements:

- Structure ticks: about **82 ms mean host completion**; SAD GPU scope about
  14.7 ms, Morton 12.1 ms, Mip 11.8 ms.
- Filler ticks: about **49 ms mean host completion**.
- The historical 1–3-second pauses were not reproduced with this VRAM-fit
  workload. That does not prove they are fixed at the original configuration.

## Ranked programming/system opportunities

### 1. VRAM budgeting and the independent viewer — highest system priority

The default capacity is not a small reserve. Running the production allocator
with a counting factory and real reflected strides gives these **requested
allocations**, excluding driver/runtime overhead:

| Bicycle at 1237×822 | Pipeline + all GT/structure textures |
|---|---:|
| 3M capacity, SH0 | 4.48 GiB |
| 10M capacity, SH0 | 9.59 GiB |
| 10M capacity, SH3 | **16.97 GiB** |

The SH0 viewer adds about **1.75 GiB** at 10M capacity; SH adds further storage.
SH3/10M already exceeds this GPU's physical capacity before the viewer, other
applications, and driver overhead. The shared arena and SH-rest Adam moments
each request **3.6 GB** at that setting. All 194 target/structure textures add
about **2.20 GiB**, even though only 169 cameras are trained.

**Suggested first actions:** expose a preflight allocation budget, measure the
peak live/pair counts, right-size reserves with headroom or explicit growth, and
avoid the second viewer allocation when benchmarking. Do not silently lower
capacity: clipping changes the trained model. Allocate only required optional
features and training-camera GPU textures. Bounding the SoA SH-moment arena by
independent components/slabs is a possible programming-level memory reduction,
but has a dispatch/transfer trade-off and needs a separate benchmark.

Severe residency pressure is a plausible explanation for second-scale stalls;
allocation arithmetic and observed slowdown support investigating it first,
but GPU page-fault/residency evidence would be needed to prove paging itself.

Source: [capacity-derived pair allocation](src/trainer/backend/render/pipeline.py#L123),
[buffer allocation](src/trainer/backend/memory/buffers.py#L652),
[SH state and scratch](src/trainer/backend/memory/buffers.py#L723),
[all-camera texture preparation](src/trainer/frontend/app.py#L1025).

### 2. Viewer synchronization and ineffective render throttling — small fixes

Every snapshot calls `live_splat_count()`, which performs a synchronous counter
readback under the shared queue lock. Default snapshot cadence is every ten
iterations. Small transfer size does **not** mean small synchronization cost.
An isolated probe adding just this readback cadence increased fixed-step time
from 5.45 to 5.69 ms, approximately **4.3%**; that excludes snapshot copies and
the viewer's rasterization, so it is not a complete viewer A/B.

Also, every snapshot marks the scene dirty, and `_scene_due()` lets dirty updates
bypass the advertised 15 FPS cap. A runtime check returned `True` after only
1 ms since the preceding render. Model-update dirtiness should be coalesced
behind the time budget, while camera/user interactions can remain responsive.

**Suggested fix:** use the GPU-resident count for snapshot copies/counter transfer
instead of a host round-trip; coalesce model updates and enforce the render budget.
Training math need not change. Headless runs are unaffected by this opportunity.

Source: [count readback and snapshot](src/trainer/frontend/viewer/viewer.py#L527-L550),
[dirty bypass](src/trainer/frontend/viewer/viewer.py#L604-L611).

### 3. Stop sorting constant key bits — measured low-risk opportunity

Forward pair sorting always performs eight byte passes: four depth bytes and
four tile-ID bytes. This frame has **15,965 tiles**, so two high tile-ID bytes
are always zero. Six passes suffice without changing the ordering contract.
Morton sorting uses 63 significant bits and must retain its separate full path.

Interleaved probe on **3,278,310 identical real unsorted pairs**:

| Forward sort | GPU ms | Host recording ms |
|---|---:|---:|
| Eight passes | 0.956 | 1.256 |
| Six passes | 0.737 | 1.026 |

**23% less sorter GPU time**, with bit-exact keys and values in this probe.
This is roughly 0.22 ms of GPU savings in that workload, not a 23% training win.
Pass count and ping-pong ownership must be handled for all frame sizes, and full
sort correctness/parity tests must run before adopting it.

Source: [fixed eight-pass loop](src/trainer/backend/render/pipeline.py#L774),
[key layout](shaders/rasterize/tile_binned/bin_emit.slang#L1-L11).

### 4. SH forward evaluates invisible splats — measured, straightforward

`shForward()` checks live count, but not this camera's existing tile-count/cull
result. It evaluates SH and writes blend records for every live splat even when
that blend record will never be consumed.

In a real captured view, **69% of live splats emitted no tile pairs**. A diagnostic
in-memory visibility gate reduced the isolated pass from **0.559 to 0.189 ms**,
about **66% less pass time**. Every visible blend record matched bit-for-bit.

**Suggested fix:** bind the already-produced tile counts and skip zero-pair
splats in SH forward. Preserve all visible shading/clamp behavior. Do **not**
copy this predicate blindly into Adam: a zero-gradient parameter can still
update from optimizer momentum, so that would change training semantics.

Source: [SH forward gates](shaders/rasterize/tile_binned/sh_forward.slang#L30-L54),
[existing tile-count result](shaders/rasterize/tile_binned/bin_count.slang#L45-L64).

### 5. Live-count work and fused clears — small to medium implementation effort

- Six seed buffers are cleared at full capacity each step: **144 MB at 3M**,
  **480 MB at 10M**. Only live slots need these zeros. Their measured cost is
  about 0.20 ms at 3M, plus six host kernel bindings.
- Cluster AABB groups execute their reductions even for entirely empty capacity
  clusters. Projection backward, SH stages, pair emission/range identification,
  and score dispatches also launch up to maximum capacity and then early-out.
- The tile-offset scan always processes `max_splats`, even if most entries are
  zero. This costs about 0.10–0.13 ms at 3M. A GPU-counted multi-level scan is more
  than a one-line change, but existing indirect-dispatch infrastructure can help.
- `accumulate_iter` reads two full-capacity int64 arrays every step. After the
  final densification window, tracking and its two extra backward atomics are
  still computed even though no further densification consumes them. Disabling
  unused tracking must preserve zero/initialization behavior and diagnostics.

**Suggested order:** combine/live-bound seed clears, skip empty cluster groups,
then use GPU counts for remaining launches/scans. Ensure new children receive
clean seeds, tails are not scanned as valid, and no host count readback is added.
At the measured 3M capacity these are modest fractions of step time; at 10M with
few live splats the waste is larger, but linear extrapolation under paging is unsafe.

Source: [hot seed clears](src/trainer/backend/render/pipeline.py#L1000-L1007),
[full-capacity scan](src/trainer/backend/render/pipeline.py#L878-L885),
[empty-cluster reductions](shaders/rasterize/cluster_cull.slang#L49-L83),
[always-on tracking](src/trainer/backend/render/pipeline.py#L1222-L1233).

### 6. Loss implementation, not loss removal — worthwhile second batch

SSIM still uses two direct 11×11 neighbourhood loops. The measured passes cost
approximately **0.435 ms maps + 0.315 ms gradient** at this resolution and budget.
Separable convolution and/or tile-shared halo reuse are implementation-level
options that retain the loss, with floating-point/parity validation required.

The maps **cannot simply be sparsely skipped**: a selected pixel's SSIM gradient
depends on neighbouring window centers, including unselected ones. Adaptive
sampling runs this loss dense by design, making efficient convolution more useful.

Adaptive-only synthetic probe: about **0.147 ms GPU** and **0.344 ms host recording**
for all eight extra kernels. `solveLambda` costs about **0.035 ms**, despite its
60 iterations and hundreds of group barriers. It has local optimization options,
but it is not the primary bottleneck in the measured default-disabled feature.

Source: [SSIM map loop](shaders/losses/ssim_loss.slang#L33-L60),
[required dense-map dependency](shaders/losses/ssim_loss.slang#L98-L107),
[adaptive solver](shaders/losses/adaptive_importance.slang#L563-L586).

## Measurement/system hygiene fixes

- `projection_backward` is nested **inside** `rasterize_backward`, while the stage
  map adds both. Stage totals therefore double-count that interval. Compare root
  scopes or subtract the nested projection interval; do not add every marker.
- Sampling every 50 or 25 iterations aligns with structure scoring every ten,
  overrepresenting scoring work. Report conditional scoring cost or stratify the
  sample cadence; the score pass is not paid on every ordinary iteration.
- Frozen CPU profiling showed the 144-byte camera upload costs about 0.13–0.16 ms
  in host API/packing overhead. Prepacked static camera buffers can reduce host
  work, but removing uploads in a fixed-camera probe did not give a repeatable
  standalone throughput gain. It is lower priority than the measured GPU waste.
- Disabling debug markers alone did not provide a clear throughput win. Preserve
  useful profiling structure rather than deleting it on speculation.
- RNG/context setup is tiny: STBN host sampling averaged roughly 0.016–0.028 ms.
  Avoid spending the next pass on cosmetic Python micro-optimization.
- Main Adam **already uses indirect dispatch** and clears parameter gradients
  itself. The fourteen-parameter full-clear helper is setup/test work, not an
  every-iteration clear. Neither is a new optimization target.
- No repeated GT or structure-tensor uploads occur in ordinary training. Target
  selection is a reference swap. Setup does build fields for held-out cameras,
  and the compressed-cache loader reads the full `fields` member twice then
  copies it; read once and retain the array when that cache path is used.
- The old frozen-iteration launcher currently calls a missing `benchmark_frozen`
  method. Repairing profiling tooling and separating viewer/evaluation/setup from
  training time should precede performance claims based on that launcher.

## Recommended next batch, pending approval

1. VRAM preflight/budget visibility and viewer readback/throttle cleanup.
2. Forward-only significant-byte radix passes and visible-only SH forward.
3. Fused/live-count seed clears and empty-cluster work elimination.
4. Profile and optimize the SSIM convolution implementation, without changing loss.
5. Only then investigate larger Adam/binding/backward-layout changes with hardware
   counters. No recommendation to change densification thresholds, sampling density,
   SH degree, rendering equations, or optimizer mathematics.

The two measured GPU candidates save about **0.59 ms combined in their isolated
probes**. That suggests a several-percent opportunity in a healthy SH3 workload;
it is not yet a measured combined end-to-end speedup. Under VRAM overcommit,
system fixes could matter much more than either shader micro-optimization.

## Evidence artifacts

- [SH0 capture and CPU profile](scripts/debug/performance_profile_sh0_20261008.json)
- [SH3 capture and CPU profile](scripts/debug/performance_profile_sh3_20261008.json)
- [Full-training sampled stages and mutations](scripts/debug/performance_full_stage_report_20261008.json)
- [Requested allocation accounting](scripts/debug/performance_allocation_report_20261008.json)
- [Constant-bit sort probe](scripts/debug/performance_sort_constant_bits_probe_20261008.json)
- [SH visibility probe](scripts/debug/performance_sh_visibility_probe_20261008.json)
- [Host recording/readback probes](scripts/debug/performance_host_probe_20261008.json)
- [Adaptive-only synthetic probe](scripts/debug/performance_adaptive_probe_20261008.json)