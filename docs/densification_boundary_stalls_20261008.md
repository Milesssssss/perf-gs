# Densification-boundary stalls — focused investigation, 2026-10-08

## Conclusion

The visible pause at a densification-aligned iteration is **not necessarily the
time spent in the split/prune kernels**. It also includes first-use native API
cost, synchronization with previously queued iterations, Morton permutation,
Mip filtering, and possibly viewer/evaluation work. Asynchronous mutation work
can make the stall appear several iterations **after** the labelled boundary.

The current environment exhibited large stalls even at 600K capacity, with no
viewer and no evaluation. No production Python or Slang changes were applied.

## What was measured

Two bounded bicycle probes, 618×411, SH0, 600K capacity, STBN 0.1, seed 0, SADGS
camera order, position-LR horizon 3000. The first ran 600 iterations with CPU
phase/fence timings. The second ran 1100 iterations and recorded GPU timestamps
at ticks 100, 200, 500 and 1000. Timestamp results were read only after the normal
final drain; no extra per-tick waits were inserted. Both deliberately avoided
allocating another default-10M trainer into the already memory-heavy environment.

These are diagnostic boundary traces, not benchmark claims for the original
configuration. GPU intervals under concurrent load may include scheduling delays;
they are not hardware-counter measurements of pure shader-active time.

### CPU-only boundary trace

At launch the device reported **14,830 / 16,376 MiB used**. Highlights:

| Location | Host duration | Observed blocking |
|---|---:|---|
| Step 100, first filler boundary | 207 ms | 192 ms inside mutation submission |
| Step 500, first structure boundary | **670 ms** | 512 ms fence wait inside structure densify |
| Step 504 | **388 ms** | 385 ms queue-backpressure wait |

The step-504 wait is important: it is invisible to timing just the controller at
step 500, but can be perceived as part of the same pause by the viewer/logs.

### CPU + GPU boundary trace

At launch the device reported **13,945 MiB used**; after the run, 15,249 MiB.

| Tick | CPU whole iteration | CPU mutation body | GPU SAD/filler | GPU Morton | GPU Mip |
|---|---:|---:|---:|---:|---:|
| 100 | 228 ms | 199 ms | 0.19 ms | 4.23 ms | 0.45 ms |
| 200 | 32 ms | 3.50 ms | 0.24 ms | 10.41 ms | 1.87 ms |
| 500 | 223 ms | 188 ms | 4.31 ms | 38.57 ms | 8.31 ms |
| 1000 | 396 ms | 346 ms | **2.88 ms** | 31.46 ms | 5.77 ms |

At tick 1000, `densify_structure_aware()` took 343 ms on the host, of which
**340.65 ms was `wait_for_submit()`**. The native command recording was a small
remainder. A fence for the structure submit must also wait for preceding submits
on the queue, including earlier training work and work from the shared device.
It is therefore incorrect to call the entire 343 ms "structure kernel time."

### First-use native costs

The first filler and structure boundaries also had significant native dispatch
costs before their normal fence wait:

- First `split_and_clone` indirect dispatch: about **40 ms** host.
- First structure decision dispatch: about **37 ms** host.
- First anisotropic split dispatch: about **42 ms** host.
- First Mip indirect dispatch: about **30 ms** host.

At the next filler tick, total mutation recording was only about 3.5 ms. This
supports a **first-use dispatch/binding/pipeline-cache cost**, not a Python loop
over splats. The exact split between shader marshalling, reflection, driver
pipeline creation and residency is not established by these API timings.

## System memory evidence

A separate Windows adapter-memory snapshot, outside the probe process, showed:

- Dedicated GPU memory: **13,963.65 MiB**.
- Shared GPU memory backed by host RAM: **7,973.85 MiB**.

The snapshot establishes substantial system GPU-memory use, but does not identify
which process owns those allocations or prove that this trainer's pages faulted.
Earlier allocation accounting shows default SH3/10M trainer + scene textures
request about **16.97 GiB**, before its independent viewer and other applications.
That configuration cannot keep all requested resources resident on a 16GB card.

## Why densification is especially exposed to this

Ordinary training culls work to the current view. Densification and Morton sorting
move the **whole live population**, including optimizer state that a given view
may not have touched recently. Their shared scratch arena is another large
resource. This expands the touched working set abruptly and is particularly
expensive if some of those resources reside outside VRAM.

The current controller performs two immediate full-state transformations:

1. Split/prune, then scatter into the arena and copy back.
2. Morton sort, then scatter the same state into the arena and copy back again.

For SH3, state carried per live splat is:

| Resource | Bytes per splat |
|---|---:|
| Gaussian | 56 |
| Main Adam | 112 |
| Packed SH rest | 92 |
| SH-rest Adam | 360 |
| **Total** | **620** |

One scatter + copy-back reads and writes that state twice: approximately
**2480 bytes of logical memory traffic per survivor**, excluding index/eta buffers
and split writes. Two transformations mean roughly **9.92 GB at 2M splats**, or
**49.6 GB at 10M splats**. These are logical GPU read/write volumes, **not measured
PCIe traffic**. Actual traffic depends on caches, residency and population size.

Thus, a multi-second boundary can plausibly be dominated by residency and broad
data movement even when the split arithmetic itself is cheap. The 600K probes
did not reproduce a full 1–3-second individual structure tick, so that precise
original symptom remains unconfirmed.

## Other work that aligns with the boundary

- **Viewer:** default snapshots every ten iterations include a synchronous count
  readback **before** `_apply_mutations()`. Every densification tick aligns with
  that cadence. The readback holds the shared queue lock while completing.
- **Structure telemetry:** two scalar readbacks force a wait on each structure
  tick. Large eta/percentile readbacks are now opt-in via `--densify-diag`.
- **Next iteration:** Morton/Mip normally run asynchronously; queue backpressure
  can charge their completion to following iterations rather than the boundary.
- **Observations:** custom log/eval/PSNR cadences may align too. These were disabled
  in both probes, so they are not needed to explain the measured stalls.
- **Console interpretation:** throughput lines are printed every ten iterations,
  not every single step. They represent a ten-step interval, including whichever
  synchronization happened inside it.

No image/structure-tensor uploads or new capacity-sized buffer allocations were
found in the normal densification hot path. Mip camera uploads happen at setup;
the per-tick all-camera calculation is on the GPU, not a Python camera loop.

## Targeted next actions, pending approval

1. **Warm first-use mutation pipelines before timed training** using isolated
   scratch/dummy state. This targets the 100/500 first-use submission spikes;
   it does not cure persistent memory pressure.
2. **Remove forced synchronous telemetry from the structure tick** by deferring
   scalar diagnostics to a completed submission or asynchronous readback. Keep
   warnings, but do not break host/GPU overlap solely for them. This reduces an
   artificial boundary drain; actual GPU work still has to complete somewhere.
3. **Fuse compaction and Morton into a single final permutation/commit.** Build a
   final survivor/Morton index map before moving Gaussian/Adam/SH state, avoiding
   the intermediate full-state move. This is a programming/dataflow change, not
   a proposal to change thresholds or skip Morton. It has moderate integration
   cost and must validate ordering, eta carry, optimizer moments and opacity cap.
4. **Address VRAM budgeting and viewer synchronization first for the original
   10M/SH3 case.** Otherwise shader improvements may simply be overwhelmed by
   residency delays. Do not lower capacity silently or stop unrelated GPU jobs.
5. Capture the original configuration when available with timings spanning
   **boundary − 5 through boundary + 6**, GPU-memory/residency counters and all
   pipeline phases. A controller-only timer misses important waits outside it.

## Evidence

- [CPU boundary trace](scripts/debug/densification_boundary_trace_20261008.json)
- [CPU/GPU boundary trace and dispatch times](scripts/debug/densification_gpu_boundary_trace_20261008.json)
- [Earlier whole-training mutation samples](scripts/debug/performance_full_stage_report_20261008.json#L51-L123)
- [Allocation accounting](scripts/debug/performance_allocation_report_20261008.json)
- [Controller compaction then Morton](src/trainer/frontend/densification.py#L98-L137)
- [Structure completion wait and telemetry](src/trainer/backend/densify/densify.py#L375-L406)
- [Next-iteration backpressure](src/trainer/frontend/queue_limiter.py#L43-L57)
- [Viewer count readback](src/trainer/frontend/viewer/viewer.py#L527-L550)