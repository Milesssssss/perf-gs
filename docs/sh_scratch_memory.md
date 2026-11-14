# Bounded SH moment scratch

SH Adam moments remain FP32 and retain their capacity-strided SoA layout.
Compaction and Morton sorting now process independent moment planes in bounded
batches instead of reserving scratch for the entire moment buffer.

## Allocation change

The shared arena is sized from reflected Gaussian/main-Adam strides and the
packed SH feature width. With the current layout, main Adam is largest at
112 bytes per allocated Gaussian. A moment plane needs four bytes per slot,
so that arena holds 28 planes at once.

SH3 contains 45 scalar higher-band coefficients and 90 moment planes: m at
plane `2*s`, v at `2*s+1`. The plane batches are 28 + 28 + 28 + 6. Each batch
is scattered from the original buffer into scratch, committed to its own live
plane prefixes, and only then is scratch reused. Neighboring planes and dead
tails are not overwritten. Counts remain on the GPU.

At **10,000,000 capacity slots**, decimal GB:

| Degree | Previous scratch | Bounded scratch | Saved |
| --- | ---: | ---: | ---: |
| SH0 | 1.12 GB | 1.12 GB | 0 |
| SH1 | 1.12 GB | 1.12 GB | 0 |
| SH2 | 1.92 GB | 1.12 GB | 0.80 GB |
| SH3 | 3.60 GB | 1.12 GB | **2.48 GB** |

The SH3 reduction is approximately 2.31 GiB, or 68.9% of the old scratch arena.
These are requested buffer sizes, not a measured reduction in process residency.
The separate persistent SH3 moment buffer is still 3.60 GB at this capacity.
Coefficients, gradients, the viewer, and the Gaussian growth limit are unchanged.

## Implementation

- [Allocation and arena ownership](../src/trainer/backend/memory/buffers.py)
- [Shared plane-batch command orchestration](../src/trainer/backend/memory/arena.py)
- [Offset GPU-counted commits](../shaders/densify/copy_active.slang)
- [Compaction kernels](../shaders/densify/scatter_compact.slang)
- [Morton kernels](../shaders/densify/morton_sort.slang)

The simple-densification service, structure-aware compaction, and Morton sorting
all call the same batching implementation. Original full-width scatter kernels
remain available for historical diagnostic references, which explicitly request
their larger arena; production does not allocate that reference scratch.

## Validation and timing

[Allocation and permutation tests](../tests/test_sh_scratch.py) run the production
allocator with metadata-only large buffers and exercise real GPU kernels on
Vulkan and D3D12 for SH1–SH3. They cover all planes, mixed/empty/full live counts,
tail preservation, exact raw uint patterns (including signed zero, subnormals,
and NaN payloads), and all three production reorder paths.
[Active-copy tests](../tests/test_active_copy.py) cover plane destination offsets,
partial final batches, and count clamping. Existing SH, Morton, structure
densification, command-ownership, and benchmark-harness tests also pass.

[The dataset-free benchmark](../scripts/debug/bench_sh_scratch.py) compares a full
arena and bounded arena on identical data, alternating their execution order.
Restoring input and checking parity are outside timing. Both arenas coexist in
the diagnostic; its total residency is not the production memory footprint.

One RTX 4080 Vulkan probe (250k capacity, 100k live, SH3, four warmups and twenty
samples per variant) measured:

| Moment-only operation | Full arena wall median | Bounded arena wall median | Full / bounded GPU medians |
| --- | ---: | ---: | ---: |
| Compaction scatter + commit | 1.002 ms | 1.370 ms | 0.637 / 0.593 ms |
| Morton scatter + commit | 0.942 ms | 1.317 ms | 0.631 / 0.618 ms |

This probe used 90 MB versus 28 MB scratch and verified bit-exact output.
Batching adds host dispatch/commit overhead at reorder boundaries, approximately
0.37 ms here; it does not run on ordinary optimizer-only iterations. GPU medians
were similar in this probe. Earlier samples varied substantially, including
unmodified operations, so these are isolated diagnostic observations—not a
full-training speedup claim or a prediction for other capacities/system loads.