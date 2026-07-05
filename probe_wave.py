"""Gate probe: verify the wave intrinsics DeviceRadixSort relies on compile
and behave correctly on the Vulkan (and D3D12) backend via SlangPy.

Checks, per lane in a single workgroup:
  - WaveGetLaneCount() reports a runtime lane count >= 16
  - WavePrefixSum is EXCLUSIVE (lane 0 -> 0)
  - WaveActiveBallot + countbits + firstbitlow behave as GPUSorting expects
  - WaveReadLaneAt broadcasts correctly
"""

from __future__ import annotations

import numpy as np
import slangpy as spy

from trainer.device import create_device


SHADER = r"""
[shader("compute")]
[numthreads(64, 1, 1)]
void waveProbe(
    uint3 dtid: SV_DispatchThreadID,
    uint lid: SV_GroupIndex,
    RWStructuredBuffer<uint> lane_count,     // [1] wave lane count
    RWStructuredBuffer<uint> prefix,         // [N] exclusive prefix-sum of 1s
    RWStructuredBuffer<uint> ballot_pop,     // [N] popcount of ballot(lane even)
    RWStructuredBuffer<uint> lowest_even,    // [N] firstbitlow of that ballot
    RWStructuredBuffer<uint> read_lane0)     // [N] WaveReadLaneAt(laneIdx, 0)
{
    uint i = dtid.x;

    if (i == 0)
        lane_count[0] = WaveGetLaneCount();

    // Exclusive prefix sum of value 1 across the wave => lane's index within wave.
    uint ps = WavePrefixSum(1u);
    prefix[i] = ps;

    // Ballot: which lanes have an even lane index.
    bool even = (WaveGetLaneIndex() & 1u) == 0u;
    uint4 b = WaveActiveBallot(even);
    ballot_pop[i] = countbits(b.x) + countbits(b.y) + countbits(b.z) + countbits(b.w);
    lowest_even[i] = firstbitlow(b.x);

    // Broadcast lane 0's lane index (== 0) to everyone.
    read_lane0[i] = WaveReadLaneAt(WaveGetLaneIndex(), 0);
}
"""


def _run(backend: str) -> None:
    try:
        device = create_device(backend=backend)
    except Exception as exc:  # pragma: no cover - env dependent
        print(f"[{backend}] unavailable: {exc}")
        return

    N = 64
    # Load by path (matches the proven pattern in _make_kernel / the tests).
    from trainer.device import SHADERS_DIR
    p = SHADERS_DIR / "_wave_probe_tmp.slang"
    p.write_text(SHADER)
    try:
        program = device.load_program(str(p), ["waveProbe"])
        kernel = device.create_compute_kernel(program)
    finally:
        p.unlink(missing_ok=True)

    lane_count = spy.NDBuffer(device=device, dtype="uint", shape=(1,))
    prefix = spy.NDBuffer(device=device, dtype="uint", shape=(N,))
    ballot_pop = spy.NDBuffer(device=device, dtype="uint", shape=(N,))
    lowest_even = spy.NDBuffer(device=device, dtype="uint", shape=(N,))
    read_lane0 = spy.NDBuffer(device=device, dtype="uint", shape=(N,))

    kernel.dispatch(
        thread_count=[N, 1, 1],
        lane_count=lane_count.storage,
        prefix=prefix.storage,
        ballot_pop=ballot_pop.storage,
        lowest_even=lowest_even.storage,
        read_lane0=read_lane0.storage,
    )
    device.wait_for_idle()

    lc = int(np.frombuffer(lane_count.to_numpy().tobytes(), dtype=np.uint32)[0])
    pre = np.frombuffer(prefix.to_numpy().tobytes(), dtype=np.uint32)[:N]
    pop = np.frombuffer(ballot_pop.to_numpy().tobytes(), dtype=np.uint32)[:N]
    low = np.frombuffer(lowest_even.to_numpy().tobytes(), dtype=np.uint32)[:N]
    rl0 = np.frombuffer(read_lane0.to_numpy().tobytes(), dtype=np.uint32)[:N]

    print(f"[{backend}] WaveGetLaneCount = {lc}")
    assert lc >= 16, f"lane count {lc} < 16; WGE16-only port assumption violated"

    # Exclusive prefix sum of 1 => index within wave [0..lc) repeating per wave.
    expected_prefix = np.tile(np.arange(lc, dtype=np.uint32), N // lc)
    ok_prefix = np.array_equal(pre, expected_prefix)
    print(f"[{backend}] WavePrefixSum exclusive: {ok_prefix} "
          f"(lane0={pre[0]}, first {min(lc,8)}={pre[:min(lc,8)].tolist()})")
    assert pre[0] == 0, "WavePrefixSum is NOT exclusive (lane0 != 0)"
    assert ok_prefix, "WavePrefixSum values unexpected"

    # Half the lanes are even => popcount == lc/2 for every lane.
    ok_pop = np.all(pop == lc // 2)
    print(f"[{backend}] WaveActiveBallot popcount == lc/2: {ok_pop} (got {pop[0]})")
    assert ok_pop, "ballot popcount wrong"

    # firstbitlow of the even-lane ballot => lane 0 is even => 0.
    ok_low = np.all(low == 0)
    print(f"[{backend}] firstbitlow(evenBallot) == 0: {ok_low} (got {low[0]})")
    assert ok_low, "firstbitlow wrong"

    ok_rl0 = np.all(rl0 == 0)
    print(f"[{backend}] WaveReadLaneAt(_,0) == 0: {ok_rl0} (got {rl0[0]})")
    assert ok_rl0, "WaveReadLaneAt broadcast wrong"

    print(f"[{backend}] ALL WAVE INTRINSIC CHECKS PASSED\n")


if __name__ == "__main__":
    for be in ("vulkan", "d3d12"):
        _run(be)
