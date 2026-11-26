# Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

"""The Metal int64 atomic-add emulation matches a native two's-complement add.

Apple GPUs cannot atomic-add a 64-bit integer. ``atomicAddFixed`` splits the addend
into little-endian halves and adds them with 32-bit atomics, carrying from the low
word's returned old value. This checks that identity, including interleaved updates.
"""

from __future__ import annotations

MASK32 = 0xFFFFFFFF


def _split(delta: int) -> tuple[int, int]:
    bits = delta & ((1 << 64) - 1)
    return bits & MASK32, bits >> 32


def _apply(lo: int, hi: int, delta: int) -> tuple[int, int]:
    add_lo, add_hi = _split(delta)
    old = lo
    lo = (lo + add_lo) & MASK32
    carry = 1 if (old + add_lo) > MASK32 else 0
    hi = (hi + add_hi + carry) & MASK32
    return lo, hi


def _as_signed(lo: int, hi: int) -> int:
    bits = (hi << 32) | lo
    if bits >= (1 << 63):
        bits -= 1 << 64
    return bits


def test_split_add_matches_int64_including_negatives() -> None:
    values = [0, 1, -1, 65536, -65536, 2**31 - 1, -(2**31), 2**40, -(2**50), 2**62, -(2**62)]
    for start in values:
        lo, hi = _split(start)
        total = start
        for delta in values:
            lo, hi = _apply(lo, hi, delta)
            total += delta
            assert _as_signed(lo, hi) == total


def test_interleaved_carries_match_the_sum() -> None:
    """Low halves commit immediately; high halves may land later, in any order."""
    adds = [0xFFFFFFFF, 2, -3, 0x100000000, -(1 << 33) + 5, 7, -1]
    lo = hi = 0
    pending: list[int] = []
    for delta in adds:
        add_lo, add_hi = _split(delta)
        old = lo
        lo = (lo + add_lo) & MASK32
        carry = 1 if (old + add_lo) > MASK32 else 0
        pending.append((add_hi + carry) & MASK32)
    for high in reversed(pending):
        hi = (hi + high) & MASK32
    assert _as_signed(lo, hi) == sum(adds)
