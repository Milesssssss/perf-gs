// Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

#ifndef TRAINER_FIXED_POINT_SLANG
#define TRAINER_FIXED_POINT_SLANG

// Fixed-point atomic gradient accumulation (int64, no overflow).
static const float FIXED_SCALE = 65536.0;
static const float FIXED_INV_SCALE = 1.0 / FIXED_SCALE;

int64_t toFixed(float value) { return int64_t(value * FIXED_SCALE); }
float fromFixed(int64_t value) { return float(value) * FIXED_INV_SCALE; }

// int64 atomic add. Vulkan and D3D12 keep a single InterlockedAdd. Metal has no 64-bit
// atomic add on any Apple GPU (only min/max on M3+), so the sum is two little-endian
// 32-bit atomic adds with the carry taken from each add's returned low word. Integer
// addition is associative, and that carry identity matches a native int64 add, including
// a negative addend's two's-complement high half. Callers never read the accumulator in
// the same dispatch, so the split update is not observed half-written.
[ForceInline]
void atomicAddFixed(RWStructuredBuffer<int64_t> buf, uint index, int64_t delta)
{
    __target_switch
    {
    case metal:
        __intrinsic_asm R"(
        do {
            device atomic_uint* p = ((device atomic_uint*)($0)) + ((uint)($1) * 2u);
            ulong bits = as_type<ulong>((long)($2));
            uint lo = (uint)bits;
            uint hi = (uint)(bits >> 32);
            uint old = atomic_fetch_add_explicit(p, lo, memory_order_relaxed);
            uint carry = (old + lo < old) ? 1u : 0u;
            atomic_fetch_add_explicit(p + 1, hi + carry, memory_order_relaxed);
        } while (0)
        )";
    default:
        InterlockedAdd(buf[index], delta);
    }
}

#endif // TRAINER_FIXED_POINT_SLANG
