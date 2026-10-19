// Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

// Minimal per-element function used by the platform smoke test.
// Dispatched via SlangPy's functional API, which vectorizes across the
// shape of the tensor passed from Python. Proves: device + module load +
// kernel codegen + dispatch + readback all round-trip on each backend.

float add_one(float x)
{
    return x + 1.0;
}
