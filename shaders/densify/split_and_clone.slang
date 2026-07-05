// Phase B of densification: produce children for SPLIT and CLONE actions.
//
// One thread per gaussian, operating over the PRE-densification range.
// - SPLIT: overwrite slot i with child-1 (scale / 1.6, jittered position
//   within parent covariance), zero its Adam state. Atomic-append child-2
//   (same rule, different jitter) at the tail with zeroed Adam state.
// - CLONE: atomic-append an EXACT copy of the parent at the tail with
//   ZEROED Adam state; parent slot i untouched. No position jitter —
//   matches gsplat's duplicate(): `torch.cat([p, p[sel]])` + fresh
//   optimizer state. Parent and child diverge via per-iter gradient
//   noise under a fresh Adam moment buffer. (Inheriting Adam would
//   make the twins take identical first steps and defeat the
//   purpose of duplicating.)
// - KEEP / PRUNE: no-op here. PRUNE is handled by the scatter-compact
//   pass using the mask from Phase C.
//
// Both the split-scale-divisor (1.6) and the random-sample jitter are
// from Kerbl et al. (3D Gaussian Splatting, §5.2).

#include "../common/types.slang"
#include "../common/math.slang"

static const uint ACTION_KEEP  = 0;
static const uint ACTION_SPLIT = 1;
static const uint ACTION_CLONE = 2;
static const uint ACTION_PRUNE = 3;

static const float SPLIT_SCALE_DIVISOR = 1.6;

// Cheap deterministic-per-seed uniform[0,1).
// Wang hash is overkill for our use but it's 5 ops and predictable.
uint _wang_hash(uint x)
{
    x = (x ^ 61u) ^ (x >> 16u);
    x *= 9u;
    x = x ^ (x >> 4u);
    x *= 0x27d4eb2du;
    x = x ^ (x >> 15u);
    return x;
}

float _rand01(uint seed)
{
    return float(_wang_hash(seed)) / 4294967295.0;
}

// Box-Muller: two uniforms -> two standard-normal samples.
float2 _randn2(uint seed)
{
    float u1 = max(_rand01(seed + 1u), 1e-7);
    float u2 = _rand01(seed + 7919u);
    float r = sqrt(-2.0 * log(u1));
    float a = 6.28318530718 * u2;
    return float2(r * cos(a), r * sin(a));
}

// Draw a 3D offset from N(0, Σ) where Σ = R S^2 R^T (the world-space
// 3D covariance from the current log_scale + rot_quat).
float3 _sample_parent_cov_offset(float3 log_scale, float4 rot_quat, uint seed)
{
    float2 a = _randn2(seed);
    float2 b = _randn2(seed + 73856093u);
    // Three independent N(0,1) samples (fourth discarded).
    float3 z = float3(a.x, a.y, b.x);

    float3 scale = exp(log_scale);
    // Σ^{1/2} = R * diag(scale): for a zero-mean N(0,I) sample z, the
    // offset R*S*z is distributed N(0, R*S*S*R^T) = N(0, Σ).
    float3x3 R = matrixFromQuaternion(rot_quat);
    float3 scaled = float3(scale.x * z.x, scale.y * z.y, scale.z * z.z);
    return mul(R, scaled);
}

Gaussian3D _child_for_split(Gaussian3D parent, uint seed)
{
    Gaussian3D child = parent;
    float3 offset = _sample_parent_cov_offset(parent.log_scale, parent.rot_quat, seed);
    child.mean = parent.mean + offset;
    // Divide scale by the split divisor. In log space that's
    // log_scale - log(1.6).
    child.log_scale = parent.log_scale - log(SPLIT_SCALE_DIVISOR);
    return child;
}

AdamState _zero_adam()
{
    AdamState s;
    s.mean_m = float3(0, 0, 0);
    s.mean_v = float3(0, 0, 0);
    s.log_scale_m = float3(0, 0, 0);
    s.log_scale_v = float3(0, 0, 0);
    s.rot_m = float4(0, 0, 0, 0);
    s.rot_v = float4(0, 0, 0, 0);
    s.opacity_m = 0.0;
    s.opacity_v = 0.0;
    s.color_m = float3(0, 0, 0);
    s.color_v = float3(0, 0, 0);
    return s;
}

[shader("compute")]
[numthreads(256, 1, 1)]
void splitAndClone(
    uint3 dtid: SV_DispatchThreadID,
    RWStructuredBuffer<Gaussian3D> gaussians,
    RWStructuredBuffer<AdamState> adam_state,
    RWStructuredBuffer<uint> num_gaussians_counter,
    StructuredBuffer<uint> action,
    // Snapshot of num_gaussians_counter[0] taken BEFORE this dispatch.
    // Kept in a separate buffer so we can atomically bump the counter
    // without racing our own bounds check. Written by snapshotCount.
    StructuredBuffer<uint> pre_count_buf,
    uniform uint max_splats)
{
    uint i = dtid.x;
    uint pre_count = pre_count_buf[0];
    if (i >= pre_count) return;

    uint a = action[i];
    if (a == ACTION_KEEP || a == ACTION_PRUNE) return;

    Gaussian3D parent = gaussians[i];
    AdamState parent_adam = adam_state[i];

    if (a == ACTION_SPLIT)
    {
        // Generate two children from independent seeds, overwrite
        // parent slot with one and append the other.
        Gaussian3D c1 = _child_for_split(parent, i * 2u);
        Gaussian3D c2 = _child_for_split(parent, i * 2u + 1u);

        uint tail;
        InterlockedAdd(num_gaussians_counter[0], 1u, tail);
        if (tail < max_splats)
        {
            gaussians[tail] = c2;
            adam_state[tail] = _zero_adam();
        }
        // Overwrite parent slot with child-1.
        gaussians[i] = c1;
        adam_state[i] = _zero_adam();
    }
    else if (a == ACTION_CLONE)
    {
        // Exact duplicate of the parent — opacity COPIED UNCHANGED, matching
        // 3DGS/gsplat duplicate(). The previous code halved opacity on both
        // parent and copy ("to keep combined alpha unchanged"), but that is
        // wrong: (1) it contradicts the reference (clone copies as-is), and
        // (2) halving is not even alpha-neutral — two overlapping p/2 clones
        // composite to p - p^2/4 != p. Repeated clone generations drove
        // opacity toward zero, manufacturing the sub-0.1 floater haze.
        // The clone gets a fresh (zeroed) Adam state so parent and child
        // diverge under per-iter gradient noise instead of stepping in
        // lockstep; the parent slot is left untouched.
        Gaussian3D clone = parent;

        uint tail;
        InterlockedAdd(num_gaussians_counter[0], 1u, tail);
        if (tail < max_splats)
        {
            gaussians[tail] = clone;
            adam_state[tail] = _zero_adam();
        }
    }
}
