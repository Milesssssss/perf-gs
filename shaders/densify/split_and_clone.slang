// Phase B of densification: produce children for SPLIT and CLONE actions.
//
// One thread per gaussian, operating over the PRE-densification range.
// - SPLIT: overwrite slot i with child-1 (scale / 1.6, jittered position
//   within parent covariance), zero its Adam state. Atomic-append child-2
//   (same rule, different jitter) at the tail with zeroed Adam state.
// - CLONE: atomic-append a clone with inherited parent Adam state;
//   parent slot i untouched. (Paper-style position jitter along
//   accumulated gradient direction is a v2.2 deferred item.)
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

Gaussian3D _child_for_clone(Gaussian3D parent)
{
    // v2.1 ships identical-position clone. Gradient-direction nudge
    // is a v2.2 TODO (task #54).
    return parent;
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
    uniform uint pre_count,       // num_gaussians_counter[0] BEFORE this dispatch
    uniform uint max_splats)
{
    uint i = dtid.x;
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
        Gaussian3D cl = _child_for_clone(parent);
        uint tail;
        InterlockedAdd(num_gaussians_counter[0], 1u, tail);
        if (tail < max_splats)
        {
            gaussians[tail] = cl;
            adam_state[tail] = parent_adam; // inherit
        }
    }
}
