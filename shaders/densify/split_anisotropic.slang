// Structure-aware anisotropic splitting (SAD-GS, arXiv 2604.28016, Eq. 8-9).
//
// Unlike split_and_clone.slang (1 gaussian -> 2 children, uniform shrink),
// this produces an n_x * n_y * n_z grid of children per selected gaussian,
// where the per-axis factors n_k = ceil(sqrt(eta_k)) were computed by the CPU
// decision and live in split_factor_{x,y,z}. Children are laid out on a regular
// grid in the parent's local frame (Eq. 9):
//     mu_child = mu_parent + R_parent * (s_parent (.) g)
//     s_child  = s_parent (./) (n_x, n_y, n_z)
// where g are grid coordinates in the unit cube centered at 0, and (.)/(./)
// are elementwise mul/div. Total children = n_x*n_y*n_z; child index 0
// overwrites the parent slot, the rest atomic-append at the tail.
//
// ACTION semantics match the rest of the densify suite: only ACTION_SPLIT
// gaussians grid-split here; CLONE still goes through split_and_clone; KEEP /
// PRUNE are no-ops (PRUNE handled by scatter-compact via the valid mask).

#include "../common/types.slang"
#include "../common/math.slang"

static const uint ACTION_KEEP  = 0;
static const uint ACTION_SPLIT = 1;
static const uint ACTION_CLONE = 2;
static const uint ACTION_PRUNE = 3;

AdamState _zero_adam()
{
    AdamState s;
    s.mean_m = float3(0, 0, 0); s.mean_v = float3(0, 0, 0);
    s.log_scale_m = float3(0, 0, 0); s.log_scale_v = float3(0, 0, 0);
    s.rot_m = float4(0, 0, 0, 0); s.rot_v = float4(0, 0, 0, 0);
    s.opacity_m = 0.0; s.opacity_v = 0.0;
    s.color_m = float3(0, 0, 0); s.color_v = float3(0, 0, 0);
    return s;
}

// One child of the grid at integer coords (ix, iy, iz) within an
// (nx, ny, nz) lattice. Grid coordinate g_k in [-(n_k-1)/2, (n_k-1)/2] / n_k,
// i.e. centered, spanning roughly +/- 0.5 * s_parent along each axis so the
// children tile the parent's 1-sigma extent.
Gaussian3D _grid_child(Gaussian3D parent, uint3 idx, uint3 n)
{
    Gaussian3D child = parent;
    float3 scale = exp(parent.log_scale);
    float3x3 R = matrixFromQuaternion(parent.rot_quat);

    // Centered grid coordinate in [-(n-1)/2 .. (n-1)/2] / n per axis.
    // For n_k == 1 this is 0 (no offset along that axis).
    float3 g;
    g.x = (n.x > 1u) ? ((float(idx.x) - 0.5 * float(n.x - 1u)) / float(n.x)) : 0.0;
    g.y = (n.y > 1u) ? ((float(idx.y) - 0.5 * float(n.y - 1u)) / float(n.y)) : 0.0;
    g.z = (n.z > 1u) ? ((float(idx.z) - 0.5 * float(n.z - 1u)) / float(n.z)) : 0.0;

    float3 local = float3(scale.x * g.x, scale.y * g.y, scale.z * g.z);
    child.mean = parent.mean + mul(R, local);
    // Shrink each axis by its split factor (Eq. 9): log(s / n) = log_scale - log(n).
    child.log_scale = parent.log_scale - float3(log(float(n.x)), log(float(n.y)), log(float(n.z)));
    return child;
}

[shader("compute")]
[numthreads(256, 1, 1)]
void splitAnisotropic(
    uint3 dtid: SV_DispatchThreadID,
    RWStructuredBuffer<Gaussian3D> gaussians,
    RWStructuredBuffer<AdamState> adam_state,
    RWStructuredBuffer<uint> num_gaussians_counter,
    StructuredBuffer<uint> action,
    StructuredBuffer<uint> split_factor_x,
    StructuredBuffer<uint> split_factor_y,
    StructuredBuffer<uint> split_factor_z,
    StructuredBuffer<uint> pre_count_buf,
    uniform uint max_splats)
{
    uint i = dtid.x;
    uint pre_count = pre_count_buf[0];
    if (i >= pre_count) return;
    uint act = action[i];
    if (act != ACTION_SPLIT && act != ACTION_CLONE) return;  // KEEP/PRUNE: no-op

    Gaussian3D parent = gaussians[i];

    // CLONE (small under-resolved gaussians): append one EXACT duplicate at the
    // tail with zeroed Adam state; parent slot untouched. Matches SADGS
    // densify_and_clone_structgs and the reference-3DGS clone (opacity copied
    // unchanged; fresh optimizer state so parent/child diverge under gradient).
    if (act == ACTION_CLONE)
    {
        uint tail;
        InterlockedAdd(num_gaussians_counter[0], 1u, tail);
        if (tail < max_splats)
        {
            gaussians[tail] = parent;
            adam_state[tail] = _zero_adam();
        }
        return;
    }

    // SPLIT (large under-resolved gaussians): anisotropic grid subdivision.
    uint3 n = uint3(max(split_factor_x[i], 1u),
                    max(split_factor_y[i], 1u),
                    max(split_factor_z[i], 1u));
    uint total = n.x * n.y * n.z;
    if (total <= 1u) return;   // decision produced no actual subdivision

    // Child 0 overwrites the parent slot; children 1..total-1 append at the
    // tail. Reserve the (total-1) tail slots with a single atomic add so the
    // block is contiguous and we never interleave with another gaussian's
    // children.
    uint tail_base = 0u;
    if (total > 1u)
        InterlockedAdd(num_gaussians_counter[0], total - 1u, tail_base);

    uint child_ordinal = 0u;
    for (uint iz = 0u; iz < n.z; ++iz)
    for (uint iy = 0u; iy < n.y; ++iy)
    for (uint ix = 0u; ix < n.x; ++ix)
    {
        Gaussian3D c = _grid_child(parent, uint3(ix, iy, iz), n);
        uint dst;
        if (child_ordinal == 0u)
        {
            dst = i;   // overwrite parent slot
        }
        else
        {
            dst = tail_base + (child_ordinal - 1u);
            if (dst >= max_splats)  // capacity clamp — drop overflow children
            {
                child_ordinal++;
                continue;
            }
        }
        gaussians[dst] = c;
        adam_state[dst] = _zero_adam();
        child_ordinal++;
    }
}
