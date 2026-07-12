// Opacity + scale regularization gradients (int64 fixed-point).

#include "../common/types.slang"
#include "../common/math.slang"

static const uint CLUSTER_SIZE = 128;  // must match cluster_cull.slang

[shader("compute")]
[numthreads(256, 1, 1)]
void addRegularization(
    uint3 dtid: SV_DispatchThreadID,
    StructuredBuffer<Gaussian3D> gaussians,
    StructuredBuffer<uint> num_gaussians_counter,
    RWStructuredBuffer<int64_t> grad_opacity,
    RWStructuredBuffer<int64_t> grad_log_scale_x,
    RWStructuredBuffer<int64_t> grad_log_scale_y,
    RWStructuredBuffer<int64_t> grad_log_scale_z,
    // Cluster-cull mask: correctness fix. adam.slang skips Adam for culled
    // clusters, so any grad added here would linger uncleared into the next
    // iter and inject spurious drift when the cluster becomes visible again.
    // Seeded all-ones when cluster-cull is disabled.
    StructuredBuffer<uint> cluster_visible,
    uniform float w_opacity,
    uniform float w_scale,
    uniform float inv_n)
{
    uint i = dtid.x;
    if (i >= num_gaussians_counter[0]) return;
    if (cluster_visible[i / CLUSTER_SIZE] == 0u) return;

    Gaussian3D g = gaussians[i];

    if (w_opacity > 0.0)
    {
        float sig = sigmoid(g.opacity);
        float d_opa = w_opacity * sig * (1.0 - sig) * inv_n;
        InterlockedAdd(grad_opacity[i], toFixed(d_opa));
    }

    if (w_scale > 0.0)
    {
        float3 scale = exp(g.log_scale);
        float3 d_scale = w_scale * scale * inv_n;
        InterlockedAdd(grad_log_scale_x[i], toFixed(d_scale.x));
        InterlockedAdd(grad_log_scale_y[i], toFixed(d_scale.y));
        InterlockedAdd(grad_log_scale_z[i], toFixed(d_scale.z));
    }
}
