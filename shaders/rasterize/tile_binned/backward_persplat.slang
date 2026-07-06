// Transposed (per-splat) backward rasterizer.
//
// The per-pixel backward (backward.slang) parallelizes over pixels and
// serializes the splat chain, so all 256 pixels touching a splat contend
// on that splat's fat gradient accumulator (~16 atomics each). This kernel
// transposes the loop nest: the SPLAT is the unit of parallelism. A lane
// owns a splat, streams over all 256 tile pixels accumulating its splat's
// gradient in registers, then issues ONE set of atomics for the whole
// tile. The contended per-pixel accumulator leaves the hot path.
//
// The lane can't walk a pixel's front-to-back chain to get the pre-splat
// transmittance T_i, so it reconstructs it from the forward checkpoints
// (checkpoint.slang): read the bucket's (T, C_in_front) checkpoint and
// re-blend the splats between the checkpoint and this lane's splat via a
// wave prefix-product of (1-alpha) and prefix-sum of c*alpha*T. All
// forward multiplies of (1-alpha) -> no division by near-zero.
//
// Layout: [numthreads(256)], one workgroup per tile = 8 warps of 32 lanes.
// Warp w processes bucket (base_bucket + w); lane L in that warp owns splat
// (base_bucket + w) * P + L. The bucket loop advances by 8 buckets/iter.
//
// Gradient math is EXACTLY equivalent to backward.slang; see the C_behind
// telescoping identity in the pixel loop below.

#include "../../common/types.slang"
#include "../../common/math.slang"
#include "../../primitives/gaussian/project.slang"
#include "../../primitives/gaussian/eval.slang"
#include "checkpoint.slang"

static const uint TILE_SIZE = 16;
static const uint TILE_AREA = TILE_SIZE * TILE_SIZE;   // 256
static const uint NUM_WARPS = TILE_AREA / 32u;         // 8 (assumes wave=32)

static const float ALPHA_MAX = 0.99;
static const float ALPHA_MIN_TRACK = 1.0 / 255.0;

// Per-pixel constants staged in LDS once (avoids re-reading global textures
// on every bucket). dL_dC (rgb) + dL_dA (a) + T_final + C_total (rgb) = 8
// floats/pixel * 256 = 2048 floats = 8 KB.
groupshared float4 gs_dLdC[TILE_AREA];   // rgb = dL_dC, a = dL_dA
groupshared float  gs_Tfinal[TILE_AREA];
groupshared float3 gs_Ctotal[TILE_AREA];
groupshared uint   gs_processed[TILE_AREA];
groupshared uint   gs_max_processed;     // deepest splat any pixel processed

[shader("compute")]
[numthreads(256, 1, 1)]
void backwardRasterizePerSplat(
    uint3 gid: SV_GroupID,
    uint3 gtid: SV_GroupThreadID,
    StructuredBuffer<Gaussian3D> gaussians,
    StructuredBuffer<Projected2DGaussian> projected_2d,
    StructuredBuffer<uint> tile_start,
    StructuredBuffer<uint> tile_end,
    StructuredBuffer<uint> sorted_gaussian_ids,
    StructuredBuffer<uint> bucket_offsets,
    StructuredBuffer<uint2> raster_checkpoints,
    StructuredBuffer<CameraData> camera,
    RWTexture2D<float4> output_image,
    RWTexture2D<float> output_transmittance,
    RWTexture2D<uint> gaussian_count_per_pixel,
    RWTexture2D<float4> dL_dC_texture,
    RWStructuredBuffer<int64_t> grad_mean_x,
    RWStructuredBuffer<int64_t> grad_mean_y,
    RWStructuredBuffer<int64_t> grad_mean_z,
    RWStructuredBuffer<int64_t> grad_log_scale_x,
    RWStructuredBuffer<int64_t> grad_log_scale_y,
    RWStructuredBuffer<int64_t> grad_log_scale_z,
    RWStructuredBuffer<int64_t> grad_rot_w,
    RWStructuredBuffer<int64_t> grad_rot_x,
    RWStructuredBuffer<int64_t> grad_rot_y,
    RWStructuredBuffer<int64_t> grad_rot_z,
    RWStructuredBuffer<int64_t> grad_opacity,
    RWStructuredBuffer<int64_t> grad_color_r,
    RWStructuredBuffer<int64_t> grad_color_g,
    RWStructuredBuffer<int64_t> grad_color_b,
    RWStructuredBuffer<int64_t> grad_2d_x_iter,
    RWStructuredBuffer<int64_t> grad_2d_y_iter)
{
    uint tid = gtid.x;
    uint2 frame_dim_px = uint2(camera[0].image_size);
    uint2 num_tiles = (frame_dim_px + TILE_SIZE - 1) / TILE_SIZE;
    // 1D dispatch: one 256-thread workgroup per tile. gid.x is the LINEAR
    // tile index; recover 2D tile coords from it (do NOT assume a 2D grid
    // like the per-pixel backward does).
    uint tile_idx = gid.x;
    uint tile_x = tile_idx % num_tiles.x;
    uint tile_y = tile_idx / num_tiles.x;
    if (tile_y >= num_tiles.y) return;

    uint t_start = tile_start[tile_idx];
    uint t_end   = tile_end[tile_idx];
    uint tile_count = (t_end > t_start) ? (t_end - t_start) : 0u;
    if (tile_count == 0u) return;

    uint tile_bucket_base = bucket_offsets[tile_idx];

    float4x4 world_to_view = camera[0].view_matrix;
    float4x4 view_to_clip  = camera[0].proj_matrix;
    float2 frame_dim = camera[0].image_size;

    // --- Stage per-pixel constants into LDS once ---
    // tid indexes a pixel-in-tile [0,256). Its screen coords:
    uint tile_origin_x = tile_x * TILE_SIZE;
    uint tile_origin_y = tile_y * TILE_SIZE;
    uint2 my_pixel = uint2(tile_origin_x + (tid % TILE_SIZE),
                           tile_origin_y + (tid / TILE_SIZE));
    bool my_pixel_valid = (my_pixel.x < frame_dim_px.x && my_pixel.y < frame_dim_px.y);
    if (my_pixel_valid)
    {
        gs_dLdC[tid]      = dL_dC_texture[my_pixel];
        gs_Tfinal[tid]    = output_transmittance[my_pixel];
        gs_Ctotal[tid]    = output_image[my_pixel].rgb;
        gs_processed[tid] = gaussian_count_per_pixel[my_pixel];
    }
    else
    {
        gs_dLdC[tid]      = float4(0.0);
        gs_Tfinal[tid]    = 1.0;
        gs_Ctotal[tid]    = float3(0.0);
        gs_processed[tid] = 0u;   // no splat contributes to an OOB pixel
    }
    GroupMemoryBarrierWithGroupSync();

    // Deepest splat any pixel in this tile actually processed. Pixels
    // early-break (forward T < 1e-4), so max_processed <= tile_count and is
    // usually far smaller in dense tiles. Buckets beyond ceil(max/P) have
    // zero contributing pixels -> skip them entirely (saturation early-out).
    if (tid == 0u) gs_max_processed = 0u;
    GroupMemoryBarrierWithGroupSync();
    InterlockedMax(gs_max_processed, gs_processed[tid]);
    GroupMemoryBarrierWithGroupSync();
    uint max_processed = gs_max_processed;

    uint lane = WaveGetLaneIndex();          // splat within the bucket [0,32)
    uint warp_idx = tid / WaveGetLaneCount(); // which bucket (offset from base)
    // Truncate to the buckets some pixel actually reached (vs. tile_count).
    uint num_buckets = (max_processed + RASTER_BUCKET_SIZE - 1u) / RASTER_BUCKET_SIZE;

    // --- Bucket loop: 8 warps handle 8 consecutive buckets per iteration ---
    for (uint base_bucket = 0u; base_bucket < num_buckets; base_bucket += NUM_WARPS)
    {
        uint bucket = base_bucket + warp_idx;
        bool bucket_valid = (bucket < num_buckets);

        // This lane's splat (global index within the tile's sorted list).
        uint local_splat = bucket * RASTER_BUCKET_SIZE + lane;
        bool splat_valid = bucket_valid && (local_splat < tile_count);

        // Load this lane's splat once (registers). Masked lanes hold junk
        // but never write output; they act as scan identities via alpha=0.
        uint g_idx = 0u;
        Gaussian3D g;
        Projected2DGaussian proj;
        if (splat_valid)
        {
            g_idx = sorted_gaussian_ids[t_start + local_splat];
            g = gaussians[g_idx];
            proj = projected_2d[g_idx];
        }

        // Fat gradient accumulators for THIS lane's splat, summed over pixels.
        float2 acc_mean2d = float2(0.0);
        float3 acc_cov2d  = float3(0.0);
        float  acc_opacity = 0.0;
        float3 acc_color  = float3(0.0);
        float  acc_absx = 0.0;
        float  acc_absy = 0.0;

        float3 sh0 = splat_valid ? g.sh0 : float3(0.0);
        float3 c_i = shDC(sh0);

        uint cp_bucket = tile_bucket_base + bucket;
        // First splat index of this bucket. A pixel contributes to this
        // bucket only if its ray reached at least this splat
        // (processed[p] > bucket_first_splat); otherwise ALL 32 lanes are
        // masked for that pixel and it produces zero gradient. `bucket` is
        // warp-uniform, so this skip is warp-uniform -> the wave scans below
        // stay valid (all 32 lanes continue together).
        uint bucket_first_splat = bucket * RASTER_BUCKET_SIZE;

        // --- Pixel loop: all 32 lanes lockstep so wave scans are valid ---
        for (uint p = 0u; p < TILE_AREA; ++p)
        {
            // Per-bucket saturation skip: if pixel p's ray terminated before
            // this bucket, no lane contributes -> skip the eval + wave ops
            // entirely. Warp-uniform (bucket + gs_processed[p] are uniform
            // across the 32 lanes), so this does not desync the wave scans.
            uint processed = gs_processed[p];
            if (processed <= bucket_first_splat)
                continue;

            // Early-break mask: pixel p only contributes gradient to splats
            // it actually processed (local_splat < processed[p]). Masked
            // lanes contribute alpha=0 -> scan identity (oma=1, cαT=0).
            bool contributes = splat_valid && (local_splat < processed);

            // Read the bucket checkpoint for pixel p (in-front-of-bucket
            // (T, C)). Only meaningful if the pixel reached this bucket,
            // i.e. some lane here is active; masked lanes ignore the value.
            float T_ck; float3 C_ck;
            unpackCheckpoint(raster_checkpoints[checkpointIndex(cp_bucket, p)], T_ck, C_ck);

            // Pixel center for pixel-in-tile p (computed unconditionally so
            // it's in scope for the eval backward below).
            float2 pc = float2(tile_origin_x + (p % TILE_SIZE),
                               tile_origin_y + (p / TILE_SIZE)) + 0.5;

            float alpha = 0.0;
            float alpha_grad_scale = 0.0;
            if (contributes)
            {
                float alpha_raw = evalGaussian2DFromCov(pc, proj.mean, proj.cov2d, g.opacity);
                alpha = min(alpha_raw, ALPHA_MAX);
                alpha_grad_scale = (alpha_raw <= ALPHA_MAX) ? 1.0 : 0.0;
            }

            float oma = 1.0 - alpha;                    // identity (1) when masked
            float3 caT_self;                            // c*alpha*T for this lane

            // T_i for this lane = checkpoint T * product of (1-alpha) for
            // the LOWER-indexed lanes in this bucket (exclusive prefix).
            float pref_prod = WavePrefixProduct(oma);   // exclusive
            float T_i = T_ck * pref_prod;

            float caT = alpha * T_i;                    // scalar weight
            // color contribution c*alpha*T for the prefix-sum of C_in_front.
            // Use the SAME luminance-free vector sum the forward used.
            float3 contrib = c_i * caT;                 // 0 when masked (alpha=0)

            // C_in_front(i) = C_ck + sum of contrib for lower lanes.
            float3 cif = C_ck + float3(
                WavePrefixSum(contrib.x),
                WavePrefixSum(contrib.y),
                WavePrefixSum(contrib.z));

            if (contributes)
            {
                float3 dL_dC = gs_dLdC[p].rgb;
                float  dL_dA = gs_dLdC[p].a;
                float  T_final = gs_Tfinal[p];
                float3 C_total = gs_Ctotal[p];

                float denom = 1.0 - alpha;
                // C_behind(i) = C_total - C_in_front(i) - c_i*alpha_i*T_i
                //            = sum_{j>i, j<processed} c_j*alpha_j*T_j.
                float3 C_behind = C_total - cif - contrib;

                // Color grad: dL_dc_i = dL_dC * (alpha * T_i), through shDC.
                var dp_sh0 = diffPair(sh0);
                float3 dL_dc_i = dL_dC * caT;
                bwd_diff(shDC)(dp_sh0, dL_dc_i);

                float3 dC_dalpha = c_i * T_i - C_behind / denom;
                float dL_dalpha = dot(dL_dC, dC_dalpha) + dL_dA * (T_final / denom);
                float dL_d_alpha_raw = dL_dalpha * alpha_grad_scale;

                var dp_mean_2d = diffPair(proj.mean);
                var dp_cov2d   = diffPair(proj.cov2d);
                var dp_opacity = diffPair(g.opacity);
                bwd_diff(evalGaussian2DFromCov)(
                    pc, dp_mean_2d, dp_cov2d, dp_opacity, dL_d_alpha_raw);

                acc_mean2d += dp_mean_2d.d;
                acc_cov2d  += dp_cov2d.d;
                acc_opacity += dp_opacity.d;
                acc_color  += dp_sh0.d;

                if (alpha >= ALPHA_MIN_TRACK)
                {
                    acc_absx += abs(dp_mean_2d.d.x);
                    acc_absy += abs(dp_mean_2d.d.y);
                }
            }
        }

        // --- Per-splat projection backward + atomics (one lane per splat) ---
        if (splat_valid)
        {
            var dp_mean      = diffPair(g.mean);
            var dp_log_scale = diffPair(g.log_scale);
            var dp_rot_quat  = diffPair(g.rot_quat);
            bwd_diff(projectGaussian3DTo2D)(
                dp_mean, dp_log_scale, dp_rot_quat,
                world_to_view, view_to_clip, frame_dim,
                acc_mean2d, acc_cov2d, 0.0, 0.0);

            InterlockedAdd(grad_mean_x[g_idx], toFixed(dp_mean.d.x));
            InterlockedAdd(grad_mean_y[g_idx], toFixed(dp_mean.d.y));
            InterlockedAdd(grad_mean_z[g_idx], toFixed(dp_mean.d.z));
            InterlockedAdd(grad_log_scale_x[g_idx], toFixed(dp_log_scale.d.x));
            InterlockedAdd(grad_log_scale_y[g_idx], toFixed(dp_log_scale.d.y));
            InterlockedAdd(grad_log_scale_z[g_idx], toFixed(dp_log_scale.d.z));
            InterlockedAdd(grad_rot_w[g_idx], toFixed(dp_rot_quat.d.x));
            InterlockedAdd(grad_rot_x[g_idx], toFixed(dp_rot_quat.d.y));
            InterlockedAdd(grad_rot_y[g_idx], toFixed(dp_rot_quat.d.z));
            InterlockedAdd(grad_rot_z[g_idx], toFixed(dp_rot_quat.d.w));
            InterlockedAdd(grad_opacity[g_idx], toFixed(acc_opacity));
            InterlockedAdd(grad_color_r[g_idx], toFixed(acc_color.x));
            InterlockedAdd(grad_color_g[g_idx], toFixed(acc_color.y));
            InterlockedAdd(grad_color_b[g_idx], toFixed(acc_color.z));
            InterlockedAdd(grad_2d_x_iter[g_idx], toFixed(acc_absx));
            InterlockedAdd(grad_2d_y_iter[g_idx], toFixed(acc_absy));
        }

        // Warps advance to their next bucket; a barrier isn't needed because
        // each warp's scans are self-contained and gs_* staging is read-only
        // here. But different warps finish their pixel loops independently,
        // and the next iteration re-reads only read-only LDS -> safe.
    }
}
