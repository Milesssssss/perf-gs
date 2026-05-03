// 2D Gaussian Gradient Descent - Image-based Loss
// Compares rendered input Gaussian against a target IMAGE (not target Gaussian params)
/*$(ShaderResources)*/

#define MAX_GAUSSIANS_PER_PIXEL 10
#define MAX_GAUSSIANS_PER_TILE 1024
#define TILE_SIZE 16
#define F_TO_I_BIT_SHIFT = 16

static float2 camera_pos = /*$(Variable:cameraPos)*/;
static float2 scale = /*$(Variable:scale)*/;
static float2 frame_dim = /*$(Variable:frameDim)*/;
//static float4x4 world_to_view = transpose(/*$(Variable:ViewMtx)*/);
//static float4x4 view_to_screen = transpose(/*$(Variable:ProjMtx)*/);
static const float FIXED_SCALE = 65536.0;
static const float FIXED_INV_SCALE = 1.0 / FIXED_SCALE;

#include "../include/s2h.hlsl"
#include "utils.slang"

//TODO: Look into packing maybe, check magnitude of these and see what is needed
struct GradientAccumulator
{
    int grad_mean_x, grad_mean_y, grad_mean_z;
    int grad_mean2d_magnitude;
    int grad_log_scale_x, grad_log_scale_y, grad_log_scale_z;
    int grad_quat_x, grad_quat_y, grad_quat_z, grad_quat_w;
    int grad_color_r, grad_color_g, grad_color_b;
    int grad_opacity;
};

groupshared GradientAccumulator gs_gradientAccumulator; 

/*$(_compute:backwardPass)*/ (
    uint3 DTid: SV_DispatchThreadID,
    uint3 Gid: SV_GroupID,
    uint Gid_flat: SV_GroupIndex
)
{
    
    uint2 pixel = DTid.xy;
    
    if (pixel.x >= frame_dim.x || pixel.y >= frame_dim.y) return;

    uint2 numTiles = ((uint2)frame_dim + TILE_SIZE - 1) / TILE_SIZE;
    uint tileIdx = Gid.y * numTiles.x + Gid.x;
    uint tile_gaussian_count = min(screen_bin_tiles_count[tileIdx], MAX_GAUSSIANS_PER_TILE);
    uint start_idx = tileIdx * MAX_GAUSSIANS_PER_TILE;

    float4x4 world_to_view = float4x4(
        target_camera_view_matrix[0],
        target_camera_view_matrix[1],
        target_camera_view_matrix[2],
        target_camera_view_matrix[3]
    );

    float4x4 view_to_screen = float4x4(
        target_camera_projection_matrix[0],
        target_camera_projection_matrix[1],
        target_camera_projection_matrix[2],
        target_camera_projection_matrix[3]

    );

    // Load target and compute dL/dC
    float4 target_rgba = target_image[pixel];
    float target_alpha = target_rgba.a;
    float T = output_transmittance[pixel].r;
    float rendered_opacity = 1.0f - T;

    // L2 loss der in srgb space and we need to chain rule the display transf function
    //Forgetting that the loss was computed in perceptual space and not linear fucked up the results quite a bit, keep this checked better.
    //We could wrap this in a diff function instead of manually deriving? Maybe more clear
    float3 dL_dC_srgb = 2.0 * (output_image[pixel].rgb - target_rgba.rgb);
    float3 C_linear = pow(output_image[pixel].rgb, 2.2);
    float3 C_linear_clamped = max(C_linear, 1e-4);
    float3 dC_srgb_dC_linear = (1.0 / 2.2) * pow(C_linear_clamped, (1.0 / 2.2) - 1.0);
    float3 dL_dC_pixel = dL_dC_srgb * dC_srgb_dC_linear;

    float dL_dA_pixel = 2.0 * (rendered_opacity - target_alpha); // Chain rule: d/dT of (1-T - target)^2

    //float3 rendered_dx = output_image[pixel + uint2(1, 0)].rgb - output_image[pixel].rgb;
    //float3 rendered_dy = output_image[pixel + uint2(0, 1)].rgb - output_image[pixel].rgb;
    //float3 target_dx = target_image[pixel + uint2(1, 0)].rgb - target_image[pixel].rgb;
    //float3 target_dy = target_image[pixel + uint2(0, 1)].rgb - target_image[pixel].rgb;
//
    //float3 dL_dC_edge = 2.0 * ((rendered_dx - target_dx) + (rendered_dy - target_dy));
//
    //float edge_weight = 0.5;

    float3 dL_dC = dL_dC_pixel;// + edge_weight * dL_dC_edge;

    float3 color_behind_accum = float3(0, 0, 0); //TODO: black background color, move to variable

    uint gaussian_count = gaussian_count_per_pixel[pixel];

    float T_i_plus1 = T;

    // Back-to-front: recover T_i by undoing alpha
    // TODO: code is cluttered with a bit too many comments, consider moving to a derivation.md file or idk, keep the explanation somewhere

    // float2 total_color_grad_accum = 0;
    for (int i = tile_gaussian_count - 1; i >= 0; i--)
    {

        if(Gid_flat == 0)
        {
            gs_gradientAccumulator.grad_mean_x = 0;
            gs_gradientAccumulator.grad_mean_y = 0;
            gs_gradientAccumulator.grad_mean_z = 0;
            gs_gradientAccumulator.grad_mean2d_magnitude = 0;

            gs_gradientAccumulator.grad_log_scale_x = 0;
            gs_gradientAccumulator.grad_log_scale_y = 0;
            gs_gradientAccumulator.grad_log_scale_z = 0;

            gs_gradientAccumulator.grad_quat_x = 0;
            gs_gradientAccumulator.grad_quat_y = 0;
            gs_gradientAccumulator.grad_quat_z = 0;
            gs_gradientAccumulator.grad_quat_w = 0;

            gs_gradientAccumulator.grad_color_r = 0;
            gs_gradientAccumulator.grad_color_g = 0;
            gs_gradientAccumulator.grad_color_b = 0;

            gs_gradientAccumulator.grad_opacity = 0;

        }

        GroupMemoryBarrierWithGroupSync();

        uint gaussian_idx = screen_bin_tiles_gaussian_ids[start_idx + i];
        bool pixel_processed_this = (i < gaussian_count);

        if (pixel_processed_this)
        {
            Struct_Gaussian3D gaussian = input_gaussians[gaussian_idx];
            Struct_Projected2DGaussian proj = projected_gaussians_2d[gaussian_idx];


            //This could be cached from forward?
            float alpha_i = evalGaussian2DFromCov(pixel, proj.mean, proj.cov2d, gaussian.opacity);

            //NB that final color C is just a summ of contributions, we need to get the der of those contributions
            //At each step here we need to compute d(a_j * T_j * c_j)/da_i

            //We now need to recover Recover T_i from T_{i+1}
            // T currently holds T_{i+1} (final transmission value front to back)
            // We need T_i (transmittance BEFORE gaussian i)
            float denom = max(1.0 - alpha_i, 1e-6);
            float T_i = T_i_plus1 / denom; // Undo the (1 - alpha) multiplication
                                        // derivation maybe helpful in the future:
            //  T_0 = 1
            //  T_1 = (1 - a_0)
            //  T_2 = (1 - a_0)(1 - a_1) so --> T_1 = T_2 / (1 - a_1) --> T_i = T_i_plus1 / (1 - a_i)
            
            var dp_color_logit = diffPair(gaussian.color_logit);
            float3 color_rgb = sigmoid3(dp_color_logit.p); //the color is stored in logit, we first need to bwd this

            float3 color_i = color_rgb;

            // Gradient wrt color chained from (dL/dc_i = dL/dC * dC/dc_i) --> as the final C = sum(a_j * T_j * c_j) for j 0,...,gaussian_count
            float dC_dColor_i = alpha_i * T_i; 
            float3 dL_dColor_i = dL_dC * dC_dColor_i;

            bwd_diff(sigmoid3)(dp_color_logit, dL_dColor_i);

            //Gradient wrt alpha chained from (dL/da_i = dL/dC * dC/da_i) --> which is 
            // alpha affects this gaussian's contribution AND transmittance for all behind
            // as everything after i depends on alpha_i. (color_i * T_i) is C deriv wrt to current splat alpha, while (- color_behind_accum / denom) is deriv of all stuff behing
            float3 dC_dAlpha_i = (color_i * T_i - color_behind_accum / denom); 

            //For future me and anyone reading this. I wanted to understand the actual derivation, so here what I found out.

            // Anything before i does not depend on a_i, but i and j>i do (remember how front to back blending works right?). note: Ill explain dT_j/da_i later
            // For i --> d(a_i * T_i * c_i) = T_i * c_i
            // then for j > i
            //d(a_j * T_j * c_j)/da_i = a_j * c_j * ( dT_j/da_i )
            // = a_j * c_j * ( -T_j / (1 - a_i) )
            // = -a_j * T_j * c_j / (1 - a_i)

            //sum_{j>i} [-a_j * T_j * c_j / (1 - a_i)] =
            //= -1/(1 - a_i) * sum_{j>i}(a_j * T_j * c_j)
            //                \________________________/
            //                 its just the color_behind!!

            // Further explanation of why dT_j/da_i == (-T_j / (1 - a_i))
            // I'll try to explain here hopefully its clear
            // (- color_behind_accum / denom) is sum of all d(contribution_j)/da_i for j > i where j i+1,...,gaussian_count, remember that in the 
            // final loss, each splat contributes to the color with a_j * T_j * c_j
            // So, define Q = (1 - a_0)(1 - a_1)...(1 - a_{i-1})(1 - a_{i+1})...(1 - a_{j-1})
            // and T_j = (1 - a_i) * Q
            // dT_j/da_i = d[(1 - a_i) * Q]/da_i
            //  = Q * d(1 - a_i)/da_i
            //  = -Q
            // Now since T_j = (1 - a_i) * Q --> Q = T_j / (1 - a_i) and sobsituting back dT_j/da_i = -Q = -T_j / (1 - a_i)

            float3 dL_dAlpha_vec = dL_dC * dC_dAlpha_i;
            float dL_dAlpha_color = dot(dL_dAlpha_vec, float3(1, 1, 1));
            float dL_dAlpha_opacity = dL_dA_pixel * T / denom;
            //TODO: removed opacity loss for now
            float opacity_loss_weight = 0.1f;
            float dL_dAlpha = dL_dAlpha_color + dL_dAlpha_opacity * opacity_loss_weight;

            // Backprop through evalGaussian2D for position/scale/rotation and overall opacity term
            var dp_mean = diffPair(gaussian.mean);

            var dp_log_scale = diffPair(gaussian.log_scale);
            var dp_rotation = diffPair(gaussian.rotation_quat);
            var dp_opacity = diffPair(gaussian.opacity);

            // We chain rule back up as all these params influence the value of alpha.
            // We could use a different primitive which encapsulates the density distribution here
            bwd_diff(evalGaussian3DAtPixel)(
                pixel,
                dp_mean, dp_log_scale, dp_rotation, dp_opacity,
                world_to_view, view_to_screen, frame_dim,
                dL_dAlpha
            );

            var dp_mean_2d = diffPair(proj.mean); // Use the cached projected mean
            var dp_cov2d = diffPair(proj.cov2d);  // We don't need this gradient, but need the value
            var dp_opacity_dummy = diffPair(gaussian.opacity);

            //I do this just because i need also the 2d mean grads
            bwd_diff(evalGaussian2DFromCov)(
                pixel,
                dp_mean_2d,
                dp_cov2d,
                dp_opacity_dummy,
                dL_dAlpha // Same upstream gradient
            );

            //  Updates for next iteration 
            color_behind_accum += color_i * alpha_i * T_i;  // Accumulate this gaussian's contribution
            T_i_plus1 = T_i;                                // We now move T backward

            accumulate_gradients_groupshared(
                dp_mean.d,
                dp_mean_2d.d,
                dp_log_scale.d,
                dp_color_logit.d,
                dp_opacity.d,
                dp_rotation.d
            );
        }

        GroupMemoryBarrierWithGroupSync();

        // Todo could still do some more clever trick accumulating in groupshared arr and then sum and compacting in parallel, this can def be sped up
        if(Gid_flat == 0)
        {
            accumulate_gradients_global(gaussian_idx);
        }

        AllMemoryBarrierWithGroupSync();
    }
    
    GroupMemoryBarrierWithGroupSync();

    // output_grads[pixel] = float4(total_color_grad_accum / gaussian_count,0, 1.0);

    ContextGather ui;
    s2h_init(ui, pixel);
    s2h_drawSRGBRamp(ui, float2(2, 2));
    // linearOutput = lerp(linearOutput, ui.dstColor.rgb, );
    //output_grads[pixel] = ui.dstColor;
}

//TODO: a lot of quantization can be done here to pack stuff and/or compress

void accumulate_gradients_groupshared(float3 mean_d, float2 mean2d_d, float3 scale_d, float3 col_d, float opacity_d, float4 rotation_d)
{
    if (bool(/*$(Variable:descend_mean)*/))
    {
        InterlockedAdd(gs_gradientAccumulator.grad_mean_x, toFixed(mean_d.x));
        InterlockedAdd(gs_gradientAccumulator.grad_mean_y, toFixed(mean_d.y));
        InterlockedAdd(gs_gradientAccumulator.grad_mean_z, toFixed(mean_d.z));

        InterlockedAdd(gs_gradientAccumulator.grad_mean2d_magnitude, toFixed(length(mean2d_d)));
        //InterlockedAdd(gs_gradientAccumulator.grad_mean2d_y, toFixed(mean2d_d.y));
    }

    if (bool(/*$(Variable:descend_scale)*/))
    {
        InterlockedAdd(gs_gradientAccumulator.grad_log_scale_x, toFixed(scale_d.x));
        InterlockedAdd(gs_gradientAccumulator.grad_log_scale_y, toFixed(scale_d.y));
        InterlockedAdd(gs_gradientAccumulator.grad_log_scale_z, toFixed(scale_d.z));
    }

    if (bool(/*$(Variable:descend_color)*/))
    {
        InterlockedAdd(gs_gradientAccumulator.grad_color_r, toFixed(col_d.r));
        InterlockedAdd(gs_gradientAccumulator.grad_color_g, toFixed(col_d.g));
        InterlockedAdd(gs_gradientAccumulator.grad_color_b, toFixed(col_d.b));
    }

    if (bool(/*$(Variable:descend_opacity)*/))
    {
        InterlockedAdd(gs_gradientAccumulator.grad_opacity, toFixed(opacity_d));
    }

    if (bool(/*$(Variable:descend_rotation)*/))
    {
        InterlockedAdd(gs_gradientAccumulator.grad_quat_x, toFixed(rotation_d.x));
        InterlockedAdd(gs_gradientAccumulator.grad_quat_y, toFixed(rotation_d.y));
        InterlockedAdd(gs_gradientAccumulator.grad_quat_z, toFixed(rotation_d.z));
        InterlockedAdd(gs_gradientAccumulator.grad_quat_w, toFixed(rotation_d.w));
    }
}

void accumulate_gradients_global(uint gaussian_idx)
{
    if (bool(/*$(Variable:descend_mean)*/))
    {
        InterlockedAdd(mean_gradient_accum_x[gaussian_idx], gs_gradientAccumulator.grad_mean_x);
        InterlockedAdd(mean_gradient_accum_y[gaussian_idx], gs_gradientAccumulator.grad_mean_y);
        InterlockedAdd(mean_gradient_accum_z[gaussian_idx], gs_gradientAccumulator.grad_mean_z);

        InterlockedAdd(mean2d_gradient_accum_magnitude[gaussian_idx], gs_gradientAccumulator.grad_mean2d_magnitude);
        //InterlockedAdd(mean2d_gradient_accum_y[gaussian_idx], gs_gradientAccumulator.grad_mean2d_y);
    }

    if (bool(/*$(Variable:descend_scale)*/))
    {
        InterlockedAdd(scale_gradient_accum_x[gaussian_idx], gs_gradientAccumulator.grad_log_scale_x);
        InterlockedAdd(scale_gradient_accum_y[gaussian_idx], gs_gradientAccumulator.grad_log_scale_y);
        InterlockedAdd(scale_gradient_accum_z[gaussian_idx], gs_gradientAccumulator.grad_log_scale_z);
    }

    if (bool(/*$(Variable:descend_color)*/))
    {
        InterlockedAdd(color_gradient_accum_r[gaussian_idx], gs_gradientAccumulator.grad_color_r);
        InterlockedAdd(color_gradient_accum_g[gaussian_idx], gs_gradientAccumulator.grad_color_g);
        InterlockedAdd(color_gradient_accum_b[gaussian_idx], gs_gradientAccumulator.grad_color_b);
    }

    if (bool(/*$(Variable:descend_opacity)*/))
    {
        InterlockedAdd(opacity_gradient_accum[gaussian_idx], gs_gradientAccumulator.grad_opacity);
    }

    if (bool(/*$(Variable:descend_rotation)*/))
    {
        InterlockedAdd(rotation_gradient_accum_x[gaussian_idx], gs_gradientAccumulator.grad_quat_x);
        InterlockedAdd(rotation_gradient_accum_y[gaussian_idx], gs_gradientAccumulator.grad_quat_y);
        InterlockedAdd(rotation_gradient_accum_z[gaussian_idx], gs_gradientAccumulator.grad_quat_z);
        InterlockedAdd(rotation_gradient_accum_w[gaussian_idx], gs_gradientAccumulator.grad_quat_w);
    }
}
