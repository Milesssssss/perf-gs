// Soft Rasterizer technique, shader update_params_accum_grads
/*$(ShaderResources)*/

static float threshold_start_value = /*$(Variable:splitting_threshold_start)*/;
static float threshold_end_value = /*$(Variable:splitting_threshold_end)*/;
static uint end_densification_iter = /*$(Variable:end_densification_iter)*/ / 6;
static uint current_iter = /*$(Variable:FrameIndex)*/;
static uint start_densification_iter = /*$(Variable:start_densification_iteration)*/ / 6;
static float2 camera_pos = /*$(Variable:cameraPos)*/;
static float2 scale = /*$(Variable:scale)*/;
static float2 frame_dim = /*$(Variable:frameDim)*/;
static bool enable_densification = /*$(Variable:enable_densification)*/;
static uint max_number_gaussians = /*$(Variable:max_number_gaussians)*/;

static const float FIXED_SCALE = 65536.0;
static const float FIXED_INV_SCALE = 1.0 / FIXED_SCALE;

#include "utils.slang"
#include "adam.slang"

GaussianGradients load_gradients(uint g_idx, float num_pixels)
{
    // TODO: defines or if/else to selectively load them
    GaussianGradients grad;

    float inv_num_views = 1.0 / 6.0; // Hardcoded for now for 6 views

    grad.mean = float3(
                    fromFixed(mean_gradient_accum_x[g_idx]),
                    fromFixed(mean_gradient_accum_y[g_idx]),
                    fromFixed(mean_gradient_accum_z[g_idx])) * inv_num_views;

    grad.mean2d_mag = fromFixed(mean2d_gradient_accum_magnitude[g_idx]);

    grad.scale = float3(
                     fromFixed(scale_gradient_accum_x[g_idx]),
                     fromFixed(scale_gradient_accum_y[g_idx]),
                     fromFixed(scale_gradient_accum_z[g_idx])
    ) * inv_num_views;

    grad.rotation = float4(fromFixed(rotation_gradient_accum_x[g_idx]),
                           fromFixed(rotation_gradient_accum_y[g_idx]),
                           fromFixed(rotation_gradient_accum_z[g_idx]),
                           fromFixed(rotation_gradient_accum_w[g_idx])) * inv_num_views;

    grad.color = float3(
                     fromFixed(color_gradient_accum_r[g_idx]),
                     fromFixed(color_gradient_accum_g[g_idx]),
                     fromFixed(color_gradient_accum_b[g_idx])) * inv_num_views;

    grad.opacity = fromFixed(opacity_gradient_accum[g_idx]) * inv_num_views;

    return grad;
}

void zero_gaussian_gradients(uint g_idx)
{
    //TODO: defines or if else to selectively zero them
    mean_gradient_accum_x[g_idx] = 0;
    mean_gradient_accum_y[g_idx] = 0;
    mean_gradient_accum_z[g_idx] = 0;

    mean2d_gradient_accum_magnitude[g_idx] = 0;

    scale_gradient_accum_x[g_idx] = 0;
    scale_gradient_accum_y[g_idx] = 0;
    scale_gradient_accum_z[g_idx] = 0;

    rotation_gradient_accum_x[g_idx] = 0;
    rotation_gradient_accum_y[g_idx] = 0;
    rotation_gradient_accum_z[g_idx] = 0;
    rotation_gradient_accum_w[g_idx] = 0;

    color_gradient_accum_r[g_idx]   = 0;
    color_gradient_accum_g[g_idx]   = 0;
    color_gradient_accum_b[g_idx]   = 0;

    opacity_gradient_accum[g_idx] = 0;
    //This should be already cleared in a separate shader
    pixel_coverage_count_per_gaussian[g_idx] = 0;
}

// https://arxiv.org/html/2503.14274v1 exponential decay of gradient threshold
float compute_grad_threshold(float threshold_start, float threshold_end, uint max_iters, uint current_iter)
{
    //The original paper usues 0.0001 and 0.0004, but for 2d it seems to work better with an order of magnitude smaller values
    return exp(log(0.0001) * (1 - (current_iter / float(max_iters))) + log(0.0004) * (current_iter / float(max_iters)));
}

bool should_densify(float grad_magnitude, uint current_iter)
{
    if (current_iter< start_densification_iter || current_iter % 100 != 0 || current_iter > end_densification_iter)
        return false;
    //return true;
    float threshold = 0.0002; 
    return grad_magnitude > threshold;
}

bool is_large_gaussian(uint g_idx)
{
    float3 splat_linear_scale = exp(input_gaussians[g_idx].log_scale);
    float max_scale = max(max(splat_linear_scale.x, splat_linear_scale.y), splat_linear_scale.z);
    float scale_threshold = 0.01f; // ← Increase 10x, tune to your scene scale
    return max_scale > scale_threshold;
}


void clone_gaussian(uint g_idx, uint current_iter)
{
    Struct_Gaussian3D parent = input_gaussians[g_idx];
    Struct_Gaussian3D clone = parent;

    float3 scale = exp(parent.log_scale);
    float3x3 rot_mat = matrixFromQuaternion(parent.rotation_quat);

    uint seed = g_idx + current_iter;

    float3 local_offset = float3(
        gaussian_random(seed) * scale.x,
        gaussian_random(seed + 1000) * scale.y,
        gaussian_random(seed + 2000) * scale.z
    );

    clone.opacity = 1.0 - sqrt(1.0 - parent.opacity);
    //clone.mean = parent.mean + mul(rot_mat, local_offset);
    //clone.opacity = parent.opacity * 0.5f;

    uint new_idx;
    if (num_gaussians_counter[0] < max_number_gaussians)
    {
        InterlockedAdd(num_gaussians_counter[0], 1, new_idx);
        if (new_idx < max_number_gaussians)
        {
            input_gaussians[new_idx] = clone;
            adam_state[new_idx] = zero_adam_state();
            zero_gaussian_gradients(new_idx);
        }
    }

    zero_gaussian_gradients(g_idx);
}


void split_gaussian(uint g_idx, uint seed_offset)
{
    Struct_Gaussian3D parent = input_gaussians[g_idx];

    float3 parent_linear_scale = exp(parent.log_scale);

    float3x3 rot_mat = matrixFromQuaternion(parent.rotation_quat);

    // Reduced scale
    float3 new_log_scale = log(parent_linear_scale * 0.625);
    // Like Gsplat does in the split function to avoid doubling density in the area
    float new_opacity = 1.0 - sqrt(1.0 - parent.opacity);

    Struct_AdamState zero_state = zero_adam_state();

    for (int i = 0; i < 2; i++) {
        
        Struct_Gaussian3D child = parent;

        child.log_scale = new_log_scale;
        child.opacity = new_opacity;

        uint seed = g_idx * 2 + i + seed_offset;

        float3 local_offset = float3(
            gaussian_random(seed)           * parent_linear_scale.x,
            gaussian_random(seed + 1000)    * parent_linear_scale.y,
            gaussian_random(seed + 2000)    * parent_linear_scale.z); //TODO: idk about this seed

        child.mean = parent.mean + mul(rot_mat, local_offset);

        if (i == 0)
        {
            // Replace parent with first splat
            input_gaussians[g_idx] = child;
            adam_state[g_idx] = zero_state;
            zero_gaussian_gradients(g_idx);
        }
        else
        {
            uint new_idx;
            if (num_gaussians_counter[0] < max_number_gaussians)
            {
                InterlockedAdd(num_gaussians_counter[0], 1, new_idx);
                if (new_idx < max_number_gaussians)
                {
                    input_gaussians[new_idx] = child;
                    adam_state[new_idx] = zero_state;
                    zero_gaussian_gradients(new_idx);
                }
            }
        }

    }

}

bool should_prune(uint g_idx, uint batch_iter)
{
    if (!(batch_iter % 100 == 0 && batch_iter >= start_densification_iter)) return false;

    Struct_Gaussian3D g = input_gaussians[g_idx];

    if (g.opacity < 0.005)
        return true;

    float3 scale = exp(g.log_scale);

    //This would better work in relative terms wrt to scene scale
    if (max(max(scale.x, scale.y), scale.z) > 0.15) //is this even valid now in 3d? 
        return true;

    return false;
}


/*$(_compute:main)*/(uint3 DTid : SV_DispatchThreadID)
{
    
    if (DTid.x >= num_gaussians_counter[0])
		return;

    uint g_idx = DTid.x;
    uint current_iter = training_step_counter[0];

    uint batch_iter = current_iter / 6;
    if (current_iter % 6 != 5) return;

	Struct_Gaussian3D in_g = input_gaussians[g_idx];

    float num_pixels = abs(pixel_coverage_count_per_gaussian[g_idx]);

	if(num_pixels < 1.0)
	    num_pixels = 1.0;

    GaussianGradients grad = load_gradients(g_idx, num_pixels);
    // grad.mean2d *= float2(frame_dim.x, frame_dim.y);
    //  Densification
    if (should_densify(grad.mean2d_mag /6.0f, batch_iter) && enable_densification)
    {
        if (is_large_gaussian(g_idx))
            split_gaussian(g_idx, batch_iter);
        else
            clone_gaussian(g_idx, batch_iter);

        return;
    }

    // Pruning
    if (should_prune(g_idx, batch_iter) && enable_densification)
    {
        // Mark for deletion by setting opacity to 0, maybe better if we set directly a valid mask here?
        // TODO: need compaction pass to remove these gaussians properly from the buffer
        in_g.opacity = 0.0;
        input_gaussians[g_idx] = in_g;
        adam_state[g_idx] = zero_adam_state();
        zero_gaussian_gradients(g_idx);
        return;
    }
    //
    // Opacity reset
    if (batch_iter % 3000 == 0 && batch_iter > 0 && enable_densification)
    {
        in_g.opacity = min(in_g.opacity, 0.01);
        Struct_AdamState state_reset = adam_state[g_idx];
        state_reset.opacity_m = 0.0f;
        state_reset.opacity_v = 0.0f;
        input_gaussians[g_idx] = in_g;
        adam_state[g_idx] = state_reset;
        //return;
    }

    // Now here we update the actual gaussian wrt to the gradients and ADAM
    Struct_AdamState state = adam_state[g_idx];

    // Hyeper param values from gsplat config
    //  Create config with bias correction for current step
    AdamConfig config = make_adam_config(
        1.6e-4, // lr_mean - OK
        2e-3,   // lr_scale - reduced from 5e-3
        1e-3,   // lr_rotation - OK
        2.5e-3, // lr_color - OK
        5e-2,   // lr_opacity - OK
        batch_iter + 1
    );

    adam_update_gaussian(
        in_g,
        grad,
        state,
        config,
        bool(/*$(Variable:descend_mean)*/),
        bool(/*$(Variable:descend_scale)*/),
        bool(/*$(Variable:descend_rotation)*/),
        bool(/*$(Variable:descend_color)*/),
        bool(/*$(Variable:descend_opacity)*/)
    );

    input_gaussians[g_idx]  = in_g;
    adam_state[g_idx]       = state;

    zero_gaussian_gradients(g_idx);
    
}



/*
Shader Resources:
	Buffer mean_gradient_accum_x (as UAV)
	Buffer mean_gradient_accum_y (as UAV)
	Buffer input_gaussians (as UAV)
*/
