// Unnamed technique, shader fillBuffer3D
/*$(ShaderResources)*/

static const uint input_num_gaussians = /*$(Variable:input_num_gaussians)*/;

static float2 camera_pos = /*$(Variable:cameraPos)*/;
static float2 scale = /*$(Variable:scale)*/;
static float2 frame_dim = /*$(Variable:frameDim)*/;
static const float FIXED_SCALE = 65536.0;
static const float FIXED_INV_SCALE = 1.0 / FIXED_SCALE;

#include "utils.slang"

/*$(_compute:fillBuffer)*/ (uint3 DTid: SV_DispatchThreadID)
{
    uint idx = DTid.x;
    if (idx >= input_num_gaussians) return;
    if (idx == 0) training_step_counter[0] = 0;

    // 8x8x4 grid
    uint x = idx % 8;
    uint y = (idx / 8) % 8;
    uint z = idx / 64;

    input_gaussians[idx].mean = float3(x - 3.5, y - 3.5, z + 4.0) * 0.5;
    input_gaussians[idx].log_scale = log(10.8);//log(/*$(Variable:input_gaussian_scale_xyz)*/)
    input_gaussians[idx].rotation_quat = /*$(Variable:input_gaussian_rotation_xyzw)*/;
    input_gaussians[idx].opacity = /*$(Variable:input_gaussian_opacity_a)*/;
    input_gaussians[idx].color_logit = logit3(float3(x / 7.0, y / 7.0, z / 3.0) * 0.8 + 0.1); //  logit3(/*$(Variable:input_gaussian_color_rgb)*/)

    InterlockedAdd(num_gaussians_counter[0], 1);
}

/*
Shader Resources:
	Buffer input_gaussians_3d (as UAV)
*/