/*$(ShaderResources)*/

static uint max_number_gaussians = /*$(Variable:max_number_gaussians)*/;

/*$(_compute:clearGaussianPixelCounts)*/(uint3 DTid : SV_DispatchThreadID)
{
    if (DTid.x >= max_number_gaussians)
        return;

    if ((training_step_counter[0]) % 6 == 0) 
        pixel_count_per_gaussian_accum[DTid.x] = 0;
}