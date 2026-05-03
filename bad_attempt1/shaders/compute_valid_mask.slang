// Soft Rasterizer technique, shader compact_gaussian_buffer
/*$(ShaderResources)*/

/*$(_compute:computeValidMask)*/(uint3 DTid : SV_DispatchThreadID)
{
    uint idx = DTid.x;
    if (idx >= num_gaussians_counter[0]) return;
    
    valid_mask[idx] = (input_gaussians[idx].opacity > 0.0) ? 1u : 0u;
}

/*
Shader Resources:
	Buffer input_gaussians (as UAV)
	Buffer num_gaussians_counter (as UAV)
*/
