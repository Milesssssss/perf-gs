// Soft Rasterizer technique, shader compact_gaussian_buffer
/*$(ShaderResources)*/

/*$(_compute:compactScatter)*/(uint3 DTid : SV_DispatchThreadID)
{
    uint idx = DTid.x;
    if (idx >= num_gaussians_counter[0]) return;
    
    if (valid_mask[idx] == 1)
    {
        uint out_idx = prefix_sum[idx];
        compacted_gaussian_buffer[out_idx] = input_gaussians[idx];
        // If we dont do this the adam params will be out of sync with the actual gaussian buffer
        compacted_adam_state[out_idx] = adam_state[idx];
    }
}

/*
Shader Resources:
	Buffer input_gaussians (as UAV)
	Buffer num_gaussians_counter (as UAV)
*/
