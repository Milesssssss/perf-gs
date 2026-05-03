// Soft Rasterizer technique, shader compact_gaussian_buffer
/*$(ShaderResources)*/

/*$(_compute:updatePass)*/(uint3 DTid : SV_DispatchThreadID)
{
    uint idx = DTid.x;
    if (idx >= compacted_num_gaussians_counter[0])
        return;

    if (idx == 0)
        num_gaussians_counter[0] = compacted_num_gaussians_counter[0];

    input_gaussians[idx] = compacted_gaussian_buffer[idx];
    adam_state[idx] = compacted_adam_state[idx];
}

/*
Shader Resources:
	Buffer input_gaussians (as UAV)
	Buffer num_gaussians_counter (as UAV)
*/
