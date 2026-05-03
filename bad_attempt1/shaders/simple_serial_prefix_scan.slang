// Soft Rasterizer technique, shader compact_gaussian_buffer
/*$(ShaderResources)*/

/*$(_compute:prefixSum)*/(uint3 DTid : SV_DispatchThreadID)
{
    //TODO: Single thread computes prefix sum, do a parallel scan instead
    if (DTid.x != 0) return;
    
    uint sum = 0;
    uint count = num_gaussians_counter[0];
    for (uint i = 0; i < count; i++)
    {
        prefix_sum[i] = sum;
        sum += valid_mask[i];
    }
    compacted_num_gaussians_counter[0] = sum;
}

/*
Shader Resources:
	Buffer input_gaussians (as UAV)
	Buffer num_gaussians_counter (as UAV)
*/
