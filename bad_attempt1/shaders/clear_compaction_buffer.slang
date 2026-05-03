// Soft Rasterizer technique, shader compact_gaussian_buffer
/*$(ShaderResources)*/

/*$(_compute:clearPass)*/(uint3 DTid : SV_DispatchThreadID)
{
    uint idx = DTid.x;

    if (idx == 0)
        compacted_num_gaussians_counter[0] = 0;

    compacted_gaussian_buffer[idx] = (Struct_Gaussian3D)0;

}

/*
Shader Resources:
	Buffer input_gaussians (as UAV)
	Buffer num_gaussians_counter (as UAV)
*/
