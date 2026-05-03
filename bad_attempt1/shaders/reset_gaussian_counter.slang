// Soft Rasterizer technique, shader compact_gaussian_buffer
/*$(ShaderResources)*/

/*$(_compute:resetCounter)*/(uint DTid : SV_DispatchThreadID)
{
	//Best shader ever
	uint idx = DTid;
	if (idx == 0)
		num_gaussians_counter[0] = 0;

}

/*
Shader Resources:
	Buffer input_gaussians (as UAV)
	Buffer num_gaussians_counter (as UAV)
*/
