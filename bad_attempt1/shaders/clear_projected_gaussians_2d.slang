/*$(ShaderResources)*/

static uint max_number_gaussians = /*$(Variable:max_number_gaussians)*/;

/*$(_compute:clearProjectedGaussians)*/(uint3 DTid : SV_DispatchThreadID)
{
    if (DTid.x >= max_number_gaussians)
        return;

    projected_gaussians[DTid.x] = (Struct_Projected2DGaussian)(0);
}