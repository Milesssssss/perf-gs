/*$(ShaderResources)*/

/*$(_compute:clearTextures)*/(
    uint3 DTid : SV_DispatchThreadID,
)
{
    uint2 pixel = DTid.xy;
    gaussian_count_per_pixel[pixel] = 0;
}