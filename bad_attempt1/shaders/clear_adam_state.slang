/*$(ShaderResources)*/

static uint max_number_gaussians = /*$(Variable:max_number_gaussians)*/;

/*$(_compute:clearAdam)*/(uint3 DTid : SV_DispatchThreadID)
{
    if (DTid.x >= max_number_gaussians)
        return;

    adam_state[DTid.x] = (Struct_AdamState)0;
}