// Zero the per-gaussian importance score accumulator before a scoring pass.

[shader("compute")]
[numthreads(256, 1, 1)]
void clearImportance(
    uint3 dtid: SV_DispatchThreadID,
    RWStructuredBuffer<uint> importance_score_accum,
    uniform uint max_splats)
{
    uint i = dtid.x;
    if (i >= max_splats) return;
    importance_score_accum[i] = 0u;
}
