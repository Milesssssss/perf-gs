[shader("compute")]
[numthreads(1, 1, 1)]
void snapshotCount(
    StructuredBuffer<uint> src,
    RWStructuredBuffer<uint> dst)
{
    dst[0] = src[0];
}

[shader("compute")]
[numthreads(1, 1, 1)]
void prepareDensifyArgs(
    StructuredBuffer<uint> count_buf,
    RWStructuredBuffer<uint> dispatch_args,
    uniform uint max_splats)   // defensive cap
{
    uint n = count_buf[0];
    uint groups = (n + 255u) / 256u;
    if (groups == 0u) groups = 1u;
    // Clamp so a corrupted count_buf can never queue an insane number of groups.
    uint max_groups = (max_splats + 255u) / 256u;
    if (max_groups == 0u) max_groups = 1u;
    groups = min(groups, max_groups);
    dispatch_args[0] = groups;
    dispatch_args[1] = 1u;
    dispatch_args[2] = 1u;
}
