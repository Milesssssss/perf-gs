// Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

// Commits a shared-arena scatter without copying unused capacity (counts stay GPU-resident). AoS records
// are a contiguous prefix; SoA records occupy the live prefix of each capacity-strided component.

#include "common/execution.slang"

static const uint COPY_GROUP_SIZE = 256u;

[shader("compute")]
[numthreads(1, 1, 1)]
void prepareActiveCopyArgs(
    StructuredBuffer<uint> count_buf,
    RWStructuredBuffer<uint> dispatch_args,
    uniform uint record_words,
    uniform uint max_splats)
{
    uint words = min(count_buf[0], max_splats) * record_words;
    uint groups = 1u + (max(words, 1u) - 1u) / COPY_GROUP_SIZE;
    dispatch_args[0] = min(groups, LINEAR_DISPATCH_WIDTH);
    dispatch_args[1] = (groups + LINEAR_DISPATCH_WIDTH - 1u) / LINEAR_DISPATCH_WIDTH;
    dispatch_args[2] = 1u;
}

[shader("compute")]
[numthreads(256, 1, 1)]
void copyActiveRecords(
    uint3 dtid: SV_DispatchThreadID,
    StructuredBuffer<uint> source,
    RWStructuredBuffer<uint> destination,
    StructuredBuffer<uint> count_buf,
    uniform uint record_words,
    uniform uint soa,
    uniform uint destination_word_offset,
    uniform uint max_splats)
{
    uint word = linearThreadIndex256(dtid);
    uint count = min(count_buf[0], max_splats);
    if (word >= count * record_words) return;
    uint offset = word;
    if (soa != 0u)
        offset = (word / count) * max_splats + word % count;
    destination[destination_word_offset + offset] = source[offset];
}