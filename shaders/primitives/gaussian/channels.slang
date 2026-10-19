// Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

#ifndef TRAINER_GAUSSIAN_CHANNELS_SLANG
#define TRAINER_GAUSSIAN_CHANNELS_SLANG

#include "../../common/types.slang"   // toFixed / fromFixed fixed-point gradients

// GaussianChannel is generated with Gaussian3D in gaussian_types.slang,
// included transitively through common/types.slang above. This file owns only
// the typed writer behavior, not channel offsets or counts.

// Typed semantic writer used by every Gaussian backward/regularization pass.
// Physical storage is channel-major SoA:
//     gradients[channel * channel_stride + primitive_index].
struct GaussianGradientWriter
{
    RWStructuredBuffer<int64_t> gradients;
    uint channel_stride;

    uint address(uint channel, uint index)
    {
        return channel * channel_stride + index;
    }

    void addScalar(uint channel, uint index, float value)
    {
        InterlockedAdd(gradients[address(channel, index)], toFixed(value));
    }

    void storeScalar(uint channel, uint index, float value)
    {
        gradients[address(channel, index)] = toFixed(value);
    }

    void addMean(uint index, float3 value)
    {
        addScalar(GaussianChannel::Mean + 0u, index, value.x);
        addScalar(GaussianChannel::Mean + 1u, index, value.y);
        addScalar(GaussianChannel::Mean + 2u, index, value.z);
    }

    void storeMean(uint index, float3 value)
    {
        storeScalar(GaussianChannel::Mean + 0u, index, value.x);
        storeScalar(GaussianChannel::Mean + 1u, index, value.y);
        storeScalar(GaussianChannel::Mean + 2u, index, value.z);
    }

    void addLogScale(uint index, float3 value)
    {
        addScalar(GaussianChannel::LogScale + 0u, index, value.x);
        addScalar(GaussianChannel::LogScale + 1u, index, value.y);
        addScalar(GaussianChannel::LogScale + 2u, index, value.z);
    }

    void storeLogScale(uint index, float3 value)
    {
        storeScalar(GaussianChannel::LogScale + 0u, index, value.x);
        storeScalar(GaussianChannel::LogScale + 1u, index, value.y);
        storeScalar(GaussianChannel::LogScale + 2u, index, value.z);
    }

    void addRotation(uint index, float4 value)
    {
        addScalar(GaussianChannel::Rotation + 0u, index, value.x);
        addScalar(GaussianChannel::Rotation + 1u, index, value.y);
        addScalar(GaussianChannel::Rotation + 2u, index, value.z);
        addScalar(GaussianChannel::Rotation + 3u, index, value.w);
    }

    void storeRotation(uint index, float4 value)
    {
        storeScalar(GaussianChannel::Rotation + 0u, index, value.x);
        storeScalar(GaussianChannel::Rotation + 1u, index, value.y);
        storeScalar(GaussianChannel::Rotation + 2u, index, value.z);
        storeScalar(GaussianChannel::Rotation + 3u, index, value.w);
    }

    void addOpacity(uint index, float value)
    {
        addScalar(GaussianChannel::Opacity, index, value);
    }

    void addColor(uint index, float3 value)
    {
        addScalar(GaussianChannel::Color + 0u, index, value.x);
        addScalar(GaussianChannel::Color + 1u, index, value.y);
        addScalar(GaussianChannel::Color + 2u, index, value.z);
    }
}

#endif // TRAINER_GAUSSIAN_CHANNELS_SLANG
