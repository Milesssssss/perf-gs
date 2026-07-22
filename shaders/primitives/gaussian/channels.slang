#ifndef TRAINER_GAUSSIAN_CHANNELS_SLANG
#define TRAINER_GAUSSIAN_CHANNELS_SLANG

#include "../../common/types.slang"   // toFixed / fromFixed fixed-point gradients

// The single semantic definition of the Gaussian's optimized scalar layout.
// rot_quat is stored (w, x, y, z), represented by float4.xyzw respectively.
namespace GaussianChannel
{
    static const uint Mean = 0u;       // 3: x, y, z
    static const uint LogScale = 3u;   // 3: x, y, z
    static const uint Rotation = 6u;   // 4: w, x, y, z
    static const uint Opacity = 10u;   // 1
    static const uint Color = 11u;     // 3: sh0 r, g, b
    static const int Count = 14;
}

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
