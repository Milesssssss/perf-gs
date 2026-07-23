#ifndef TRAINER_GAUSSIAN_PARAMETERS_SLANG
#define TRAINER_GAUSSIAN_PARAMETERS_SLANG

#include "../../common/types.slang"
#include "../../optimizers/primitive_adam.slang"
#include "channels.slang"

// Only the primitive-value mapping is Gaussian-specific. Optimizer gradients,
// moments, learning rates, update math, and lifecycle live in AdamPlanes/AdamPass.
struct GaussianParameterStorage : IPrimitiveParameterStorage<GaussianChannel::Count>
{
    RWStructuredBuffer<Gaussian3D> primitives;

    PrimitiveValues<GaussianChannel::Count> loadValues(uint index)
    {
        Gaussian3D primitive = primitives[index];
        PrimitiveValues<GaussianChannel::Count> values;
        ChannelCursor<GaussianChannel::Count> cursor;
        cursor.offset = 0u;
        cursor.write3(values.data, primitive.mean);
        cursor.write3(values.data, primitive.log_scale);
        cursor.write4(values.data, primitive.rot_quat);
        cursor.write1(values.data, primitive.opacity);
        cursor.write3(values.data, primitive.sh0);
        return values;
    }

    void storeValues(uint index, PrimitiveValues<GaussianChannel::Count> values)
    {
        // Preserve future non-optimized fields.
        Gaussian3D primitive = primitives[index];
        ChannelCursor<GaussianChannel::Count> cursor;
        cursor.offset = 0u;
        primitive.mean = cursor.read3(values.data);
        primitive.log_scale = cursor.read3(values.data);
        primitive.rot_quat = cursor.read4(values.data);
        primitive.opacity = cursor.read1(values.data);
        primitive.sh0 = cursor.read3(values.data);
        normalizeGaussian3D(primitive);
        primitives[index] = primitive;
    }
}

#endif // TRAINER_GAUSSIAN_PARAMETERS_SLANG
