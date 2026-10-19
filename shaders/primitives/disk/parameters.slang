// Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

#ifndef TRAINER_DISK_PARAMETERS_SLANG
#define TRAINER_DISK_PARAMETERS_SLANG

#include "types.slang"
#include "../../optimizers/primitive_adam.slang"

struct DiskParameterStorage : IPrimitiveParameterStorage<DiskChannel::Count>
{
    RWStructuredBuffer<Disk2D> primitives;

    PrimitiveValues<DiskChannel::Count> loadValues(uint index)
    {
        Disk2D primitive = primitives[index];
        PrimitiveValues<DiskChannel::Count> values;
        ChannelCursor<DiskChannel::Count> cursor;
        cursor.offset = 0u;
        cursor.write2(values.data, primitive.center);
        cursor.write1(values.data, primitive.log_radius);
        cursor.write1(values.data, primitive.opacity);
        cursor.write3(values.data, primitive.color_value);
        return values;
    }

    void storeValues(uint index, PrimitiveValues<DiskChannel::Count> values)
    {
        Disk2D primitive = primitives[index];
        ChannelCursor<DiskChannel::Count> cursor;
        cursor.offset = 0u;
        primitive.center = cursor.read2(values.data);
        primitive.log_radius = cursor.read1(values.data);
        primitive.opacity = cursor.read1(values.data);
        primitive.color_value = cursor.read3(values.data);
        primitives[index] = primitive;
    }
}

#endif // TRAINER_DISK_PARAMETERS_SLANG
