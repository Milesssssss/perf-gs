// Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

// Execution constants shared by the kernels; mirrored by trainer.backend.compute.execution.

#ifndef TRAINER_EXECUTION_SLANG
#define TRAINER_EXECUTION_SLANG

static const uint TILE_WIDTH = 8;
static const uint TILE_HEIGHT = 8;
static const uint TILE_AREA = 64;
static const uint CLUSTER_SIZE = 128;
static const uint RADIX_DIGITS = 256;
static const uint RADIX_KEYS_PER_THREAD = 8;
static const uint RADIX_PASSES = 8;
static const uint RADIX_PART_SIZE = 2048;
static const uint RADIX_GLOBAL_HIST_SIZE = 2048;
static const uint SCAN_BLOCK_SIZE = 1024;
static const uint RASTER_BUCKET_SIZE = 32;

#endif // TRAINER_EXECUTION_SLANG
