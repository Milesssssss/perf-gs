// Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

// Front-to-back compositing (extinction) operator for the tile blend loop, orthogonal to
// the primitive's coverage and color (core/interfaces.slang).

#ifndef TRAINER_CORE_COMPOSITOR_SLANG
#define TRAINER_CORE_COMPOSITOR_SLANG

// Host sets 1 when hardness is fixed at 1.0 (alpha-over), compiling out the pow paths whose
// live registers lower occupancy in the hottest backward loop.
#ifndef COMPOSITOR_HARDNESS_ONE
#define COMPOSITOR_HARDNESS_ONE 0
#endif

struct Compositor
{
    float hardness;   // k >= 1; 1 = alpha-over

    // Light passing through this splat: T <- T * transmittance(a).
    float transmittance(float a)
    {
#if COMPOSITOR_HARDNESS_ONE
        return 1.0 - a;
#else
        if (hardness == 1.0) return 1.0 - a;             // exact alpha-over
        return pow(1.0 - a, hardness);
#endif
    }

    // This splat's color-contribution weight: C += color * weight(a) * T.
    float weight(float a)
    {
#if COMPOSITOR_HARDNESS_ONE
        return a;
#else
        if (hardness == 1.0) return a;                   // exact alpha-over
        return 1.0 - pow(1.0 - a, hardness);
#endif
    }

    float coverageGrad(float a, float dLdc, float T_i,
                       float dLdCbehind, float dL_dA, float T_final)
    {
#if COMPOSITOR_HARDNESS_ONE
        {
            float inv_one_minus_a = 1.0 / (1.0 - a);
            return dLdc * T_i + (dL_dA * T_final - dLdCbehind) * inv_one_minus_a;
        }
#else
        if (hardness == 1.0)
        {
            // alpha-over: weight'=1, transmittance'/transmittance = -1/(1-a)
            float inv_one_minus_a = 1.0 / (1.0 - a);
            return dLdc * T_i + (dL_dA * T_final - dLdCbehind) * inv_one_minus_a;
        }
        float beta = 1.0 - a;
        float dWeight = hardness * pow(beta, hardness - 1.0);   // weight'(a)
        float transLogGrad = -hardness / beta;                 // t'(a)/t(a)
        return dWeight * dLdc * T_i + transLogGrad * (dLdCbehind - dL_dA * T_final);
#endif
    }
};

#endif // TRAINER_CORE_COMPOSITOR_SLANG
