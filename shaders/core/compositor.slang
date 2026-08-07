// The compositing / extinction operator for the tile blend loop.
//
// Orthogonal to the primitive: IBlendable (core/interfaces.slang) says what a
// splat looks like at a pixel (coverage + color); the Compositor says how those
// accumulate front-to-back. 

#ifndef TRAINER_CORE_COMPOSITOR_SLANG
#define TRAINER_CORE_COMPOSITOR_SLANG

// Compile-time specialization: when the run's extinction hardness is fixed at
// 1.0 (the default alpha-over compositor — the schedule target is <= 1), the
// host injects COMPOSITOR_HARDNESS_ONE=1 so the `pow`-based hardened-extinction
// paths compile out entirely. Those dead branches otherwise keep `pow` calls +
// their registers live in the hottest backward loop (coverageGrad runs per pixel
// per splat), measurably lowering occupancy. Defaults to 0 (full runtime path)
// so a hardness>1 run is bit-identical to before. Set via
// ShaderSpecialization.compiler_defines (see app.py).
#ifndef COMPOSITOR_HARDNESS_ONE
#define COMPOSITOR_HARDNESS_ONE 0
#endif

struct Compositor
{
    float hardness;   // k >= 1; 1 = alpha-over

    // Light passing THROUGH this splat: T <- T * transmittance(a).
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
            float oneMinusA = 1.0 - a;
            return dLdc * T_i - dLdCbehind / oneMinusA + dL_dA * (T_final / oneMinusA);
        }
#else
        if (hardness == 1.0)
        {
            // alpha-over: weight'=1, transmittance'/transmittance = -1/(1-a).
            // Exact 3DGS backward (bit-identical to the previous hardcode).
            float oneMinusA = 1.0 - a;
            return dLdc * T_i - dLdCbehind / oneMinusA + dL_dA * (T_final / oneMinusA);
        }
        float beta = 1.0 - a;
        float dWeight = hardness * pow(beta, hardness - 1.0);   // weight'(a)
        float transLogGrad = -hardness / beta;                 // t'(a)/t(a)
        //maybe could be clearer mathematically if we had dL/dai = dL_dC * (ci*dWeight*Ti + t'(ai)/t(ai)*Cbehind))
        return dWeight * dLdc * T_i + transLogGrad * (dLdCbehind - dL_dA * T_final);
#endif
    }
};

#endif // TRAINER_CORE_COMPOSITOR_SLANG
