// The compositing / extinction operator for the tile blend loop.
//
// Orthogonal to the primitive: IBlendable (core/interfaces.slang) says what a
// splat looks like at a pixel (coverage + color); the Compositor says how those
// accumulate front-to-back. The loop advances per splat by:
//     C += color * weight(a) * T;    T *= transmittance(a);
// with `a` the primitive's per-pixel coverage in [0, 1].
//
// `hardness` k is a UNIFORM, so it is tunable per run with no recompile:
//     transmittance(a) = (1 - a)^k        weight(a) = 1 - (1 - a)^k
//   k = 1  -> standard alpha-over (the exact 3DGS blend; taken via a fast path
//             so the default is bit-identical to the hardcoded blend).
//   k > 1  -> sharper extinction: T decays faster, so the early-out fires
//             sooner and fewer occluded splats are shaded in BOTH the forward
//             and the backward. (Beer-Lambert with the optical depth scaled by
//             k: (1-a)^k = exp(-k * -ln(1-a)).)
//
// The transposed backward reconstructs T as a product of per-splat
// transmittance factors; coverageGrad() returns this splat's dLoss/dcoverage
// for whichever k, so all the extinction math lives in ONE place.

#ifndef TRAINER_CORE_COMPOSITOR_SLANG
#define TRAINER_CORE_COMPOSITOR_SLANG

struct Compositor
{
    float hardness;   // k >= 1; 1 = alpha-over

    // Light passing THROUGH this splat: T <- T * transmittance(a).
    float transmittance(float a)
    {
        if (hardness == 1.0) return 1.0 - a;             // exact alpha-over
        return pow(1.0 - a, hardness);
    }

    // This splat's color-contribution weight: C += color * weight(a) * T.
    float weight(float a)
    {
        if (hardness == 1.0) return a;                   // exact alpha-over
        return 1.0 - pow(1.0 - a, hardness);
    }

    // Per-splat dLoss/dcoverage, from the blend quantities the backward already
    // has in hand for this splat/pixel:
    //   dLdc        = dot(dL_dC, splatColor)   (color projected onto dL_dC)
    //   T_i         = transmittance in front of this splat
    //   dLdCbehind  = dot(dL_dC, C_behind)     (color accumulated behind it)
    //   dL_dA       = alpha-channel gradient seed
    //   T_final     = final transmittance of the pixel
    // From C = sum(color * weight * T) and A = 1 - prod(transmittance):
    //   dL/da = weight'(a) * dLdc * T_i
    //         + (transmittance'(a)/transmittance(a)) * (dLdCbehind - dL_dA * T_final).
    float coverageGrad(float a, float dLdc, float T_i,
                       float dLdCbehind, float dL_dA, float T_final)
    {
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
        return dWeight * dLdc * T_i + transLogGrad * (dLdCbehind - dL_dA * T_final);
    }
};

#endif // TRAINER_CORE_COMPOSITOR_SLANG
