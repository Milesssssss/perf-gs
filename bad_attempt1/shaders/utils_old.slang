//I know this may not work as its an include file and maybe these cannot get populated by GIGI if its not actually set 
//as a shader file in the editor?

//differentiable util functions

// many util functions around gaussian math are taken directly from https://github.com/electronicarts/ShaderToHuman/blob/master/examples/GaussianSplatting/SplatCommon.hlsl 

#define UINT_MAX 0xffffffff

[Differentiable]
[PreferRecompute]
float2x2 invert2x2(float2x2 m)
{
    float a = m[0][0], b = m[0][1], c = m[1][0], d = m[1][1];
    return (1.0 / (a * d - b * c)) * float2x2(d, -b, -c, a);
}

[Differentiable]
[PreferRecompute]
float logit(float x)
{
    return log(x / (1.0f - x));
}

[Differentiable]
[PreferRecompute]
float3 logit3(float3 a)
{
    return float3(logit(a.x), logit(a.y), logit(a.z));
}

[Differentiable]
[PreferRecompute]
float sigmoid(float x)
{
    return 1.0 / (1.0 + exp(-x));
}

[Differentiable]
[PreferRecompute]
float3 sigmoid3(float3 x)
{
    return float3(sigmoid(x.r), sigmoid(x.g), sigmoid(x.b));
}

[Differentiable]
float loss(float3 rendered, float3 target)
{
    return dot(abs(rendered - target), float3(1, 1, 1));
}

[Differentiable]
float loss_l2(float3 rendered, float3 target)
{
    float3 diff = rendered - target;
    return dot(diff, diff);
}

[Differentiable]
[PreferRecompute]
float3x3 buildCovariance3D(float3 scale, float4 quat)
{
    float3x3 R = matrixFromQuaternion(quat);
    float3x3 S = float3x3(scale.x, 0, 0,
                          0, scale.y, 0,
                          0, 0, scale.z);

    float3x3 M = mul(R, S);
    return mul(M, transpose(M));
}

// TODO: Improve readability of this function
[Differentiable]
[PreferRecompute]
void projectGaussian3DTo2D_diff(
    float3 mean,
    float3 log_scale,
    float4 quat,
    no_diff float4x4 world_to_view,
    no_diff float4x4 view_to_screen,
    no_diff float2 frame_dim,
    out float2 mean_2d,
    out float3 cov2d,
    out float depth_vs)
{
    // View space
    float3 mean_vs = mul(world_to_view, float4(mean, 1.0)).xyz;
    depth_vs = mean_vs.z;

    // Clip space (for mean projection only)
    float4 mean_clip = mul(view_to_screen, float4(mean_vs, 1.0));
    float inv_w = 1.0 / max(mean_clip.w, 1e-6);
    float2 mean_ndc = mean_clip.xy * inv_w;
    float2 mean_normalized = (mean_ndc * 0.5 + 0.5);
    mean_2d = float2(mean_normalized.x, mean_normalized.y) * frame_dim;

    // 3D covariance in view space
    float3 scale = exp(log_scale);
    float3x3 Sigma3D = buildCovariance3D(scale, quat);

    float3x3 W = float3x3(
        world_to_view[0].xyz,
        world_to_view[1].xyz,
        world_to_view[2].xyz
    );
    float3x3 Sigma_view = mul(mul(W, Sigma3D), transpose(W));

    float focal_x = view_to_screen[0][0] * frame_dim.x * 0.5;
    float focal_y = view_to_screen[1][1] * frame_dim.y * 0.5;

    float x = mean_vs.x;
    float y = mean_vs.y;
    float z = mean_vs.z;
    float z2 = z * z;

    // CORRECTED JACOBIAN for OpenGL projection (clip.w = -z)
    // d(screen.x)/d(view.x) = -focal_x / z  (not +)
    // d(screen.x)/d(view.z) = +focal_x * x / z^2 (not -)
    // d(screen.y)/d(view.y) = -focal_y / z
    // d(screen.y)/d(view.z) = +focal_y * y / z^2

    float2 dsdX = float2(-focal_x / z, 0.0);                  // FIXED: was positive
    float2 dsdY = float2(0.0, -focal_y / z);                  // unchanged
    float2 dsdZ = float2(focal_x * x / z2, focal_y * y / z2); // FIXED: x term was negative

    // J * Sigma_view * J^T
    float cov_00 = dsdX.x * (Sigma_view[0][0] * dsdX.x + Sigma_view[0][1] * dsdY.x + Sigma_view[0][2] * dsdZ.x)
                 + dsdY.x * (Sigma_view[1][0] * dsdX.x + Sigma_view[1][1] * dsdY.x + Sigma_view[1][2] * dsdZ.x)
                 + dsdZ.x * (Sigma_view[2][0] * dsdX.x + Sigma_view[2][1] * dsdY.x + Sigma_view[2][2] * dsdZ.x);

    float cov_01 = dsdX.x * (Sigma_view[0][0] * dsdX.y + Sigma_view[0][1] * dsdY.y + Sigma_view[0][2] * dsdZ.y)
                 + dsdY.x * (Sigma_view[1][0] * dsdX.y + Sigma_view[1][1] * dsdY.y + Sigma_view[1][2] * dsdZ.y)
                 + dsdZ.x * (Sigma_view[2][0] * dsdX.y + Sigma_view[2][1] * dsdY.y + Sigma_view[2][2] * dsdZ.y);

    float cov_11 = dsdX.y * (Sigma_view[0][0] * dsdX.y + Sigma_view[0][1] * dsdY.y + Sigma_view[0][2] * dsdZ.y)
                 + dsdY.y * (Sigma_view[1][0] * dsdX.y + Sigma_view[1][1] * dsdY.y + Sigma_view[1][2] * dsdZ.y)
                 + dsdZ.y * (Sigma_view[2][0] * dsdX.y + Sigma_view[2][1] * dsdY.y + Sigma_view[2][2] * dsdZ.y);

    cov2d = float3(cov_00 + 0.3, cov_01, cov_11 + 0.3);
}

[Differentiable]
float evalGaussian2DFromCov(no_diff float2 pixel_pos, float2 mean_2d, float3 cov2d, float opacity)
{
    float2 d = pixel_pos - mean_2d;
    float det = cov2d.x * cov2d.z - cov2d.y * cov2d.y;
    det = max(det, 1e-20);

    float inv_det = 1.0 / det;
    float cov_inv_00 = cov2d.z * inv_det;
    float cov_inv_01 = -cov2d.y * inv_det;
    float cov_inv_11 = cov2d.x * inv_det;

    float power = -0.5 * (d.x * d.x * cov_inv_00 +
                          2.0 * d.x * d.y * cov_inv_01 +
                          d.y * d.y * cov_inv_11);

    return opacity * exp(power);
}

[Differentiable]
float evalGaussian3DAtPixel(
    no_diff float2 pixel,
    float3 mean,
    float3 log_scale,
    float4 quat,
    float opacity,
    no_diff float4x4 world_to_view,
    no_diff float4x4 view_to_screen,
    no_diff float2 frame_dim)
{
    float3 cov2d;
    float depth;
    float2 mean_2d;

    projectGaussian3DTo2D_diff(
        mean, log_scale, quat,
        world_to_view, view_to_screen, frame_dim,
        mean_2d, cov2d, depth
    );

    return evalGaussian2DFromCov(pixel, mean_2d, cov2d, opacity);
}


//Non-differentiable util functions

int toFixed(float f)
{
    return int(f * FIXED_SCALE);
}

float fromFixed(int i)
{
    return float(i) * FIXED_INV_SCALE;
}

float3 rgbToLogit(float3 rgb) {
    rgb = clamp(rgb, 1e-5f, 1.0f - 1e-5f); // avoid inf
    return log(rgb / (1.0f - rgb));        // inverse sigmoid
}

float rgbToLogit(float rgb) {
    rgb = clamp(rgb, 1e-5f, 1.0f - 1e-5f); // avoid inf
    return log(rgb / (1.0f - rgb));        // inverse sigmoid
}

// https://gist.github.com/fboldog/a76648e1580b88ca9957f60b20edf49c
float random(uint seed)
{
    seed = (seed ^ 61) ^ (seed >> 16);
    seed *= 9;
    seed = seed ^ (seed >> 4);
    seed *= 0x27d4eb2d;
    seed = seed ^ (seed >> 15);
    return float(seed) / UINT_MAX;
}

float rand2(int2 coords)
{
    return random(coords.x * 73856093u ^ coords.y * 19349663u);
}

float3 log3(float3 v)
{
    return float3(log(v.x), log(v.y), log(v.z));
}

float3 index_to_random_color_rgb(uint index)
{
    float r = rand2(int2(index + 0, index + 1));
    float g = rand2(int2(index + 2, index + 3));
    float b = rand2(int2(index + 4, index + 5));
    return float3(r, g, b);
}

float gaussian_random(uint seed)
{
    float u1 = random(seed);
    float u2 = random(seed + 1);
    return sqrt(-2.0 * log(u1 + 1e-10)) * cos(6.28318530718 * u2);
}

// Adapted from https://github.com/electronicarts/ShaderToHuman/blob/master/examples/GaussianSplatting/SplatCommon.hlsl 
[Differentiable]
[PreferRecompute]
float3x3 matrixFromQuaternion(float4 q)
{
    float3x3 m;

    // has positive effect on some content
    q = normalize(q);

    float r = q.x;
    float x = q.y;
    float y = q.z;
    float z = q.w;

    // Compute rotation matrix from quaternion
    m = float3x3(
        1.f - 2.f * (y * y + z * z), 2.f * (x * y - r * z), 2.f * (x * z + r * y),
        2.f * (x * y + r * z), 1.f - 2.f * (x * x + z * z), 2.f * (y * z - r * x),
        2.f * (x * z - r * y), 2.f * (y * z + r * x), 1.f - 2.f * (x * x + y * y));

    return m;
}


