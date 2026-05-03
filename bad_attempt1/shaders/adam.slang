static const float ADAM_BETA1 = 0.9f;
static const float ADAM_BETA2 = 0.999f;
static const float ADAM_EPSILON = 1e-8f;

struct AdamConfig
{
    float lr_mean;
    float lr_scale;
    float lr_rotation;
    float lr_color;
    float lr_opacity;
    float bias_correction1;
    float bias_correction2;
};

struct GaussianGradients
{
    float3 mean;
    float mean2d_mag; //not used to update learnable params, just to load it and use for heuristics
    float3 scale;
    float4 rotation;
    float3 color;
    float opacity;
};


// Adam Step Functions,  we should make one single generic (?) Will GIGI be ok with it?

float adam_step_1d(
    float param,
    float grad,
    inout float m,
    inout float v,
    float lr,
    float bias_correction1,
    float bias_correction2
)
{
    m = ADAM_BETA1 * m + (1.0f - ADAM_BETA1) * grad;
    v = ADAM_BETA2 * v + (1.0f - ADAM_BETA2) * grad * grad;

    float m_hat = m / bias_correction1;
    float v_hat = v / bias_correction2;

    return param - lr * m_hat / (sqrt(v_hat) + ADAM_EPSILON);
}

float3 adam_step_3d(
    float3 param,
    float3 grad,
    inout float3 m,
    inout float3 v,
    float lr,
    float bias_correction1,
    float bias_correction2
)
{
    m = ADAM_BETA1 * m + (1.0f - ADAM_BETA1) * grad;
    v = ADAM_BETA2 * v + (1.0f - ADAM_BETA2) * grad * grad;

    float3 m_hat = m / bias_correction1;
    float3 v_hat = v / bias_correction2;

    return param - lr * m_hat / (sqrt(v_hat) + ADAM_EPSILON);
}

float4 adam_step_4d(
    float4 param,
    float4 grad,
    inout float4 m,
    inout float4 v,
    float lr,
    float bias_correction1,
    float bias_correction2
)
{
    m = ADAM_BETA1 * m + (1.0f - ADAM_BETA1) * grad;
    v = ADAM_BETA2 * v + (1.0f - ADAM_BETA2) * grad * grad;

    float4 m_hat = m / bias_correction1;
    float4 v_hat = v / bias_correction2;

    return param - lr * m_hat / (sqrt(v_hat) + ADAM_EPSILON);
}

//-------

AdamConfig make_adam_config(
    float lr_mean,
    float lr_scale,
    float lr_rotation,
    float lr_color,
    float lr_opacity,
    uint step
)
{
    AdamConfig config;
    config.lr_mean = lr_mean;
    config.lr_scale = lr_scale;
    config.lr_rotation = lr_rotation;
    config.lr_color = lr_color;
    config.lr_opacity = lr_opacity;
    config.bias_correction1 = 1.0f - pow(ADAM_BETA1, float(step));
    config.bias_correction2 = 1.0f - pow(ADAM_BETA2, float(step));
    return config;
}

Struct_AdamState zero_adam_state()
{
    Struct_AdamState s;
    s.mean_m = float3(0.0f, 0.0f, 0.0);
    s.mean_v = float3(0.0f, 0.0f, 0.0);
    s.scale_m = float3(0.0f, 0.0f, 0.0);
    s.scale_v = float3(0.0f, 0.0f, 0.0);
    s.rotation_m = float4(0,0,0,0);
    s.rotation_v = float4(0,0,0,0);
    s.color_m = float3(0.0f, 0.0f, 0.0f);
    s.color_v = float3(0.0f, 0.0f, 0.0f);
    s.opacity_m = 0.0f;
    s.opacity_v = 0.0f;
    return s;
}


// Gaussian Update

void adam_update_gaussian(
    inout Struct_Gaussian3D gaussian,
    in GaussianGradients grad,
    inout Struct_AdamState state,
    in AdamConfig config,
    bool descend_mean,
    bool descend_scale,
    bool descend_rotation,
    bool descend_color,
    bool descend_opacity
)
{
    if (descend_mean)
    {
        gaussian.mean = adam_step_3d(
            gaussian.mean,
            grad.mean,
            state.mean_m,
            state.mean_v,
            config.lr_mean,
            config.bias_correction1,
            config.bias_correction2
        );

    }

    if (descend_scale)
    {
        gaussian.log_scale = adam_step_3d(
            gaussian.log_scale,
            grad.scale,
            state.scale_m,
            state.scale_v,
            config.lr_scale,
            config.bias_correction1,
            config.bias_correction2
        );
        gaussian.log_scale = clamp(gaussian.log_scale, -6.0f, 2.0f);
    }

    if (descend_rotation)
    {

        gaussian.rotation_quat = adam_step_4d(
            gaussian.rotation_quat,
            grad.rotation,
            state.rotation_m,
            state.rotation_v,
            config.lr_rotation,
            config.bias_correction1,
            config.bias_correction2
        );
        //gaussian.rotation_quat = normalize(gaussian.rotation_quat);
    }

    if (descend_color)
    {
        gaussian.color_logit = adam_step_3d(
            gaussian.color_logit.rgb,
            grad.color,
            state.color_m,
            state.color_v,
            config.lr_color,
            config.bias_correction1,
            config.bias_correction2
        );
        //TODO: Clamping extreme values in logit, double check this!
        //gaussian.color_logit = clamp(gaussian.color_logit, -10.0f, 10.0f);
    }

    if (descend_opacity)
    {
        gaussian.opacity = adam_step_1d(
            gaussian.opacity,
            grad.opacity,
            state.opacity_m,
            state.opacity_v,
            config.lr_opacity,
            config.bias_correction1,
            config.bias_correction2
        );
        //TODO: probably should use logit space also for opacity
        gaussian.opacity = clamp(gaussian.opacity, 0.0f, 1.0f);
    }
}