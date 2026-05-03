// test_data.slang - Hardcoded COLMAP data for testing
/*$(ShaderResources)*/

static float2 camera_pos = /*$(Variable:cameraPos)*/;
static float2 scale = /*$(Variable:scale)*/;
static float2 frame_dim = /*$(Variable:frameDim)*/;
static const float FIXED_SCALE = 65536.0;
static const float FIXED_INV_SCALE = 1.0 / FIXED_SCALE;

#include "utils.slang"

// AMERA INTRINSICS (from cameras.txt)
static const float FOCAL_LENGTH = 2666.6666666666665;
static const float CX = 960.0;
static const float CY = 540.0;
static const float IMG_WIDTH = 1920.0;
static const float IMG_HEIGHT = 1080.0;

// This stuff is just to not write a colmap importer for now, easier to hardcode a simple test scene
static const float4 CAMERA_QUATS[6] = {
    float4(0.4999999999999995, 0.49999997814430636, -0.5000000000000006, 0.5000000218556947),        // Image 0
    float4(-3.0908621590457434e-08, 3.0908616554804224e-08, -0.707106723585394, 0.7071068387877057), // Image 1
    float4(0.4999999999999995, 0.49999997814430636, 0.5000000000000006, -0.5000000218556947),        // Image 2
    float4(0.7071068387877057, 0.707106723585394, 0.0, 0.0),                                         // Image 3
    float4(0.0, -4.371138828673793e-08, 0.0, 1.0),                                                   // Image 4
    float4(0.0, 1.0, 0.0, 0.0)     // Image 5
};

static const float3 CAMERA_TVECS[6] = {
    float3(-5.07484700867652e-07, 5.07484700867652e-07, 11.609896659851074),   // Image 0
    float3(1.014969401735304e-06, 1.8914923159397642e-06, 11.609896659851074), // Image 1
    float3(5.07484700867652e-07, 5.07484700867652e-07, 11.609896659851074),    // Image 2
    float3(0.0, 1.8914923159397642e-06, 11.609896659851074),                   // Image 3
    float3(-1.014969401735304e-06, 0.0, 11.609896659851074),                   // Image 4
    float3(0.0, 0.0, 11.609896659851074)    // Image 5
};

//This function is fucked, data is in blender format, then transformed to colmap and then to hlsl/dx12 notation
//ideally we have just two formats, i do this like this as its temporary as the data SHOULD NOT be in Blender format
float4x4 getViewMatrix(uint camera_idx)
{
    float4 q = CAMERA_QUATS[camera_idx];
    float3 t = CAMERA_TVECS[camera_idx];

    float3x3 R_blender = matrixFromQuaternion(q);

    // The data is actually stored in Blender world-space ffs, need to fix that...
    float3x3 B2C = float3x3(
        1, 0, 0,
        0, -1, 0,
        0, 0, -1
    );

    float3x3 R = transpose(mul(B2C, R_blender));
    float3 t_colmap = mul(B2C, t);

    return float4x4(
        float4(R[0], 0),
        float4(R[1], 0),
        float4(R[2], 0),
        float4(t_colmap, 1)
    );
}

float4x4 getProjectionMatrix()
{
    float n = 0.01;
    float f = 1000.0;

    float fx = FOCAL_LENGTH;
    float fy = FOCAL_LENGTH;

    return float4x4(
        float4(2.0 * fx / IMG_WIDTH, 0, 1.0 - 2.0 * CX / IMG_WIDTH, 0),
        float4(0, 2.0 * fy / IMG_HEIGHT, 2.0 * CY / IMG_HEIGHT - 1.0, 0),
        float4(0, 0, -(f + n) / (f - n), -2.0 * f * n / (f - n)),
        float4(0, 0, -1, 0)
    );
}

/*$(_compute:selectCameraAndTarget)*/(uint3 DTid: SV_DispatchThreadID)
{
    uint2 pixel = DTid.xy;

    if (pixel.x >= frame_dim.x || pixel.y >= frame_dim.y) return;

    uint step = training_step_counter[0];
    // for multiview select the step %6, for single image use var
    uint camera_idx = step % 6; 
    //camera_idx = /*$(Variable:cameraIndex)*/;
    
    if (DTid.x == 0 && DTid.y==0)
        selected_camera_index[0] = camera_idx;

    // Only thread 0 writes the camera matrix
    if (pixel.x == 0 && pixel.y == 0)
    {
        float4x4 view_mtx = transpose(getViewMatrix(camera_idx));
        float4x4 proj_mtx = getProjectionMatrix();

        target_camera_view_matrix[0] = view_mtx[0];
        target_camera_view_matrix[1] = view_mtx[1];
        target_camera_view_matrix[2] = view_mtx[2];
        target_camera_view_matrix[3] = view_mtx[3];

        target_camera_projection_matrix[0] = proj_mtx[0];
        target_camera_projection_matrix[1] = proj_mtx[1];
        target_camera_projection_matrix[2] = proj_mtx[2];
        target_camera_projection_matrix[3] = proj_mtx[3];
    }

    float4 color;

    switch (camera_idx)
    {
        case 0: color = dataset_target_image_0[pixel]; break;
        case 1: color = dataset_target_image_1[pixel]; break;
        case 2: color = dataset_target_image_2[pixel]; break;
        case 3: color = dataset_target_image_3[pixel]; break;
        case 4: color = dataset_target_image_4[pixel]; break;
        case 5: color = dataset_target_image_5[pixel]; break;
        default: color = float4(1, 0, 1, 1); break; // Magenta error
    }
    target_image[pixel] = color;
}
