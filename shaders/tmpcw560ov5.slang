
[shader("compute")]
[numthreads(1,1,1)]
void testMul(
    RWStructuredBuffer<float> result,
    uniform float2x3 J,
    uniform float3x3 Sigma)
{
    float2x2 cov = mul(J, mul(Sigma, transpose(J)));
    result[0] = cov[0][0];
    result[1] = cov[0][1];
    result[2] = cov[1][0];
    result[3] = cov[1][1];
    
    // Also output what J[0] looks like
    result[4] = J[0][0];
    result[5] = J[0][1];
    result[6] = J[0][2];
    result[7] = J[1][0];
    result[8] = J[1][1];
    result[9] = J[1][2];
}
