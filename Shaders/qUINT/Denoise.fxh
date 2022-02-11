/*=============================================================================

    Copyright (c) Pascal Gilcher. All rights reserved.

 * Unauthorized copying of this file, via any medium is strictly prohibited
 * Proprietary and confidential

=============================================================================*/

#pragma once

/*===========================================================================*/

namespace Denoise
{

struct FilterSample
{
    float4 gbuffer;
    float4 val;
};

FilterSample fetch_sample(in float2 uv, sampler gi)
{
    FilterSample o;
    o.gbuffer = tex2Dlod(sGBufferTexRef, uv, 0);
    o.val     = tex2Dlod(gi, uv, 0);
    return o;
}

float4 filter(in VSOUT i, in sampler gi, int iteration, int mode)
{
    FilterSample center = fetch_sample(i.uv, gi);

    if(mode < 0) //skip
        return center.val;

    float4 value_sum = center.val * 0.01; 
    float weight_sum = 0.01; 

    float4 kernel = float4(1.5,3.5,7,15);
    float4 sigma_z = float4(0.7,0.7,0.7,0.7);
    float4 sigma_n = float4(0.75,1.5,1.5,5);
    float4 sigma_v = float4(0.035,0.6,1.4,5);

    if(mode == 1)
    {
        sigma_z *= 2.0;
        sigma_n *= 2.0;
        sigma_v *= 8.0;      
    }

    float expectederror = sqrt(RT_RAY_AMOUNT);

    for(float x = -1; x <= 1; x++)
    for(float y = -1; y <= 1; y++)
    {        
        float2 uv = i.uv + float2(x, y) * kernel[iteration] * BUFFER_PIXEL_SIZE;
        FilterSample tap = fetch_sample(uv, gi);

        float wz = sigma_z[iteration] * 16.0 *  (1.0 - tap.gbuffer.w / center.gbuffer.w);
        wz = saturate(0.5 - lerp(wz, abs(wz), 0.75));

        float wn = saturate(dot(tap.gbuffer.xyz, center.gbuffer.xyz) * (sigma_n[iteration] + 1) - sigma_n[iteration]);
        float wi = dot(abs(tap.val - center.val), float4(0.3, 0.59, 0.11, 3.0));
 
        wi = exp(-wi * wi * 2.0 * sigma_v[iteration] * expectederror);

        wn = lerp(wn, 1, saturate(wz * 1.42 - 0.42));
        float w = saturate(wz * wn * wi);

        value_sum += tap.val * w;
        weight_sum += w;
    }

    float4 result = value_sum / weight_sum;
    return result;
}

} //Namespace