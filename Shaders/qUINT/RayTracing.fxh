/*=============================================================================

    Copyright (c) Pascal Gilcher. All rights reserved.

 * Unauthorized copying of this file, via any medium is strictly prohibited
 * Proprietary and confidential
 
=============================================================================*/

#pragma once

/*===========================================================================*/

namespace RayTracing
{

struct RayDesc 
{
    float3 origin;
    float3 pos;
    float3 dir;
    float2 uv;
    float length;
    float width; //faux cone tracing
};

float compute_intersection(inout RayDesc ray, float maxT, float incT, float epsilon, bool snap_intersection)
{
    float intersected = 0;
    float4 prev = float4(ray.origin, 0);
    float2 uv = ray.uv;

    [loop]
    while(ray.length < maxT)
    {
        float lambda = ray.length / maxT;  
        lambda = lambda * (lambda * (1.25 * lambda - 0.375) + 0.125);

        ray.pos = ray.origin + ray.dir * lambda * maxT;
        ray.uv = Projection::proj_to_uv(ray.pos);

        if(!all(saturate(-ray.uv * ray.uv + ray.uv))) break; 

        ray.width = clamp(log2(length((ray.uv - uv) * BUFFER_SCREEN_SIZE)) - 4.0, 0, MIP_AMT);

        float3 pos = Projection::uv_to_proj(ray.uv, sZTexCS, ray.width);
        float delta = pos.z - ray.pos.z;

        float z_tolerance = epsilon * maxT;
        z_tolerance *= lerp(0.2, 10.0, lambda); 
        z_tolerance *= abs(ray.dir.z); 

        [branch]
        if(abs(delta * 2.0 + z_tolerance) < z_tolerance)
        {
            intersected = saturate(1 - lambda);
            ray.uv = prev.w < 0 ? ray.uv : Projection::proj_to_uv(lerp(prev.xyz, ray.pos, prev.w / (prev.w + abs(delta))));
            if(snap_intersection) 
                ray.dir = normalize(lerp(pos - ray.origin, ray.dir, saturate(0.111 * maxT * lambda)));
            break;
        }
    
        ray.length += incT;
        prev = float4(ray.pos, delta);
    }

    return intersected;
}

} //namespace
