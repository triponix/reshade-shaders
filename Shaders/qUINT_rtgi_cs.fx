/*=============================================================================

	ReShade 4 effect file
    github.com/martymcmodding

	Support me:
   		paypal.me/mcflypg
   		patreon.com/mcflypg    

    * Unauthorized copying of this file, via any medium is strictly prohibited
 	* Proprietary and confidential

=============================================================================*/

#if __RENDERER__ >= 0xb000
#define COMPUTER_SHADERS_YAY
#endif

#if(!defined(COMPUTER_SHADERS_YAY))
 #warning "Game API does not support compute shaders with required feature set. Please use RTGI 0.21 instead until further notice."
#endif

#if __RESHADE__ < 40900
 #error "Update ReShade to at least 4.9.0"
#endif

/*=============================================================================
	Preprocessor settings
=============================================================================*/

#ifndef SMOOTHNORMALS
 #define SMOOTHNORMALS 			0   //[0 to 3]      0: off | 1: enables some filtering of the normals derived from depth buffer to hide 3d model blockyness
#endif

#ifndef INFINITE_BOUNCES
 #define INFINITE_BOUNCES       0   //[0 or 1]      If enabled, path tracer samples previous frame GI as well, causing a feedback loop to simulate secondary bounces, causing a more widespread GI.
#endif

#ifndef MATERIAL_TYPE
 #define MATERIAL_TYPE          0   //[0 to 1]      0: Lambert diffuse | 1: GGX BRDF
#endif

/*=============================================================================
	UI Uniforms
=============================================================================*/

uniform float RT_SAMPLE_RADIUS <
	ui_type = "drag";
	ui_min = 0.5; ui_max = 20.0;
    ui_step = 0.01;
    ui_label = "Ray Length";
	ui_tooltip = "Maximum ray length, directly affects\nthe spread radius of shadows / bounce lighting";
    ui_category = "Ray Tracing";
> = 4.0;

uniform float RT_SAMPLE_RADIUS_FAR <
	ui_type = "drag";
	ui_min = 0.0; ui_max = 1.0;
    ui_step = 0.01;
    ui_label = "Extended Ray Length Multiplier";
	ui_tooltip = "Increases ray length in the background to achieve ultra wide light bounces.";
    ui_category = "Ray Tracing";
> = 0.0;

uniform int RT_RAY_AMOUNT <
	ui_type = "slider";
	ui_min = 1; ui_max = 20;
    ui_label = "Amount of Rays";
    ui_tooltip = "Amount of rays launched per pixel in order to\nestimate the global illumination at this location.\nAmount of noise to filter is proportional to sqrt(rays).";
    ui_category = "Ray Tracing";
> = 3;

uniform int RT_RAY_STEPS <
	ui_type = "slider";
	ui_min = 1; ui_max = 40;
    ui_label = "Amount of Steps per Ray";
    ui_tooltip = "RTGI performs step-wise raymarching to check for ray hits.\nFewer steps may result in rays skipping over small details.";
    ui_category = "Ray Tracing";
> = 12;

uniform float RT_Z_THICKNESS <
	ui_type = "drag";
	ui_min = 0.0; ui_max = 4.0;
    ui_step = 0.01;
    ui_label = "Z Thickness";
	ui_tooltip = "The shader can't know how thick objects are, since it only\nsees the side the camera faces and has to assume a fixed value.\n\nUse this parameter to remove halos around thin objects.";
    ui_category = "Ray Tracing";
> = 0.5;

uniform bool RT_HIGHP_LIGHT_SPREAD <
    ui_label = "Enable precise light spreading";
    ui_tooltip = "Rays accept scene intersections within a small error margin.\nEnabling this will snap rays to the actual hit location.\nThis results in sharper but more realistic lighting.";
    ui_category = "Ray Tracing";
> = true;

uniform bool RT_BACKFACE_MIRROR <
    ui_label = "Enable simulation of backface lighting";
    ui_tooltip = "RTGI can only simulate light bouncing of the objects visible on the screen.\nTo estimate light coming from non-visible sides of otherwise visible objects,\nthis feature will just take the front-side color instead.";
    ui_category = "Ray Tracing";
> = false;

#if MATERIAL_TYPE == 1
uniform float RT_SPECULAR <
	ui_type = "drag";
	ui_min = 0.01; ui_max = 1.0;
    ui_step = 0.01;
    ui_label = "Specular";
    ui_tooltip = "Specular Material parameter for GGX Microfacet BRDF";
    ui_category = "Material";
> = 1.0;

uniform float RT_ROUGHNESS <
	ui_type = "drag";
	ui_min = 0.05; ui_max = 1.0;
    ui_step = 0.01;
    ui_label = "Roughness";
    ui_tooltip = "Roughness Material parameter for GGX Microfacet BRDF";
    ui_category = "Material";
> = 1.0;
#endif

uniform float RT_AO_AMOUNT <
	ui_type = "drag";
	ui_min = 0; ui_max = 10.0;
    ui_step = 0.01;
    ui_label = "Ambient Occlusion Intensity";
    ui_category = "Blending";
> = 4.0;

uniform float RT_IL_AMOUNT <
	ui_type = "drag";
	ui_min = 0; ui_max = 10.0;
    ui_step = 0.01;
    ui_label = "Bounce Lighting Intensity";
    ui_category = "Blending";
> = 4.0;

#if INFINITE_BOUNCES != 0
    uniform float RT_IL_BOUNCE_WEIGHT <
        ui_type = "drag";
        ui_min = 0; ui_max = 2.0;
        ui_step = 0.01;
        ui_label = "Next Bounce Weight";
        ui_category = "Blending";
    > = 0.0;
#endif

uniform int FADEOUT_MODE_UI < //rename because possible clash with older config
	ui_type = "slider";
    ui_min = 0; ui_max = 2;
    ui_label = "Fade Out Mode";
    ui_category = "Blending";
> = 2;

uniform float RT_FADE_DEPTH <
	ui_type = "drag";
    ui_label = "Fade Out Range";
	ui_min = 0.001; ui_max = 1.0;
	ui_tooltip = "Distance falloff, higher values increase RTGI draw distance.";
    ui_category = "Blending";
> = 0.3;

uniform int RT_DEBUG_VIEW <
	ui_type = "radio";
    ui_label = "Enable Debug View";
	ui_items = "None\0Lighting Channel\0Normal Channel\0";
	ui_tooltip = "Different debug outputs";
    ui_category = "Debug";
> = 0;
/*
uniform float4 tempF1 <
    ui_type = "drag";
    ui_min = -100.0;
    ui_max = 100.0;
> = float4(1,1,1,1);

uniform float4 tempF2 <
    ui_type = "drag";
    ui_min = -100.0;
    ui_max = 100.0;
> = float4(1,1,1,1);

uniform float4 tempF3 <
    ui_type = "drag";
    ui_min = -100.0;
    ui_max = 100.0;
> = float4(1,1,1,1);
*/
/*=============================================================================
	Textures, Samplers, Globals, Structs
=============================================================================*/

uniform uint  FRAMECOUNT  < source = "framecount"; >;
uniform float FRAMETIME   < source = "frametime";  >;

#define RTGI_DEBUG_SKIP_FILTER      0

#if MATERIAL_TYPE == 1
#define WORKGROUPSIZE_X             16
#define WORKGROUPSIZE_Y             16
#else 
#define WORKGROUPSIZE_X             16
#define WORKGROUPSIZE_Y             24
#endif

//log2 macro for uints up to 16 bit, inefficient in runtime but preprocessor doesn't care
#define T1(x,n) ((uint(x)>>(n))>0)
#define T2(x,n) (T1(x,n)+T1(x,n+1))
#define T4(x,n) (T2(x,n)+T2(x,n+2))
#define T8(x,n) (T4(x,n)+T4(x,n+4))
#define LOG2(x) (T8(x,0)+T8(x,8))

//integer divide, rounding up
#define CEIL_DIV(num, denom) ((((num) - 1) / (denom)) + 1)

//for 1920x1080, use 3 mip levels
//double the screen size, use one mip level more
//log2(1920/240) = 3
//log2(3840/240) = 4
#define MIP_AMT 	LOG2(BUFFER_WIDTH / 240)

texture ZTexCS          { Width = BUFFER_WIDTH;   Height = BUFFER_HEIGHT;   Format = R16F;      MipLevels = MIP_AMT; };
texture NTexCS          { Width = BUFFER_WIDTH;   Height = BUFFER_HEIGHT;   Format = RGB10A2;                        };
texture CTexCS          { Width = BUFFER_WIDTH;   Height = BUFFER_HEIGHT;   Format = RGB10A2;   MipLevels = MIP_AMT; };
texture GBufferTexRef	{ Width = BUFFER_WIDTH;   Height = BUFFER_HEIGHT;   Format = RGBA16F;                        };
texture GBufferTex_0    { Width = BUFFER_WIDTH;   Height = BUFFER_HEIGHT;   Format = RGBA16F;                        };
texture GBufferTex_1    { Width = BUFFER_WIDTH;   Height = BUFFER_HEIGHT;   Format = RGBA16F;                        };
texture GBufferTex_2    { Width = BUFFER_WIDTH;   Height = BUFFER_HEIGHT;   Format = RGBA16F;                        };
texture GITex_0	        { Width = BUFFER_WIDTH;   Height = BUFFER_HEIGHT;   Format = RGBA16F;                        };
texture GITex_1	        { Width = BUFFER_WIDTH;   Height = BUFFER_HEIGHT;   Format = RGBA16F;                        };
texture GITex_2	        { Width = BUFFER_WIDTH;   Height = BUFFER_HEIGHT;   Format = RGBA16F;                        };
texture GITex_Filter0	{ Width = BUFFER_WIDTH;   Height = BUFFER_HEIGHT;   Format = RGBA16F;                        };
texture GITex_Filter1	{ Width = BUFFER_WIDTH;   Height = BUFFER_HEIGHT;   Format = RGBA16F;                        };

sampler sZTexCS         { Texture = ZTexCS;         };
sampler sNTexCS         { Texture = NTexCS;         };
sampler sCTexCS         { Texture = CTexCS;         };
sampler sGBufferTexRef	{ Texture = GBufferTexRef;	};
sampler sGBufferTex_0	{ Texture = GBufferTex_0;	};
sampler sGBufferTex_1	{ Texture = GBufferTex_1;	};
sampler sGBufferTex_2	{ Texture = GBufferTex_2;	};
sampler sGITex_0       	{ Texture = GITex_0;        };
sampler sGITex_1       	{ Texture = GITex_1;        };
sampler sGITex_2       	{ Texture = GITex_2;        };
sampler sGITex_Filter0  { Texture = GITex_Filter0;  };
sampler sGITex_Filter1  { Texture = GITex_Filter1;  };

storage stGBufferTex_0	{ Texture = GBufferTex_0;	};
storage stGBufferTex_1	{ Texture = GBufferTex_1;	};
storage stGBufferTex_2	{ Texture = GBufferTex_2;	};
storage stGITex_0       { Texture = GITex_0;        };
storage stGITex_1       { Texture = GITex_1;        };
storage stGITex_2       { Texture = GITex_2;        };
storage stGITex_Filter0 { Texture = GITex_Filter0;  };
storage stGITex_Filter1 { Texture = GITex_Filter1;  };

texture JitterTex < source = "bluenoise.png"; > { Width = 32; 			  Height = 32; 				Format = RGBA8; };
sampler	sJitterTex          { Texture = JitterTex; AddressU = WRAP; AddressV = WRAP;};

struct CSIN 
{
    uint3 groupthreadid     : SV_GroupThreadID;         //XYZ idx of thread inside group
    uint3 groupid           : SV_GroupID;               //XYZ idx of group inside dispatch
    uint3 dispatchthreadid  : SV_DispatchThreadID;      //XYZ idx of thread inside dispatch
    uint threadid           : SV_GroupIndex;            //flattened idx of thread inside group
};

struct VSOUT
{
	float4 vpos : SV_Position;
    float2 uv   : TEXCOORD0;
};

#include "qUINT\Global.fxh"
#include "qUINT\Depth.fxh"
#include "qUINT\Projection.fxh"
#include "qUINT\Normal.fxh"
#include "qUINT\Random.fxh"
#include "qUINT\RaySorting.fxh"
#include "qUINT\RayTracing.fxh"
#include "qUINT\Denoise.fxh"

/*=============================================================================
	Functions
=============================================================================*/

float4 safe_tex2Dfetch(sampler src, int2 coord)
{
    uint2 size = tex2Dsize(src);
    coord = clamp(coord, 0, size - 1); //CLAMP mode
    return tex2Dfetch(src, coord);
}

float2 pixel_idx_to_uv(uint2 pos, float2 texture_size)
{
    float2 inv_texture_size = rcp(texture_size);
    return pos * inv_texture_size + 0.5 * inv_texture_size;
}

void unpack_hdr(inout float3 color)
{
    color  = saturate(color);
    color = color * rcp(1.01 - saturate(color)); 
    //color = sRGB2AP1(color);
}

void pack_hdr(inout float3 color)
{
    //color = AP12sRGB(color);
    color = 1.01 * color * rcp(color + 1.0);
    color  = saturate(color);
}

float3 ggx_vndf(float2 uniform_disc, float2 alpha, float3 v)
{
	//scale by alpha, 3.2
	float3 Vh = normalize(float3(alpha * v.xy, v.z));
	//point on projected area of hemisphere
	float2 p = uniform_disc;
	p.y = lerp(sqrt(1.0 - p.x*p.x), 
		       p.y,
		       Vh.z * 0.5 + 0.5);

	float3 Nh =  float3(p.xy, sqrt(saturate(1.0 - dot(p, p)))); //150920 fixed sqrt() of z

	//reproject onto hemisphere
	Nh = mul(Nh, Normal::base_from_vector(Vh));

	//revert scaling
	Nh = normalize(float3(alpha * Nh.xy, saturate(Nh.z)));

	return Nh;
}

float3 schlick_fresnel(float vdoth, float3 f0)
{
	vdoth = saturate(vdoth);
	return lerp(pow(vdoth, 5), 1, f0);
}

float ggx_g2_g1(float3 l, float3 v, float2 alpha)
{
	//smith masking-shadowing g2/g1, v and l in tangent space
	l.xy *= alpha;
	v.xy *= alpha;
	float nl = length(l);
	float nv = length(v);

    float ln = l.z * nv;
    float lv = l.z * v.z;
    float vn = v.z * nl;
    //in tangent space, v.z = ndotv and l.z = ndotl
    return (ln + lv) / (vn + ln + 1e-7);
}

void store_data_revolver(storage st0, storage st1, storage st2, uint2 coord, float4 res)
{
    uint writeslot = FRAMECOUNT % 3u;
         if(writeslot == 0) tex2Dstore(st0, coord, res);
    else if(writeslot == 1) tex2Dstore(st1, coord, res);
    else if(writeslot == 2) tex2Dstore(st2, coord, res);
}

float4 fetch_data_revolver(sampler s0, sampler s1, sampler s2, uint2 coord)
{
    uint readslot = FRAMECOUNT % 3u;
    float4 o = 0;

         if(readslot == 0) o = tex2Dfetch(s0, coord, 0);
    else if(readslot == 1) o = tex2Dfetch(s1, coord, 0);
    else if(readslot == 2) o = tex2Dfetch(s2, coord, 0);

    return o;
}

float3 smooth_normals(in float2 iuv)
{ 
    const float max_n_n = 0.63;
    const float max_v_s = 0.65;
    const float max_c_p = 0.5;
    const float searchsize = 0.0125;
    const int dirs = 5;

    #define fetch_gbuf(coord) float4(tex2Dlod(sNTexCS, coord, 0).xyz * 2.0 - 1.0, tex2Dlod(sZTexCS, coord, 0).x)

    float4 gbuf_center = fetch_gbuf(iuv);//tex2D(sRTGITempTex1, i.uv);

    float3 n_center = gbuf_center.xyz;
    float3 p_center = Projection::uv_to_proj(iuv, gbuf_center.w);
    float radius = searchsize + searchsize * rcp(p_center.z) * 2.0;
    float worldradius = radius * p_center.z;

    int steps = clamp(ceil(radius * 300.0) + 1, 1, 7);
    float3 n_sum = 0.001 * n_center;

    for(float j = 0; j < dirs; j++)
    {
        float2 dir; sincos(radians(360.0 * j / dirs + 0.666), dir.y, dir.x);

        float3 n_candidate = n_center;
        float3 p_prev = p_center;

        for(float stp = 1.0; stp <= steps; stp++)
        {
            float fi = stp / steps;   
            fi *= fi * rsqrt(fi);

            float offs = fi * radius;
            offs += length(BUFFER_PIXEL_SIZE);

            float2 uv = iuv + dir * offs * BUFFER_ASPECT_RATIO;            
            if(!all(saturate(uv - uv*uv))) break;

            float4 gbuf = fetch_gbuf(uv);//tex2Dlod(sRTGITempTex1, float4(uv, 0, 0));
            float3 n = gbuf.xyz;
            float3 p = Projection::uv_to_proj(uv, gbuf.w);

            float3 v_increment  = normalize(p - p_prev);

            float ndotn         = dot(n, n_center); 
            float vdotn         = dot(v_increment, n_center); 
            float v2dotn        = dot(normalize(p - p_center), n_center); 
          
            ndotn *= max(0, 1.0 + fi *0.5 * (1.0 - abs(v2dotn)));

            if(abs(vdotn)  > max_v_s || abs(v2dotn) > max_c_p) break;       

            if(ndotn > max_n_n)
            {
                float d = distance(p, p_center) / worldradius;
                float w = saturate(4.0 - 2.0 * d) * smoothstep(max_n_n, lerp(max_n_n, 1.0, 2), ndotn); //special recipe
                w = stp < 1.5 && d < 2.0 ? 1 : w;  //special recipe       
                n_candidate = lerp(n_candidate, n, w);
                n_candidate = normalize(n_candidate);
            }

            p_prev = p;
            n_sum += n_candidate;
        }
    }

    n_sum = normalize(n_sum);
    return n_sum;
}

float fade_distance(in VSOUT i)
{
    float distance = saturate(length(Projection::uv_to_proj(i.uv)) / RESHADE_DEPTH_LINEARIZATION_FAR_PLANE);
    float fade;
    switch(FADEOUT_MODE_UI)
    {
        case 0:
        fade = saturate((RT_FADE_DEPTH - distance) / RT_FADE_DEPTH);
        break;
        case 1:
        fade = saturate((RT_FADE_DEPTH - distance) / RT_FADE_DEPTH);
        fade *= fade; fade *= fade;
        break;
        case 2:
        float fadefact = rcp(RT_FADE_DEPTH * 0.32);
        float cutoff = exp(-fadefact);
        fade = saturate((exp(-distance * fadefact) - cutoff)/(1 - cutoff));
        break;
    }    

    return fade;    
}

/*=============================================================================
	Shader entry points
=============================================================================*/

VSOUT VS_Main(in uint id : SV_VertexID)
{
    VSOUT o;
    VS_FullscreenTriangle(id, o.vpos, o.uv); //use original fullscreen triangle VS
    return o;
}

void PS_MakeInputs(in VSOUT i, out MRT3 o)
{ 
    o.t0 = Depth::get_linear_depth(i.uv);
    o.t1 = tex2D(ColorInput, i.uv);
    o.t1 *= saturate(999.0 - o.t0.x * 1000.0); //mask sky
    o.t0 = Projection::depth_to_z(o.t0.x);
    o.t2.xyz = Normal::normal_from_depth(i.uv) * 0.5 + 0.5;
    o.t2.w = 1;
}

void CS_RTMain(in CSIN i)
{  
    uint2 groupsize = uint2(WORKGROUPSIZE_X, WORKGROUPSIZE_Y);
    uint2 dispatchsize = uint2(CEIL_DIV(BUFFER_WIDTH, WORKGROUPSIZE_X), CEIL_DIV(BUFFER_HEIGHT, WORKGROUPSIZE_Y));

    float3 jitter        = tex2Dfetch(sJitterTex,  i.dispatchthreadid.xy        & 0x1F).xyz;
    jitter = frac(jitter + tex2Dfetch(sJitterTex, (i.dispatchthreadid.xy >> 5u) & 0x1F).xyz);

    float2 uv = pixel_idx_to_uv(i.dispatchthreadid.xy, BUFFER_SCREEN_SIZE);
    float3 n = normalize(tex2Dfetch(sNTexCS, i.dispatchthreadid.xy).xyz * 2.0 - 1.0);

#if SMOOTHNORMALS != 0
    n = smooth_normals(uv);
#endif

    float3 p = Projection::uv_to_proj(uv); //can't hurt to have best data..
    float d = Projection::z_to_depth(p.z); p *= 0.999; p += n * d;  

    float ray_maxT = lerp(RT_SAMPLE_RADIUS * RT_SAMPLE_RADIUS, RT_SAMPLE_RADIUS * RT_SAMPLE_RADIUS * 100.0, saturate(d * RT_SAMPLE_RADIUS_FAR));
    ray_maxT = min(ray_maxT, RESHADE_DEPTH_LINEARIZATION_FAR_PLANE);

    SampleSet sampleset = ray_sorting(FRAMECOUNT, jitter.x);

#if MATERIAL_TYPE == 1
    float3 specular_color = tex2Dlod(ColorInput, uv, 0).rgb; 
    specular_color = lerp(dot(specular_color, 0.333), specular_color, 0.666) * 2.0;
    specular_color *= RT_SPECULAR;
    float3 eyedir = normalize(-p);
    float3x3 tangent_base = Normal::base_from_vector(n);
    float3 tangent_eyedir = mul(eyedir, transpose(tangent_base));
#endif 
    
    float4 rtgi = 0;
    for(uint r = 0; r < RT_RAY_AMOUNT; r++)
    {  
        RayTracing::RayDesc ray;  
#if MATERIAL_TYPE == 0
        ray.dir.z = (r + sampleset.index) / RT_RAY_AMOUNT * 2.0 - 1.0; 
        ray.dir.xy = sampleset.dir_xy * sqrt(1.0 - ray.dir.z * ray.dir.z); //build sphere
        ray.dir = normalize(ray.dir + n);
#elif MATERIAL_TYPE == 1
        float alpha = RT_ROUGHNESS * RT_ROUGHNESS; //isotropic  
        //"random" point on disc - do I have to do sqrt() ?
        float2 uniform_disc = sqrt((r + sampleset.index) / RT_RAY_AMOUNT) * sampleset.dir_xy;
        float3 v = tangent_eyedir;
        float3 h = ggx_vndf(uniform_disc, alpha.xx, v);
        float3 l = reflect(-v, h);

        //single scatter lobe
        float3 brdf = ggx_g2_g1(l, v , alpha.xx); //if l.z > 0 is checked later
        brdf = l.z < 1e-7 ? 0 : brdf; //test?
        float vdoth = dot(eyedir, h);
        brdf *= schlick_fresnel(vdoth, specular_color);

        ray.dir = mul(l, tangent_base); //l from tangent to projection
#endif
        //advance to next ray dir
        sampleset.dir_xy = mul(sampleset.dir_xy, sampleset.nextdir);        

        float ray_incT = (ray_maxT / RT_RAY_STEPS) * rsqrt(saturate(1.0 - ray.dir.z * ray.dir.z) + 0.001);        
        ray.length = ray_incT * frac(jitter.y + r * 1.6180339887);
        ray.origin = p;
        ray.uv = uv;

        float intersected = RayTracing::compute_intersection(ray, ray_maxT, ray_incT, RT_Z_THICKNESS * RT_Z_THICKNESS, RT_HIGHP_LIGHT_SPREAD);
      
        [branch]
        if(intersected > 0.05)
        {
            ray.width = max(ray.width, 1.0);
#if MATERIAL_TYPE == 1
            //revert to fullres mips for sharper reflection at low roughness settings
            ray.width *= smoothstep(0.05, 0.2, RT_ROUGHNESS);
#endif
            float3 albedo           = tex2Dlod(sCTexCS, ray.uv, ray.width).rgb; unpack_hdr(albedo); 
            float3 intersect_normal = tex2Dlod(sNTexCS, ray.uv, 0).xyz * 2.0 - 1.0;
            float backface_check = saturate(dot(-intersect_normal, ray.dir) * 64.0);

            backface_check = RT_BACKFACE_MIRROR ? lerp(backface_check, 1.0, 0.1) : backface_check;            
            albedo *= backface_check;

#if INFINITE_BOUNCES != 0
            float4 nextbounce = tex2Dlod(sGITex_Filter1, ray.uv, ray.width); unpack_hdr(nextbounce.rgb);           
            float3 compounded = albedo * nextbounce.rgb + 0.0001;
            //if(tempF1.x > 0) compounded = normalize(sqrt(compounded)+compounded) * length(compounded);            
            albedo += compounded * RT_IL_BOUNCE_WEIGHT * 10.0;
#endif         

#if MATERIAL_TYPE == 1  
            albedo *= brdf;
            albedo *= 10.0;
#endif
            //for lambert: * cos theta / pdf == 1 because cosine weighted
            rtgi += float4(albedo * intersected, intersected);
        }     
    }

    rtgi /= RT_RAY_AMOUNT;        
    pack_hdr(rtgi.rgb);
    store_data_revolver(stGITex_0,      stGITex_1,      stGITex_2,      i.dispatchthreadid.xy, rtgi);
    store_data_revolver(stGBufferTex_0, stGBufferTex_1, stGBufferTex_2, i.dispatchthreadid.xy, float4(n.xyz, p.z));
}

void PS_Combine(in VSOUT i, out MRT2 o)
{
    float4 gi[3], gbuf[3];
    gi[0] = tex2Dfetch(sGITex_0, i.vpos.xy);
    gi[1] = tex2Dfetch(sGITex_1, i.vpos.xy);
    gi[2] = tex2Dfetch(sGITex_2, i.vpos.xy);
    gbuf[0] = tex2Dfetch(sGBufferTex_0, i.vpos.xy);
    gbuf[1] = tex2Dfetch(sGBufferTex_1, i.vpos.xy);
    gbuf[2] = tex2Dfetch(sGBufferTex_2, i.vpos.xy);

    uint cframe = FRAMECOUNT % 3u;
    float4 gbuf_ref = gbuf[cframe];
    float4 gi_ref = gi[cframe];

    float timefact = 16.7 / max(FRAMETIME, 1.0); //~1 for 60 fps, expected range
    
    float4 d1 = abs(gbuf_ref - gbuf[0]), 
           d2 = abs(gbuf_ref - gbuf[1]),
           d3 = abs(gbuf_ref - gbuf[2]);

    d1 *= timefact, d2 *= timefact, d3 *= timefact;

    float3 d = float3(dot(d1, float4(d1.xyz * 2.0, 1.0)), 
                      dot(d2, float4(d2.xyz * 2.0, 1.0)),
                      dot(d3, float4(d3.xyz * 2.0, 1.0))); //normal squared, depth linear

    float3 w = exp(-d);
    w /= dot(w, 1);
    o.t0 = gi[0] * w.x + gi[1] * w.y + gi[2] * w.z;
    o.t1 = gbuf_ref;
}

void PS_Filter0(in VSOUT i, out float4 o : SV_Target0)
{
    o = Denoise::filter(i, sGITex_Filter0, 0, -RTGI_DEBUG_SKIP_FILTER);
}
void PS_Filter1(in VSOUT i, out float4 o : SV_Target0)
{
    o = Denoise::filter(i, sGITex_Filter1, 1, -RTGI_DEBUG_SKIP_FILTER);
}
void PS_Filter2(in VSOUT i, out float4 o : SV_Target0)
{
    o = Denoise::filter(i, sGITex_Filter0, 2, -RTGI_DEBUG_SKIP_FILTER);
}
void PS_Filter3AndCombine(in VSOUT i, out float4 o : SV_Target0)
{
    float4 gi = Denoise::filter(i, sGITex_Filter1, 3, -RTGI_DEBUG_SKIP_FILTER);
    float3 color = tex2D(ColorInput, i.uv).rgb;

    unpack_hdr(color);
    unpack_hdr(gi.rgb);

    color = RT_DEBUG_VIEW == 1 ? 1 : color; 

    float similarity = distance(normalize(color + 0.00001), normalize(gi.rgb + 0.00001));
	similarity = saturate(similarity * 3.0);
	gi.rgb = lerp(dot(gi.rgb, 0.3333), gi.rgb, saturate(similarity * 0.5 + 0.5)); 

    gi *= fade_distance(i);
   
    color += lerp(color, dot(color, 0.333), saturate(1 - dot(color, 3.0))) * gi.rgb * RT_IL_AMOUNT * RT_IL_AMOUNT; //apply GI
    color = color / (1.0 + gi.w * RT_AO_AMOUNT);    

    pack_hdr(color.rgb);

    color = RT_DEBUG_VIEW == 2 ? tex2D(sGBufferTexRef, i.uv).xyz * float3(0.5, 0.5, -0.5) + 0.5 : color;
    o = float4(color, 1);
}

/*=============================================================================
	Techniques
=============================================================================*/

technique RTGlobalIlluminationCS
< ui_tooltip = "              >> qUINT::RTGI 0.23 CS <<\n\n"
               "         EARLY ACCESS -- PATREON ONLY\n"
               "Official versions only via patreon.com/mcflypg\n"
               "\nRTGI is written by Marty McFly / Pascal Gilcher\n"
               "Early access, featureset might be subject to change"; >
{
#if(defined(COMPUTER_SHADERS_YAY))
    pass
	{
		VertexShader = VS_Main;
        PixelShader  = PS_MakeInputs;
        RenderTarget0 = ZTexCS;
        RenderTarget1 = CTexCS;
        RenderTarget2 = NTexCS;
    }
    pass 
    { 
        ComputeShader = CS_RTMain<WORKGROUPSIZE_X, WORKGROUPSIZE_Y>;
        DispatchSizeX = CEIL_DIV(BUFFER_WIDTH, WORKGROUPSIZE_X); 
        DispatchSizeY = CEIL_DIV(BUFFER_HEIGHT, WORKGROUPSIZE_Y);
    }
    pass
	{
		VertexShader = VS_Main;
        PixelShader  = PS_Combine;
        RenderTarget0 = GITex_Filter0;
        RenderTarget1 = GBufferTexRef;
	}
    pass
    {
        VertexShader = VS_Main;
        PixelShader  = PS_Filter0;
        RenderTarget0 = GITex_Filter1;
    }
    pass
    {
        VertexShader = VS_Main;
        PixelShader  = PS_Filter1;
        RenderTarget0 = GITex_Filter0;
    } 
    pass
    {
        VertexShader = VS_Main;
        PixelShader  = PS_Filter2;
        RenderTarget0 = GITex_Filter1;
    } 
    pass
    {
        VertexShader = VS_Main;
        PixelShader  = PS_Filter3AndCombine;
    }
#endif
}
