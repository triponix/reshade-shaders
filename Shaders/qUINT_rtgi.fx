/*=============================================================================

   Copyright (c) Pascal Gilcher. All rights reserved.

	ReShade effect file
    github.com/martymcmodding

	Support me:
   		patreon.com/mcflypg

    Path Traced Global Illumination 

    * Unauthorized copying of this file, via any medium is strictly prohibited
 	* Proprietary and confidential
    * See accompanying license document for terms and conditions

=============================================================================*/

#if __RESHADE__ < 40802
 #error "Update ReShade to at least 4.8.2."
#endif

/*=============================================================================
	Preprocessor settings
=============================================================================*/

#ifndef INFINITE_BOUNCES
 #define INFINITE_BOUNCES       0   //[0 or 1]      If enabled, path tracer samples previous frame GI as well, causing a feedback loop to simulate secondary bounces, causing a more widespread GI.
#endif

#ifndef SKYCOLOR_MODE
 #define SKYCOLOR_MODE          0   //[0 to 3]      0: skycolor feature disabled | 1: manual skycolor | 2: dynamic skycolor | 3: dynamic skycolor with manual tint overlay
#endif

#ifndef IMAGEBASEDLIGHTING
 #define IMAGEBASEDLIGHTING     0   //[0 to 3]      0: no ibl infill | 1: use ibl infill
#endif

#ifndef MATERIAL_TYPE
 #define MATERIAL_TYPE          0   //[0 to 1]      0: Lambert diffuse | 1: GGX BRDF
#endif

#ifndef HALFRES_INPUT
 #define HALFRES_INPUT 			0   //[0 to 1]      0: use full resolution color and depth input | 1 : use half resolution color and depth input (faster)
#endif

#ifndef SMOOTHNORMALS
 #define SMOOTHNORMALS 			0   //[0 to 3]      0: off | 1: enables some filtering of the normals derived from depth buffer to hide 3d model blockyness
#endif

#ifndef FADEOUT_MODE
 #define FADEOUT_MODE 			0   //[0 to 3]      0: smoothstep original* using distance vs depth | 1: linear | 2: biquadratic | 3: exponential
#endif

/*=============================================================================
	UI Uniforms
=============================================================================*/

uniform int UIHELP <
	ui_type = "radio";
	ui_label = " ";	
	ui_text ="This shader adds ray traced / ray marched global illumination to games\nby traversing the height field described by the depth map of the game.\n\nHover over the settings below to display more information.\n\n          >>>>>>>>>> IMPORTANT <<<<<<<<<      \n\nIf the shader appears to do nothing when enabled, make sure ReShade's\ndepth access is properly set up - no output without proper input.\n\n          >>>>>>>>>> IMPORTANT <<<<<<<<<      ";
	ui_category = ">>>> OVERVIEW / HELP (click me) <<<<";
	ui_category_closed = true;
>;

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

uniform bool RT_ALTERNATE_INTERSECT_TEST <
    ui_label = "Alternate intersection test";
    ui_tooltip = "Enables an alternate way to accept or reject ray hits.\nWill remove halos around thin objects but generate\nmore illumination and shadows elsewhere.\nMore accurate lighting but less pronounced >>SSAO<< outline look.";
    ui_category = "Ray Tracing";
> = true;

uniform bool RT_USE_FALLBACK <
    ui_label = "Enable ray miss fallback (experimental)";
    ui_tooltip = "RTGI normally only accepts actual ray intersection locations. To not waste already completed computing effort,\nenabling this will cause invalid samples to be treated like a regular SSAO sample and possible GI influence will be recovered.\n!!EXPERIMENTAL!!";
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

#if SKYCOLOR_MODE != 0

#if SKYCOLOR_MODE == 1
uniform float3 SKY_COLOR <
	ui_type = "color";
	ui_label = "Sky Color";
    ui_category = "Blending";
> = float3(1.0, 1.0, 1.0);
#endif

#if SKYCOLOR_MODE == 3
uniform float3 SKY_COLOR_TINT <
	ui_type = "color";
	ui_label = "Sky Color Tint";
    ui_category = "Blending";
> = float3(1.0, 1.0, 1.0);
#endif

#if SKYCOLOR_MODE == 2 || SKYCOLOR_MODE == 3
uniform float SKY_COLOR_SAT <
	ui_type = "drag";
	ui_min = 0; ui_max = 5.0;
    ui_step = 0.01;
    ui_label = "Auto Sky Color Saturation";
    ui_category = "Blending";
> = 1.0;
#endif

uniform float SKY_COLOR_AMBIENT_MIX <
	ui_type = "drag";
	ui_min = 0; ui_max = 1.0;
    ui_step = 0.01;
    ui_label = "Sky Color Ambient Mix";
    ui_tooltip = "How much of the occluded ambient color is considered skycolor\n\nIf 0, Ambient Occlusion removes white ambient color,\nif 1, Ambient Occlusion only removes skycolor";
    ui_category = "Blending";
> = 0.2;

uniform float SKY_COLOR_AMT <
	ui_type = "drag";
	ui_min = 0; ui_max = 10.0;
    ui_step = 0.01;
    ui_label = "Sky Color Intensity";
    ui_category = "Blending";
> = 4.0;
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

#if IMAGEBASEDLIGHTING != 0
uniform float RT_IBL_AMOUT <
	ui_type = "drag";
	ui_min = 0; ui_max = 1.0;
    ui_step = 0.01;
    ui_label = "Image Based Lighting Intensity";
    ui_category = "Blending";
> = 0.0;
#endif
#if INFINITE_BOUNCES != 0
    uniform float RT_IL_BOUNCE_WEIGHT <
        ui_type = "drag";
        ui_min = 0; ui_max = 2.0;
        ui_step = 0.01;
        ui_label = "Next Bounce Weight";
        ui_category = "Blending";
    > = 0.0;
#endif

uniform float2 RT_FADE_DEPTH <
	ui_type = "drag";
    ui_label = "Fade Out Start / End";
	ui_min = 0.00; ui_max = 1.00;
	ui_tooltip = "Distance where GI starts to fade out | is completely faded out.";
    ui_category = "Blending";
> = float2(0.0, 0.5);

uniform int RT_DEBUG_VIEW <
	ui_type = "radio";
    ui_label = "Enable Debug View";
	ui_items = "None\0Lighting Channel\0Normal Channel\0";
	ui_tooltip = "Different debug outputs";
    ui_category = "Debug";
> = 0;

uniform bool RT_DO_RENDER <
    ui_label = "Render a still frame (for screenshots)";
    ui_category = "Experimental";
    ui_tooltip = "This will progressively render a still frame with extreme quality.\nTo start rendering, check the box and wait until the result is sufficiently noise-free.\nYou can still adjust blending and toggle debug mode, but do not touch anything else.\nTo resume the game, uncheck the box.\n\nRequires a scene with no moving objects to work properly.";
> = false;

uniform int UIHELP2 <
	ui_type = "radio";
	ui_label = " ";	
	ui_text ="Description for preprocessor definitions:\n\nINFINITE_BOUNCES\n0: off\n1: allows the light to reflect more than once.\n\nSKYCOLOR_MODE\n0: off\n1: static color\n2: dynamic detection (wip)\n3: dynamic detection + manual tint\n\nIMAGEBASELIGHTING:\n0: off\n1: analyzes the image for main lighting directions and recovers rays that did not return data.\n\nFADEOUT_MODE\n0-3: multiple different fadeout modes\n\nMATERIAL_TYPE\n0: Lambertian surface (matte)\n1: GGX Material, allows to model matte, glossy, specular surfaces based off roughness and specularity parameters\n\nHALFRES_INPUT\n0: RTGI uses full resolution color + depth\n1: RTGI uses half resolution color + depth. Usually faster, but can lead to flickering.\n\nSMOOTHNORMALS\n0: off\n1: enables normal map filtering, reduces blockyness on low poly surfaces.";
	ui_category = ">>>> PREPROCESSOR DEFINITION GUIDE (click me) <<<<";
	ui_category_closed = false;
>;
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
	Textures, Samplers, Globals
=============================================================================*/

#define RESHADE_QUINT_COMMON_VERSION_REQUIRE 202
#define RESHADE_QUINT_EFFECT_DEPTH_REQUIRE
#include "qUINT_common.fxh"

//debug flags, toy around at your own risk
#define SKIP_FILTER 0
#define MONTECARLO_MAX_STACK_SIZE 1500

//only works for positive numbers up to 8 bit but I don't expect buffer_width to exceed 61k pixels
//forcing uints so it works for NvCamera as well
#define CONST_LOG2(v)   (((uint(v) >> 1u) != 0u) + ((uint(v) >> 2u) != 0u) + ((uint(v) >> 3u) != 0u) + ((uint(v) >> 4u) != 0u) + ((uint(v) >> 5u) != 0u) + ((uint(v) >> 6u) != 0u) + ((uint(v) >> 7u) != 0u))

//for 1920x1080, use 3 mip levels
//double the screen size, use one mip level more
//log2(1920/240) = 3
//log2(3840/240) = 4
#define MIP_AMT 	CONST_LOG2(BUFFER_WIDTH / 240)

#if HALFRES_INPUT != 0
#define MIP_BIAS_IL	1
texture ZTex               < pooled = true; >  { Width = BUFFER_WIDTH>>1;   Height = BUFFER_HEIGHT>>1;   Format = R16F;      MipLevels = MIP_AMT;};
texture ColorTex           < pooled = true; >  { Width = BUFFER_WIDTH>>1;   Height = BUFFER_HEIGHT>>1;   Format = RGB10A2;   MipLevels = MIP_AMT + MIP_BIAS_IL;  };
#else 
#define MIP_BIAS_IL	2
texture ZTex               < pooled = true; >  { Width = BUFFER_WIDTH;   Height = BUFFER_HEIGHT;   Format = R16F;      MipLevels = MIP_AMT;};
texture ColorTex           < pooled = true; >  { Width = BUFFER_WIDTH;   Height = BUFFER_HEIGHT;   Format = RGB10A2;   MipLevels = MIP_AMT + MIP_BIAS_IL;  };
#endif

texture GBufferTex      					    { Width = BUFFER_WIDTH;   Height = BUFFER_HEIGHT;   Format = RGBA16F; };
texture GBufferTex1      					    { Width = BUFFER_WIDTH;   Height = BUFFER_HEIGHT;   Format = RGBA16F; };
texture GBufferTex2      					    { Width = BUFFER_WIDTH;   Height = BUFFER_HEIGHT;   Format = RGBA16F; };
texture GITex0	            					{ Width = BUFFER_WIDTH;   Height = BUFFER_HEIGHT;   Format = RGBA16F; };
texture GITex1	            					{ Width = BUFFER_WIDTH;   Height = BUFFER_HEIGHT;   Format = RGBA16F; };
texture GITex2	            					{ Width = BUFFER_WIDTH;   Height = BUFFER_HEIGHT;   Format = RGBA16F; };
texture RTGITempTex0 /*infinite bounces*/	    { Width = BUFFER_WIDTH;   Height = BUFFER_HEIGHT;   Format = RGBA16F; };
texture RTGITempTex1 /*smooth normals*/         { Width = BUFFER_WIDTH;   Height = BUFFER_HEIGHT;   Format = RGBA16F; };
texture JitterTex < source = "bluenoise.png"; > { Width = 32; 			  Height = 32; 				Format = RGBA8; };

#if IMAGEBASEDLIGHTING != 0 || SKYCOLOR_MODE >= 2
texture ProbeTex      					        { Width = 64;   Height = 64;  Format = RGBA16F;};
sampler sProbeTex	    					    { Texture = ProbeTex;	};
texture ProbeTexPrev      					    { Width = 64;   Height = 64;  Format = RGBA16F;};
sampler sProbeTexPrev	    					{ Texture = ProbeTexPrev;	};
#endif

sampler sZTex	            					{ Texture = ZTex;	    };
sampler sColorTex	        					{ Texture = ColorTex;	};
sampler sGBufferTex								{ Texture = GBufferTex;	};
sampler sGBufferTex1							{ Texture = GBufferTex1;	};
sampler sGBufferTex2							{ Texture = GBufferTex2;	};
sampler sGITex0       							{ Texture = GITex0;    };
sampler sGITex1       							{ Texture = GITex1;    };
sampler sGITex2       							{ Texture = GITex2;    };
sampler sRTGITempTex0       				    { Texture = RTGITempTex0;    };
sampler sRTGITempTex1       				    { Texture = RTGITempTex1;    };
sampler	sJitterTex          					{ Texture = JitterTex; AddressU = WRAP; AddressV = WRAP;};

/*=============================================================================
	Vertex Shader
=============================================================================*/

struct VSOUT
{
	float4                  vpos        : SV_Position;
    float2                  uv          : TEXCOORD0;
};

VSOUT VS_RT(in uint id : SV_VertexID)
{
    VSOUT o;
    PostProcessVS(id, o.vpos, o.uv); //use original fullscreen triangle VS
    return o;
}

/*=============================================================================
	Functions
=============================================================================*/

struct RTInputs
{
	//per pixel
    float3 pos;
    float3 normal;
    float3 eyedir;
    float3x3 tangent_base;
    float3 jitter;

    //runtime pixel independent
    int nrays;
    int nsteps;
};

#include "RTGI/Projection.fxh"
#include "RTGI/Normal.fxh"
#include "RTGI/RaySorting.fxh"
#include "RTGI/RayTracing.fxh"
#include "RTGI\Denoise.fxh"
#include "RTGI\Random.fxh"

RTInputs init(VSOUT i)
{
	RTInputs o;
	o.pos = Projection::uv_to_proj(i.uv);
	o.eyedir = -normalize(o.pos);
	o.normal = tex2D(sGBufferTex, i.uv).xyz;

    o.normal += tex2D(sGBufferTex, i.uv + float2(1,1) * qUINT::PIXEL_SIZE * 0.75).xyz;
    o.normal += tex2D(sGBufferTex, i.uv + float2(-1,1) * qUINT::PIXEL_SIZE * 0.75).xyz;
    o.normal += tex2D(sGBufferTex, i.uv + float2(1,-1) * qUINT::PIXEL_SIZE * 0.75).xyz;
    o.normal += tex2D(sGBufferTex, i.uv + float2(-1,-1) * qUINT::PIXEL_SIZE * 0.75).xyz;    
    o.normal = normalize(o.normal);

    //Todo investigate if sufficiently high frequency noise can be generated from
    //permuted golden weyl quasirandom sequences
	o.jitter =                 tex2Dfetch(sJitterTex,  i.vpos.xy 	   % 32u).xyz;
    o.jitter = frac(o.jitter + tex2Dfetch(sJitterTex, (i.vpos.xy / 32) % 32u).xyz);  

	o.tangent_base = Normal::base_from_vector(o.normal);

    o.nrays   = RT_DO_RENDER ? 2 : RT_RAY_AMOUNT;
    o.nsteps  = RT_DO_RENDER ? 200 : RT_RAY_STEPS;
	return o;
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

float3 dither(in VSOUT i)
{
    const float2 magicdot = float2(0.75487766624669276, 0.569840290998);
    const float3 magicadd = float3(0, 0.025, 0.0125) * dot(magicdot, 1);

    const int bit_depth = 8; //TODO: add BUFFER_COLOR_DEPTH once it works
    const float lsb = exp2(bit_depth) - 1;

    float3 dither = frac(dot(i.vpos.xy, magicdot) + magicadd);
    dither /= lsb;
    
    return dither;
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

float fade_distance(in VSOUT i)
{
    float distance = saturate(length(Projection::uv_to_proj(i.uv)) / RESHADE_DEPTH_LINEARIZATION_FAR_PLANE);
    float fade;
#if(FADEOUT_MODE == 1)
    fade = saturate((RT_FADE_DEPTH.y - distance) / (RT_FADE_DEPTH.y - RT_FADE_DEPTH.x + 1e-6));
#elif(FADEOUT_MODE == 2)
    fade = saturate((RT_FADE_DEPTH.y - distance) / (RT_FADE_DEPTH.y - RT_FADE_DEPTH.x + 1e-6));
    fade *= fade; 
    fade *= fade;
#elif(FADEOUT_MODE == 3)
    fade = exp(-distance * rcp(RT_FADE_DEPTH.y * RT_FADE_DEPTH.y * 8.0 + 0.001) + RT_FADE_DEPTH.x);
#else
    fade = smoothstep(RT_FADE_DEPTH.y+0.001, RT_FADE_DEPTH.x, distance);  
#endif

    return fade;    
}

/*=============================================================================
	Pixel Shaders
=============================================================================*/

void PS_InputSetup(in VSOUT i, out float4 color : SV_Target0, out float depth : SV_Target1, out float4 gbuffer : SV_Target2)
{ 
    depth = qUINT::linear_depth(i.uv);
    color = tex2D(qUINT::sBackBufferTex, i.uv);
    color *= saturate(999.0 - depth * 1000.0); //mask sky
    depth = Projection::depth_to_z(depth);
    gbuffer.xyz = Normal::normal_from_depth(i);
    gbuffer.w = depth;
}

void PS_InputSetupHalf1(in VSOUT i, out float4 gbuffer : SV_Target0)
{ 
    gbuffer.xyz = Normal::normal_from_depth(i);
    gbuffer.w = Projection::depth_to_z(qUINT::linear_depth(i.uv));
}

void PS_InputSetupHalf2(in VSOUT i, out float4 color : SV_Target0, out float depth : SV_Target1)
{
   	float4 texels; //TODO: replace with gather()
    texels.x = qUINT::linear_depth(i.uv + float2( 0.5, 0.5) * qUINT::PIXEL_SIZE);
    texels.y = qUINT::linear_depth(i.uv + float2(-0.5, 0.5) * qUINT::PIXEL_SIZE);
    texels.z = qUINT::linear_depth(i.uv + float2( 0.5,-0.5) * qUINT::PIXEL_SIZE);
    texels.w = qUINT::linear_depth(i.uv + float2(-0.5,-0.5) * qUINT::PIXEL_SIZE);
    float   avg = dot(texels, 0.25);
    float4 diff = saturate(1.0 - avg / texels);    
    depth = dot(texels, diff);
    depth /= dot(diff, 1); 

    color = tex2D(qUINT::sBackBufferTex, i.uv);
    color *= saturate(999.0 - depth * 1000.0); //mask sky, also multiply alpha

    depth = Projection::depth_to_z(depth);
}

void PS_Smoothnormals(in VSOUT i, out float4 gbuffer : SV_Target0)
{ 
    const float max_n_n = 0.63;
    const float max_v_s = 0.65;
    const float max_c_p = 0.5;
    const float searchsize = 0.0125;
    const int dirs = 5;

    float4 gbuf_center = tex2D(sRTGITempTex1, i.uv);

    float3 n_center = gbuf_center.xyz;
    float3 p_center = Projection::uv_to_proj(i.uv, gbuf_center.w);
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
            offs += length(qUINT::PIXEL_SIZE);

            float2 uv = i.uv + dir * offs * qUINT::ASPECT_RATIO;            
            if(!all(saturate(uv - uv*uv))) break;

            float4 gbuf = tex2Dlod(sRTGITempTex1, float4(uv, 0, 0));
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
    gbuffer = float4(n_sum, gbuf_center.w);
}

//1 -> 2
void PS_Copy_1_to_2(in VSOUT i, out float4 o0 : SV_Target0, out float4 o1 : SV_Target1)
{
	o0 = tex2D(sGITex1, i.uv);
	o1 = tex2D(sGBufferTex1, i.uv);
}

//0 -> 1
void PS_Copy_0_to_1(in VSOUT i, out float4 o0 : SV_Target0, out float4 o1 : SV_Target1)
{
	o0 = tex2D(sGITex0, i.uv);
	o1 = tex2D(sGBufferTex, i.uv);
}

//update 0
void PS_RTMain(in VSOUT i, out float4 o : SV_Target0)
{   
    RTInputs parameters = init(i);
	//bias position a bit to fix precision issues
	parameters.pos *= 0.999;
	parameters.pos += parameters.normal * Projection::z_to_depth(parameters.pos.z);
	SampleSet sampleset = ray_sorting(i, qUINT::FRAME_COUNT, parameters.jitter.x); 

#if MATERIAL_TYPE == 1
    float3 specular_color = tex2D(qUINT::sBackBufferTex, i.uv).rgb; 
    specular_color = lerp(dot(specular_color, 0.333), specular_color, 0.666)  * 2.0;
    specular_color *= RT_SPECULAR;
#endif 

    o = 0;

    [loop]
    for(int r = 0; r < 0 + parameters.nrays; r++)
    {
        RayTracing::RayDesc ray;

        [branch]
        if(RT_DO_RENDER) 
        {
            parameters.jitter = Random::goldenweyl3((qUINT::FRAME_COUNT % uint(MONTECARLO_MAX_STACK_SIZE)) * parameters.nrays + r, parameters.jitter);
            sampleset.index = parameters.jitter.x;
            sincos(parameters.jitter.z * 3.1415927 * 2, sampleset.dir_xy.y, sampleset.dir_xy.x);
        }

#if MATERIAL_TYPE == 0
        //lambert cosine distribution without TBN reorientation
        ray.dir.z = (r + sampleset.index) / parameters.nrays * 2.0 - 1.0; 
        ray.dir.xy = sampleset.dir_xy * sqrt(1.0 - ray.dir.z * ray.dir.z); //build sphere
        ray.dir = normalize(ray.dir + parameters.normal);

#elif MATERIAL_TYPE == 1
        float alpha = RT_ROUGHNESS * RT_ROUGHNESS; //isotropic  
       
        //"random" point on disc - do I have to do sqrt() ?
        float2 uniform_disc = sqrt((r + sampleset.index) / parameters.nrays) * sampleset.dir_xy;
        float3 v = mul(parameters.eyedir, transpose(parameters.tangent_base)); //v to tangent space
        float3 h = ggx_vndf(uniform_disc, alpha.xx, v);
        float3 l = reflect(-v, h);

        //single scatter lobe
        float3 brdf = ggx_g2_g1(l, v , alpha.xx); //if l.z > 0 is checked later
        brdf = l.z < 1e-7 ? 0 : brdf; //test?
        float vdoth = dot(parameters.eyedir, h);
        brdf *= schlick_fresnel(vdoth, specular_color);

        ray.dir = mul(l, parameters.tangent_base); //l from tangent to projection
#endif
        //advance to next ray dir
        sampleset.dir_xy = mul(sampleset.dir_xy, sampleset.nextdir);

        if (dot(ray.dir, parameters.normal) < 0.0)
            continue; 

        ray.maxlen = lerp(RT_SAMPLE_RADIUS * RT_SAMPLE_RADIUS, RT_SAMPLE_RADIUS * RT_SAMPLE_RADIUS * 100.0, saturate(Projection::z_to_depth(parameters.pos.z) * RT_SAMPLE_RADIUS_FAR));
        ray.maxlen = min(ray.maxlen, RESHADE_DEPTH_LINEARIZATION_FAR_PLANE);
        ray.steplen = (ray.maxlen / parameters.nsteps);
        ray.steplen *= rsqrt(saturate(1.0 - ray.dir.z * ray.dir.z) + 0.001);
        ray.currlen = ray.steplen * frac(parameters.jitter.y + r * 1.6180339887); //break up step size per ray
        
        float intersected = RayTracing::compute_intersection(ray, parameters, i, RT_USE_FALLBACK);        
        o.w += intersected;

        if(RT_IL_AMOUNT * intersected == 0)
        {
#if IMAGEBASEDLIGHTING != 0
            float4 probe = tex2Dlod(sProbeTex, float4(ray.dir.xy * 0.5 + 0.5, 0, 0));  unpack_hdr(probe.rgb);
            o += probe * RT_IBL_AMOUT;
#endif
            continue;     
        }

#if MATERIAL_TYPE == 1
        //revert to fullres mips for sharper reflection at low roughness settings
        ray.width = lerp(-MIP_BIAS_IL, ray.width, smoothstep(0.05, 0.2, RT_ROUGHNESS));
#endif
        float3 albedo           = tex2Dlod(sColorTex,    float4(ray.uv, 0, ray.width + MIP_BIAS_IL)).rgb; unpack_hdr(albedo);        

#if INFINITE_BOUNCES != 0
        float3 nextbounce       = tex2Dlod(sRTGITempTex0,  float4(ray.uv, 0, 0)).rgb; unpack_hdr(nextbounce);            
        albedo += nextbounce * RT_IL_BOUNCE_WEIGHT;
#endif
        float3 intersect_normal = tex2Dlod(sGBufferTex,  float4(ray.uv, 0, 0)).xyz;
        float backface_check = saturate(dot(-intersect_normal, ray.dir) * 100.0);
        
        if(RT_BACKFACE_MIRROR)                             
            backface_check = lerp(backface_check, 1.0, 0.1);

        albedo *= backface_check;

#if MATERIAL_TYPE == 0
        o.rgb += albedo;   // * cos(theta) / pdf == 1 here for cosine weighted sampling  
#elif MATERIAL_TYPE == 1

        albedo *= brdf;
        albedo *= 10.0;

        o.rgb += albedo;
#endif   
    }

    o /= parameters.nrays; 

//temporal integration stuff

#define read_counter(tex) tex2Dfetch(tex, 0).w
#define store_counter(val) o.w = max(i.vpos.x, i.vpos.y) <= 1.0 ? val : o.w;

    if(!RT_DO_RENDER)
    {
    	store_counter(0);
    }
    else
    {
    	float counter = read_counter(sGITex1);
        float4 last_accumulated = tex2D(sGITex1, i.uv);
        unpack_hdr(last_accumulated.rgb);

        //accumulation with that method causes corruption with too high values
        //residual noise is gone with the filter anyways
        if(counter < MONTECARLO_MAX_STACK_SIZE) 
        {
            counter++;            
            o = lerp(last_accumulated, o, rcp(counter));            
        }
        else 
        {
            o = last_accumulated;
        }
        store_counter(counter);    	
    }

	pack_hdr(o.rgb);
}

void PS_Combine(in VSOUT i, out float4 o : SV_Target0)
{
	float4 gi[2], gbuf[2];
	gi[0] = tex2D(sGITex1, i.uv);
	gi[1] = tex2D(sGITex2, i.uv);
	gbuf[0] = tex2D(sGBufferTex1, i.uv);
	gbuf[1] = tex2D(sGBufferTex2, i.uv);

	float4 combined = tex2D(sGITex0, i.uv);
	float sumweight = 1.0;
	float4 gbuf_reference = tex2D(sGBufferTex, i.uv);

	[unroll]
	for(int j = 0; j < 2; j++)
	{
		float4 delta = abs(gbuf_reference - gbuf[j]);

		float normal_sensitivity = 2.0;
		float z_sensitivity = 1.0;

		//TODO: investigate corner cases, if this is actually useful
		float time_delta = qUINT::FRAME_TIME; 
		time_delta = max(time_delta, 1.0) / 16.7; //~1 for 60 fps, expected range
		delta /= time_delta;

		float d = dot(delta, float4(delta.xyz * normal_sensitivity, z_sensitivity)); //normal squared, depth linear
		float w = exp(-d);

		combined += gi[j] * w;
		sumweight += w;
	}
	combined /= sumweight;
	o = combined;
}

void PS_Filter0(in VSOUT i, out float4 o : SV_Target0)
{
    o = Denoise::filter(i, sRTGITempTex0, 0, RT_DO_RENDER - SKIP_FILTER * 2);
}
void PS_Filter1(in VSOUT i, out float4 o : SV_Target0)
{
    o = Denoise::filter(i, sRTGITempTex1, 1, RT_DO_RENDER - SKIP_FILTER * 2);
}
void PS_Filter2(in VSOUT i, out float4 o : SV_Target0)
{
    o = Denoise::filter(i, sRTGITempTex0, 2, RT_DO_RENDER - SKIP_FILTER * 2);
}
void PS_Filter3(in VSOUT i, out float4 o : SV_Target0)
{
    o = Denoise::filter(i, sRTGITempTex1, 3, RT_DO_RENDER - SKIP_FILTER * 2);
}

void PS_Blending(in VSOUT i, out float4 o : SV_Target0)
{
    float4 gi = tex2D(sRTGITempTex0, i.uv);
    float3 color = tex2D(qUINT::sBackBufferTex, i.uv).rgb;

    unpack_hdr(color);
    unpack_hdr(gi.rgb);

    color = RT_DEBUG_VIEW == 1 ? 1 : color; 

    float similarity = distance(normalize(color + 0.00001), normalize(gi.rgb + 0.00001));
	similarity = saturate(similarity * 3.0);
	gi.rgb = lerp(dot(gi.rgb, 0.3333), gi.rgb, saturate(similarity * 0.5 + 0.5));  
   
    float fade = fade_distance(i);  
    gi *= fade; 

#if SKYCOLOR_MODE != 0
 #if SKYCOLOR_MODE == 1
    float3 skycol = SKY_COLOR;
 #elif SKYCOLOR_MODE == 2
    float3 skycol = tex2Dfetch(sProbeTex, 0).rgb; //take topleft pixel of probe tex, outside of hemisphere range //tex2Dfetch(sSkyCol, 0).rgb;
    skycol = lerp(dot(skycol, 0.333), skycol, SKY_COLOR_SAT * 0.2);
 #elif SKYCOLOR_MODE == 3
    float3 skycol = tex2Dfetch(sProbeTex, 0).rgb * SKY_COLOR_TINT; //tex2Dfetch(sSkyCol, 0).rgb * SKY_COLOR_TINT;
    skycol = lerp(dot(skycol, 0.333), skycol, SKY_COLOR_SAT * 0.2);
 #endif
    skycol *= fade;

    color += lerp(color, dot(color, 0.333), saturate(1 - dot(color, 3.0))) * gi.rgb * RT_IL_AMOUNT * RT_IL_AMOUNT; //apply GI
    color = color / (1.0 + lerp(1.0, skycol, SKY_COLOR_AMBIENT_MIX) * gi.w * RT_AO_AMOUNT); //apply AO as occlusion of skycolor
    color = color * (1.0 + skycol * SKY_COLOR_AMT);
#else    
    color += lerp(color, dot(color, 0.333), saturate(1 - dot(color, 3.0))) * gi.rgb * RT_IL_AMOUNT * RT_IL_AMOUNT; //apply GI
    color = color / (1.0 + gi.w * RT_AO_AMOUNT);    
#endif

    pack_hdr(color.rgb);

    //dither a little bit as large scale lighting might exhibit banding
    color += dither(i);

    color = RT_DEBUG_VIEW == 2 ? tex2D(sGBufferTex, i.uv).xyz * float3(0.5, 0.5, -0.5) + 0.5 : color;
    o = float4(color, 1);
}

#if IMAGEBASEDLIGHTING != 0 || SKYCOLOR_MODE >= 2
void PS_Probe(in VSOUT i, out float4 o : SV_Target0)
{
    float3 n;
    n.xy = i.uv * 2.0 - 1.0;
    n.z  = sqrt(saturate(1.0 - dot(n.xy, n.xy)));

    bool probe = length(n.xy) < 1.3; //n.z > 1e-3; //padding

    uint2 kernel_spatial   = uint2(32 * qUINT::ASPECT_RATIO.yx);
    uint kernel_temporal   = 64;
    uint frame_num         = qUINT::FRAME_COUNT;
    float2 grid_increment   = rcp(kernel_spatial); //blocksize in % of screen
    float2 grid_start      = Random::goldenweyl2(frame_num % kernel_temporal) * grid_increment;
    float2 grid_pos        = grid_start;

    float4 probe_light = 0;
    float4 sky_light   = 0;

    float wsum = 0.00001;

    for(int x = 0; x < kernel_spatial.x; x++)
    {
        for(int y = 0; y < kernel_spatial.y; y++)
        {
            float4 tapg = tex2Dlod(sGBufferTex, float4(grid_pos, 0, 0));
            float4 tapc = tex2Dlod(sColorTex, float4(grid_pos, 0, 2));

            tapg.a = Projection::z_to_depth(tapg.a);
            
            float similarity = saturate(dot(tapg.xyz, -n)); //similarity *= similarity;           
            bool issky = tapg.a > 0.999;

            float4 tapc_o = tapc;
            unpack_hdr(tapc.rgb);
            tapc.rgb = normalize(tapc_o.rgb) * length(tapc.rgb + 0.01);

            sky_light   += float4(tapc_o.rgb, 1) * issky;
            probe_light += float4(tapc.rgb * similarity  * !issky, similarity * !issky) * tapg.a * probe;//float4(tapc.rgb * similarity * !issky, 1) * tapg.a * 0.01 * probe;    
            wsum += tapg.a * probe;
            grid_pos.y += grid_increment.y;          
        }
        grid_pos.y = grid_start.y;
        grid_pos.x += grid_increment.x;
    }
    
    //probe_light.rgb /= probe_light.a + 1e-3;
    probe_light /= wsum;
    sky_light.rgb   /= sky_light.a   + 1e-3;

    pack_hdr(probe_light.rgb); 

    float4 prev_probe = tex2D(sProbeTexPrev, i.uv); 

    o = 0;
    if(probe) //process central area with hemispherical probe light
    {
        o = lerp(prev_probe, probe_light, 0.02);  
        o = saturate(o);        
    }
    else
    {
        bool skydetectedthisframe = sky_light.w > 0.000001;
        bool skydetectedatall = prev_probe.w; 

        float h = 0;

        if(skydetectedthisframe)
            h = skydetectedatall ? saturate(0.1 * 0.01 * qUINT::FRAME_TIME) : 1; 

        o.rgb = lerp(prev_probe.rgb, sky_light.rgb, h);
        o.w = skydetectedthisframe || skydetectedatall;
    }
}

void PS_CopyProbe(in VSOUT i, out float4 o : SV_Target0)
{
    o = tex2D(sProbeTex, i.uv);
}
#endif

/*=============================================================================
	Techniques
=============================================================================*/

technique RTGlobalIllumination
< ui_tooltip = "              >> qUINT::RTGI 0.21 <<\n\n"
               "         EARLY ACCESS -- PATREON ONLY\n"
               "Official versions only via patreon.com/mcflypg\n"
               "\nRTGI is written by Marty McFly / Pascal Gilcher\n"
               "Early access, featureset might be subject to change"; >
{
#if IMAGEBASEDLIGHTING != 0 || SKYCOLOR_MODE >= 2
    pass
    {
        VertexShader = VS_RT;
        PixelShader  = PS_Probe;
        RenderTarget = ProbeTex;
    } 
    pass
    {
        VertexShader = VS_RT;
        PixelShader  = PS_CopyProbe;
        RenderTarget = ProbeTexPrev;
    }
#endif
	//Update history chain
	pass
    {
		VertexShader = VS_RT;
		PixelShader  = PS_Copy_1_to_2; //1 -> 2
		RenderTarget0 = GITex2; 
		RenderTarget1 = GBufferTex2; 
    }
    pass
    {
		VertexShader = VS_RT;
		PixelShader  = PS_Copy_0_to_1; //0 -> 1
		RenderTarget0 = GITex1;
		RenderTarget1 = GBufferTex1; 
    }
    //Create new inputs

#if SMOOTHNORMALS == 0
    #if HALFRES_INPUT == 0
    pass
    {
        VertexShader = VS_RT;
        PixelShader  = PS_InputSetup;
        RenderTarget0 = ColorTex;
        RenderTarget1 = ZTex;
        RenderTarget2 = GBufferTex;
    }
    #else
    pass
    {
        VertexShader = VS_RT;
        PixelShader  = PS_InputSetupHalf1;
        RenderTarget0 = GBufferTex;
    }
    pass
    {
        VertexShader = VS_RT;
        PixelShader  = PS_InputSetupHalf2;
        RenderTarget0 = ColorTex;
        RenderTarget1 = ZTex;
    }
    #endif
#else 
    #if HALFRES_INPUT == 0
    pass
    {
        VertexShader = VS_RT;
        PixelShader  = PS_InputSetup;
        RenderTarget0 = ColorTex;
        RenderTarget1 = ZTex;
        RenderTarget2 = RTGITempTex1;
    }
    #else
    pass
    {
        VertexShader = VS_RT;
        PixelShader  = PS_InputSetupHalf1;
        RenderTarget0 = RTGITempTex1;
    }
    pass
    {
        VertexShader = VS_RT;
        PixelShader  = PS_InputSetupHalf2;
        RenderTarget0 = ColorTex;
        RenderTarget1 = ZTex;
    }
    #endif //halfres
    pass
    {
        VertexShader = VS_RT;
        PixelShader  = PS_Smoothnormals;
        RenderTarget0 = GBufferTex;
    }
#endif //smoothnormals


    pass
	{
		VertexShader = VS_RT;
		PixelShader  = PS_RTMain; //update 0
		RenderTarget0 = GITex0;      
	}
	//Combine temporal layers
	pass
	{
		VertexShader = VS_RT;
		PixelShader  = PS_Combine;
		RenderTarget0 = RTGITempTex0;
	}
	//Filter
	pass
    {
        VertexShader = VS_RT;
        PixelShader  = PS_Filter0;
        RenderTarget0 = RTGITempTex1;
    }
    pass
    {
        VertexShader = VS_RT;
        PixelShader  = PS_Filter1;
        RenderTarget0 = RTGITempTex0;
    } 
    pass
    {
        VertexShader = VS_RT;
        PixelShader  = PS_Filter2;
        RenderTarget0 = RTGITempTex1;
    } 
    pass
    {
        VertexShader = VS_RT;
        PixelShader  = PS_Filter3;
        RenderTarget = RTGITempTex0;
    }   
    //Blend
    pass
	{
		VertexShader = VS_RT;
        PixelShader  = PS_Blending;
	}
}