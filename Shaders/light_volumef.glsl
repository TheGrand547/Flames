#version 440 core

layout(location = 0) in vec2 fTex;
layout(location = 1) in float depth;
layout(location = 2) flat in vec3 fPos;
layout(location = 3) flat in vec4 inputData;
layout(location = 4) flat in vec3 lightColor;

layout(location = 0) out vec4 fColor;

#include "camera"

const float PI = 1.0 / radians(180);

#include "lighting"

layout(location = 0) uniform sampler2D position;
layout(location = 1) uniform sampler2D normals;
layout(location = 2) uniform sampler2D colors;

void main()
{
	vec3 lightPosition = inputData.xyz;
	float radius = inputData.w;
	
	// From https://github.com/paroj/gltut/blob/master/Tut%2013%20Impostors/data/GeomImpostor.frag
	vec3 adjusted = vec3(fTex * radius, 0.0) + fPos;
	vec3 ray = normalize(adjusted);
	
	float B = 2.0 * dot(ray, -fPos);
	float C = dot(fPos, fPos) - (radius * radius);
	
	float det = (B * B) - (4 * C);
	if(det < 0.0)
		discard;
		
	float sqrtDet = sqrt(det);
	float posT = (-B + sqrtDet)/2;
	float negT = (-B - sqrtDet)/2;
	
	// To get near/Far simply replace min/max with min/max respectively
	float intersectT = max(posT, negT);
	
	// Outputs
	vec3 finalPos = ray * intersectT;
	vec3 finalNorm = normalize(finalPos - fPos);
	
	// I don't think this is necessary since we only care about the lighting in this stage
	/*
	// TODO: Figure out how to avoid matrix multiplication in fragment shader if at all possible
	vec4 clipPos = Projection * vec4(finalPos, 1.0);
	float ndcDepth = clipPos.z / clipPos.w;
	gl_FragDepth = ((gl_DepthRange.diff * ndcDepth) + gl_DepthRange.near + gl_DepthRange.far) / 2.0;
	*/
	
	vec2 textureCoord = gl_FragCoord.xy;
	
	// Get the position of the texture to sample
	vec4 rawPosition = texture(position, textureCoord);
	
	// Don't light things if they're too far away
	if (rawPosition.w > intersectT)
		discard;
	vec3 worldPosition = rawPosition.xyz;
	vec3 viewDirection = normalize(View[3].xyz - worldPosition);
	vec3 normal = texture(normals, textureCoord).xyz;
	
	vec3 lightEffect = PointLight(lightPosition, lightColor, normal, worldPosition, viewDirection);
	fColor = vec4(lightEffect, 1.f);
	//fColor = vec4(1, 0, 0.5, 1);
}