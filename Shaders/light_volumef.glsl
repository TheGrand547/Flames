#version 440 core

layout(location = 0) in vec2 fTex;
layout(location = 1) flat in vec3 relativePosition;
layout(location = 2) flat in vec4 inputData;
layout(location = 3) flat in vec3 lightColor;
layout(location = 4) flat in vec3 lightConstants;

layout(location = 0) out vec4 fColor;

#include "camera"

#include "lighting"

layout(location = 0) uniform sampler2D gPosition;
layout(location = 1) uniform sampler2D gNormal;

void main()
{
	vec3 lightPosition = inputData.xyz;
	float radius = inputData.w;
	
	// From https://github.com/paroj/gltut/blob/master/Tut%2013%20Impostors/data/GeomImpostor.frag
	vec3 adjusted = vec3(fTex, 0.0) + relativePosition;
	vec3 ray = normalize(adjusted);
	
	float B = 2.0 * dot(ray, -relativePosition);
	float C = dot(relativePosition, relativePosition) - (radius * radius);
	
	float det = (B * B) - (4 * C);
	if(det < 0.0)
		discard;
		
	float sqrtDet = sqrt(det);
	float posT = (-B + sqrtDet)/2;
	float negT = (-B - sqrtDet)/2;
	
	float intersectT = max(posT, negT);
	vec3 cameraPos = ray * intersectT;
	vec4 laDeDa = Projection * vec4(cameraPos, 1.f);
	laDeDa /= laDeDa.w;
	gl_FragDepth = laDeDa.z / 2 + 0.5;
	// Something is very fishy and wrong here but I don't know what
	
	vec2 textureCoord = gl_FragCoord.xy / 1000;
	
	// Get the position of the texture to sample
	vec4 rawPosition = texture(gPosition, textureCoord);
	
	// Don't light things if they're too far away
	vec3 worldPosition = rawPosition.xyz;
	vec3 viewDirection = normalize(View[3].xyz - worldPosition);
	vec3 normal = texture(gNormal, textureCoord).xyz;
	if (length(worldPosition - lightPosition) > radius)
		discard;
	
	vec3 lightEffect = PointLightConstants(lightPosition, lightColor, lightConstants, normal, worldPosition, viewDirection);
	fColor = vec4(lightEffect, 1.f);
}