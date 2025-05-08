#version 440 core

layout(location = 0) in vec4 positionRadius;
layout(location = 1) in vec3 color;
layout(location = 2) in vec3 constants;

layout(location = 0) out vec2 fTex;
layout(location = 1) flat out vec3 relativePosition;
layout(location = 2) flat out vec4 inputData;
layout(location = 3) flat out vec3 lightColor;
layout(location = 4) flat out vec3 lightConstants;

#include "camera"

struct PointLight
{
	vec3  position;
	vec3  color;
	vec3  falloff;
	float radius;
};


// TODO: Do performance comparisons between the square and octagon
/*
vec2 positions[] = {
	vec2(-1.0f, -1.0f), vec2( 1.0f, -1.0f),
	vec2(-1.0f,  1.0f), vec2( 1.0f,  1.0f)
};
*/

const float ratio = 1.f / (1.f + sqrt(2));

vec2 positions[] = 
{
	vec2(0.f, 0.f), // center
	vec2(   1.f,  ratio), vec2( ratio,    1.f),
	vec2(-ratio,    1.f), vec2(  -1.f,  ratio),
	vec2(  -1.f, -ratio), vec2(-ratio,   -1.f), 
	vec2( ratio,   -1.f), vec2(   1.f, -ratio), 
	vec2(   1.f,  ratio) // Needed to close the loop
};

uniform vec3 cameraForward;
uniform vec3 cameraPosition;

void main()
{
	float radius = positionRadius.w;
	vec3 position = positionRadius.xyz;
	
	// Multiplying fTex by higher ratios and not adjusting it later leads to some weird stuff
	fTex = positions[gl_VertexID % 10].xy * radius * 1.5f;
	inputData = positionRadius;
	lightColor = color.xyz;
	lightConstants = constants;
	
	float cameraDistance = length(cameraPosition - position);
	vec3 cameraDelta = normalize(cameraPosition - position);
	vec3 shiftedPosition = position;
	
	// If the light source is behind the camera, move it towards(and past) the near plane
	if (dot(cameraDelta, cameraForward) > 0)
	{
		// TODO: Integrate near plane cutoff for the constant
		shiftedPosition += (dot(cameraDelta, cameraForward) * cameraDistance * 1.1f + 0.1f) * cameraForward;
	}
	//else
	{
		vec3 adjusted = vec3(fTex, 0) + (View * vec4(shiftedPosition, 1)).xyz;
		gl_Position = Projection * vec4(adjusted, 1);
	}
	relativePosition = (View * vec4(shiftedPosition, 1)).xyz;
}