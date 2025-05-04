#version 440 core

layout(location = 0) in vec4 positionRadius;
layout(location = 1) in vec3 color;

layout(location = 0) out vec2 fTex;
layout(location = 1) flat out vec3 relativePosition;
layout(location = 2) flat out vec4 inputData;
layout(location = 3) flat out vec3 lightColor;

#include "camera"

vec2 positions[] = {
	vec2(-1.0f, -1.0f), vec2( 1.0f, -1.0f),
	vec2(-1.0f,  1.0f), vec2( 1.0f,  1.0f)
};

/*
const float ratio = 1.f / (1.f + sqrt(2));

vec2 positions[] = 
{
	vec2(0.f, 0.f) // center
	vec2(   1.f,  ratio), vec2( ratio,    1.f),
	vec2(-ratio,    1.f), vec2(  -1.f,  ratio),
	vec2(  -1.f, -ratio), vec2(-ratio,   -1.f), 
	vec2( ratio,   -1.f), vec2(   1.f, -ratio), 
	vec2(   1.f,  ratio) // Needed to close the loop
};
*/


void main()
{
	fTex = positions[gl_VertexID % 4].xy * 1.5f;
	inputData = positionRadius;
	lightColor = color.xyz;
	
	float radius = positionRadius.w;
	vec3 position = positionRadius.xyz;
	vec3 adjusted = vec3(fTex * radius, 0) + (View * vec4(position, 1)).xyz;
	relativePosition = (View * vec4(position, 1)).xyz;
	gl_Position = Projection * vec4(adjusted, 1);
}