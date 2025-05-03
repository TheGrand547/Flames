#version 440 core

layout(location = 0) in vec4 positionRadius;
layout(location = 1) in vec3 color;

layout(location = 0) out vec2 fTex;
layout(location = 1) out float depth;
layout(location = 2) flat out vec3 fPos;
layout(location = 3) flat out vec4 inputData;
layout(location = 4) flat out vec3 lightColor;

#include "camera"

vec2 positions[] = {
	vec2(-1.0f, -1.0f), vec2( 1.0f, -1.0f),
	vec2(-1.0f,  1.0f), vec2( 1.0f,  1.0f)
};

void main()
{
	fTex = positions[gl_VertexID % 4].xy * 1.5f;
	inputData = positionRadius;
	lightColor = color.xyz;
	
	float radius = positionRadius.w;
	vec3 position = positionRadius.xyz;
	vec3 adjusted = vec3(fTex * radius, 0) + (View * vec4(position, 1)).xyz;
	fPos = (View * vec4(position, 1)).xyz;
	gl_Position = Projection * vec4(adjusted, 1);
	depth = adjusted.z;
}