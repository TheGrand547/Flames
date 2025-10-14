#version 440 core
#include "camera"

layout(location = 0) in vec3 vPos;
//layout(location = 1) in vec3 vNorm;

layout(location = 0) flat out vec3 fPos;
layout(location = 1) out vec3 fNorm;
layout(location = 2) out vec3 relativePosition;
layout(location = 3) out vec2 fTex;

vec2 positions[] = {
	vec2(-1.0f, -1.0f), vec2( 1.0f, -1.0f),
	vec2(-1.0f,  1.0f), vec2( 1.0f,  1.0f)
};

void main()
{
	const float radius = 0.5f;
	
	// Multiplying fTex by higher ratios and not adjusting it later leads to some weird stuff
	fTex = positions[gl_VertexID % 4].xy * radius * 1.5f;
	
	vec3 adjusted = vec3(fTex, 0) + (View * vec4(vPos, 1)).xyz;
	gl_Position = Projection * vec4(adjusted, 1);
	relativePosition = (View * vec4(vPos, 1)).xyz;
}