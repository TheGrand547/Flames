#version 440 core
#include "camera"

layout(location = 0) in vec4 vPos;

layout(location = 0) flat out vec3 fPos;
layout(location = 1) flat out float radius;
layout(location = 2) out vec3 relativePosition;
layout(location = 3) out vec2 fTex;

vec2 positions[] = {
	vec2(-1.0f, -1.0f), vec2( 1.0f, -1.0f),
	vec2(-1.0f,  1.0f), vec2( 1.0f,  1.0f)
};

void main()
{
	radius = vPos.w;
	vec3 pos = vPos.xyz;
	// Multiplying fTex by higher ratios and not adjusting it later leads to some weird stuff
	fTex = positions[gl_VertexID % 4].xy * radius * 1.5f;
	
	vec3 adjusted = vec3(fTex, 0) + (View * vec4(pos, 1)).xyz;
	gl_Position = Projection * vec4(adjusted, 1);
	relativePosition = (View * vec4(pos, 1)).xyz;
}