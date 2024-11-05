#version 440 core

layout(location = 0) out vec2 fTex;
layout(location = 1) out float depth;

layout(std140) uniform Camera
{
	mat4 View;
	mat4 Projection;
};

vec2 positions[] = {
	vec2(-1.0f, -1.0f), vec2( 1.0f, -1.0f),
	vec2(-1.0f,  1.0f), vec2( 1.0f,  1.0f)
};

uniform vec3 position;
uniform float radius;

void main()
{
	fTex = positions[gl_VertexID % 4].xy;
	vec3 adjusted = vec3(fTex * radius, 0) + (View * vec4(position, 1)).xyz;
	gl_Position = Projection * vec4(adjusted, 1);
	depth = adjusted.z;
}