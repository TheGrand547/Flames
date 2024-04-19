#version 440 core

layout(location = 0) in vec3 vPos;
layout(location = 0) out vec4 fColor;

layout(std140) uniform Camera
{
	mat4 View;
	mat4 Projection;
};

vec4 colors[] = {
	vec4(1.f, 0.f, 0.f, 1.f), 
	vec4(0.f, 1.f, 0.f, 1.f),
	vec4(0.f, 0.f, 1.f, 1.f)
};


void main()
{
	gl_Position = Projection * View * vec4(vPos, 1.0);
	fColor = colors[int(gl_VertexID / 3) % 3];
}